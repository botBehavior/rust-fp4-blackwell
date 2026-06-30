/*
 * Track B — FP4 quantized GEMV for the DECODE path (bandwidth-bound), pure Rust / cuda-oxide.
 *   y[n] = sum_k dequant(W[n][k]) * x[k],  W = E2M1 fp4 (8/word), per-(row,32-block) f32 scale.
 * Optimizations vs naive: arithmetic E2M1 decode (no LUT/local-mem), 16-iter u32 stream
 * (high memory-level parallelism), and x staged once per block in shared memory (kills the
 * N-times re-read of the activation). CUDA-core dequant+FMA — the bandwidth-bound decode lane.
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

#[cuda_module]
mod kernels {
    use super::*;

    const MAXK: usize = 4096; // shared-mem activation capacity

    // Arithmetic E2M1 decode — no memory LUT (a const array lowers to local memory).
    // {0,.5,1,1.5,2,3,4,6}: e==0 -> 0.5*m ; else 2^(e-1)*(1+0.5*m).
    #[inline(always)]
    fn fp4_val(code: u32) -> f32 {
        let e = (code >> 1) & 3;
        let m = (code & 1) as f32;
        let mag = if e == 0 {
            0.5 * m
        } else {
            let base = (1u32 << (e - 1)) as f32;
            base + base * 0.5 * m
        };
        if (code & 8) != 0 { -mag } else { mag }
    }

    #[kernel]
    pub fn fp4_gemv(
        n: u32,
        k: u32,
        w: &[u32],      // packed E2M1, row-major, k/8 words per row
        scales: &[f32], // per (row, 32-block)
        x: &[f32],      // activation, len k
        mut y: DisjointSlice<f32>,
    ) {
        static mut XS: SharedArray<f32, MAXK> = SharedArray::UNINIT;

        let kk = k as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;

        // stage activation into shared memory once per block (reused by all rows in the block)
        let mut i = tid;
        while i < kk {
            unsafe { XS[i] = x[i] };
            i += bdim;
        }
        thread::sync_threads();

        let lane = tid % 32;
        let row = thread::index_1d().get() / 32; // one warp per row
        if row >= n as usize {
            return;
        }
        let kdiv8 = kk / 8;
        let wbase = row * kdiv8;
        let sbase = row * (kk / 32);

        let mut acc = 0.0f32;
        let mut u = lane;
        while u < kdiv8 {
            let word = w[wbase + u]; // coalesced across the warp, 16 iters => high MLP
            let base_k = u * 8;
            let sc = scales[sbase + base_k / 32];
            let mut blk = 0.0f32;
            let mut p = 0usize;
            while p < 8 {
                let code = (word >> (p * 4)) & 0xF;
                blk += fp4_val(code) * unsafe { XS[base_k + p] };
                p += 1;
            }
            acc += sc * blk;
            u += 32;
        }

        acc += warp::shuffle_xor_f32(acc, 16);
        acc += warp::shuffle_xor_f32(acc, 8);
        acc += warp::shuffle_xor_f32(acc, 4);
        acc += warp::shuffle_xor_f32(acc, 2);
        acc += warp::shuffle_xor_f32(acc, 1);

        if lane == 0 {
            unsafe {
                *y.as_mut_ptr().add(row) = acc;
            }
        }
    }
}

fn main() {
    const N: usize = 4096;
    const K: usize = 4096;
    const G: usize = 32;
    let blocks = K / G;

    let mut s: u32 = 0x1234_5678;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };

    let mut codes = vec![0u8; N * K];
    for c in codes.iter_mut() {
        *c = (rng() & 0xF) as u8;
    }
    let mut w = vec![0u32; N * K / 8];
    for (i, word) in w.iter_mut().enumerate() {
        let mut v = 0u32;
        for p in 0..8 {
            v |= (codes[i * 8 + p] as u32) << (p * 4);
        }
        *word = v;
    }
    let mut scales = vec![0f32; N * blocks];
    for v in scales.iter_mut() {
        *v = 0.5 + (rng() % 8) as f32 * 0.25;
    }
    let mut x = vec![0f32; K];
    for v in x.iter_mut() {
        *v = ((rng() % 17) as f32 - 8.0) * 0.1;
    }

    let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let fp4 = |c: u8| {
        let m = mags[(c & 7) as usize];
        if c & 8 != 0 { -m } else { m }
    };
    let mut yref = vec![0f32; N];
    for nrow in 0..N {
        let mut acc = 0f32;
        for kk in 0..K {
            acc += fp4(codes[nrow * K + kk]) * scales[nrow * blocks + kk / G] * x[kk];
        }
        yref[nrow] = acc;
    }

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let w_d = DeviceBuffer::from_host(&stream, &w).unwrap();
    let sc_d = DeviceBuffer::from_host(&stream, &scales).unwrap();
    let x_d = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut y_d = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let module = kernels::load(&ctx).expect("load");

    let cfg = LaunchConfig {
        grid_dim: ((N / 8) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    module
        .fp4_gemv(&stream, cfg, N as u32, K as u32, &w_d, &sc_d, &x_d, &mut y_d)
        .expect("launch");
    let y = y_d.to_host_vec(&stream).unwrap();
    let mut maxrel = 0f32;
    let mut bad = 0;
    for nrow in 0..N {
        let d = (y[nrow] - yref[nrow]).abs();
        let rel = d / yref[nrow].abs().max(1e-3);
        if rel > 1e-2 {
            bad += 1;
        }
        if rel > maxrel {
            maxrel = rel;
        }
    }
    println!("=== cuda-oxide FP4 GEMV (decode) {N}x{K}, group={G} ===");
    println!("correctness: max_rel_err={maxrel:.2e}, bad(>1e-2)={bad}/{N}");

    stream.synchronize().unwrap();
    let iters = 200u32;
    let start = stream
        .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .unwrap();
    for _ in 0..iters {
        module
            .fp4_gemv(&stream, cfg, N as u32, K as u32, &w_d, &sc_d, &x_d, &mut y_d)
            .unwrap();
    }
    let end = stream
        .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    let ms = start.elapsed_ms(&end).unwrap() / iters as f32;
    let wbytes = (N * K / 2) as f64; // fp4 weights — the dominant, FP4-shrunk traffic
    let gbps = wbytes / (ms as f64 * 1e-3) / 1e9;
    println!(
        "perf: {:.4} ms/iter, weights {:.1} GB/s ({:.1}% of 896 ceiling)",
        ms,
        gbps,
        gbps / 896.0 * 100.0
    );

    if bad == 0 {
        println!("\u{2713} SUCCESS: FP4 GEMV correct + measured");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
