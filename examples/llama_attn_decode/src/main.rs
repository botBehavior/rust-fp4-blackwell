/*
 * Track A — Llama decode kernel #3: flash-attention-decode (single query token), GQA.
 * One block per query head h. kv head = h / (n_heads/n_kv_heads).
 *   scores[t] = (Q[h] . K[t,kvh]) / sqrt(head_dim);  softmax over t;  out[h] = sum_t p[t] V[t,kvh]
 * Block-reduced max + exp-sum (warp shuffle + shared partials). CPU-parity gated.
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

#[cuda_module]
mod kernels {
    use super::*;

    const MAXSEQ: usize = 4096;
    const HD: usize = 64; // head_dim

    #[kernel]
    pub fn attn_decode(
        n_heads: u32,
        n_kv_heads: u32,
        seq: u32,
        q: &[f32],  // [n_heads*HD]
        kc: &[f32], // [seq, n_kv_heads, HD]
        vc: &[f32], // [seq, n_kv_heads, HD]
        mut out: DisjointSlice<f32>, // [n_heads*HD]
    ) {
        static mut SCORES: SharedArray<f32, MAXSEQ> = SharedArray::UNINIT;
        static mut QSH: SharedArray<f32, HD> = SharedArray::UNINIT;
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;

        let h = thread::blockIdx_x() as usize;
        if h >= n_heads as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let lane = tid % 32;
        let warp = tid / 32;
        let nwarps = bdim / 32;
        let nkv = n_kv_heads as usize;
        let group = n_heads as usize / nkv;
        let kvh = h / group;
        let s = seq as usize;
        let scale = 1.0f32 / (HD as f32).sqrt();

        if tid < HD {
            unsafe { QSH[tid] = q[h * HD + tid] };
        }
        thread::sync_threads();

        // Phase 1: scores
        let mut t = tid;
        while t < s {
            let kbase = (t * nkv + kvh) * HD;
            let mut d = 0.0f32;
            let mut i = 0;
            while i < HD {
                d += unsafe { QSH[i] } * kc[kbase + i];
                i += 1;
            }
            unsafe { SCORES[t] = d * scale };
            t += bdim;
        }
        thread::sync_threads();

        // Phase 2: block max
        let mut m = -1e30f32;
        let mut t = tid;
        while t < s {
            m = m.max(unsafe { SCORES[t] });
            t += bdim;
        }
        m = m.max(warp::shuffle_xor_f32(m, 16));
        m = m.max(warp::shuffle_xor_f32(m, 8));
        m = m.max(warp::shuffle_xor_f32(m, 4));
        m = m.max(warp::shuffle_xor_f32(m, 2));
        m = m.max(warp::shuffle_xor_f32(m, 1));
        if lane == 0 {
            unsafe { RED[warp] = m };
        }
        thread::sync_threads();
        let mut gm = -1e30f32;
        let mut w = 0;
        while w < nwarps {
            gm = gm.max(unsafe { RED[w] });
            w += 1;
        }
        thread::sync_threads(); // RED is reused below

        // Phase 3: exp (in place) + sum
        let mut ls = 0.0f32;
        let mut t = tid;
        while t < s {
            let e = (unsafe { SCORES[t] } - gm).exp();
            unsafe { SCORES[t] = e };
            ls += e;
            t += bdim;
        }
        ls += warp::shuffle_xor_f32(ls, 16);
        ls += warp::shuffle_xor_f32(ls, 8);
        ls += warp::shuffle_xor_f32(ls, 4);
        ls += warp::shuffle_xor_f32(ls, 2);
        ls += warp::shuffle_xor_f32(ls, 1);
        if lane == 0 {
            unsafe { RED[warp] = ls };
        }
        thread::sync_threads();
        let mut l = 0.0f32;
        let mut w = 0;
        while w < nwarps {
            l += unsafe { RED[w] };
            w += 1;
        }
        let inv = 1.0f32 / l;
        thread::sync_threads(); // SCORES now holds exp weights, fully written

        // Phase 4: weighted sum of V
        let mut i = tid;
        while i < HD {
            let mut acc = 0.0f32;
            let mut t = 0;
            while t < s {
                acc += unsafe { SCORES[t] } * vc[(t * nkv + kvh) * HD + i];
                t += 1;
            }
            unsafe { *out.as_mut_ptr().add(h * HD + i) = acc * inv };
            i += bdim;
        }
    }
}

fn main() {
    const NH: usize = 32; // query heads
    const NKV: usize = 8; // kv heads (GQA group = 4)
    const HD: usize = 64;
    const SEQ: usize = 512; // cached context length

    let mut s = 0xabcd_1234u32;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s % 2001) as f32 - 1000.0) / 700.0
    };
    let q: Vec<f32> = (0..NH * HD).map(|_| rng()).collect();
    let kc: Vec<f32> = (0..SEQ * NKV * HD).map(|_| rng()).collect();
    let vc: Vec<f32> = (0..SEQ * NKV * HD).map(|_| rng()).collect();
    let scale = 1.0f32 / (HD as f32).sqrt();

    // cpu reference
    let mut oref = vec![0f32; NH * HD];
    let group = NH / NKV;
    for h in 0..NH {
        let kvh = h / group;
        let mut sc = vec![0f32; SEQ];
        let mut m = f32::MIN;
        for t in 0..SEQ {
            let mut d = 0f32;
            for i in 0..HD {
                d += q[h * HD + i] * kc[(t * NKV + kvh) * HD + i];
            }
            sc[t] = d * scale;
            if sc[t] > m {
                m = sc[t];
            }
        }
        let mut l = 0f32;
        for t in 0..SEQ {
            sc[t] = (sc[t] - m).exp();
            l += sc[t];
        }
        for i in 0..HD {
            let mut acc = 0f32;
            for t in 0..SEQ {
                acc += sc[t] * vc[(t * NKV + kvh) * HD + i];
            }
            oref[h * HD + i] = acc / l;
        }
    }

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let q_d = DeviceBuffer::from_host(&stream, &q).unwrap();
    let kc_d = DeviceBuffer::from_host(&stream, &kc).unwrap();
    let vc_d = DeviceBuffer::from_host(&stream, &vc).unwrap();
    let mut o_d = DeviceBuffer::<f32>::zeroed(&stream, NH * HD).unwrap();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig {
        grid_dim: (NH as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .attn_decode(
            &stream, cfg, NH as u32, NKV as u32, SEQ as u32, &q_d, &kc_d, &vc_d, &mut o_d,
        )
        .expect("launch");
    let o = o_d.to_host_vec(&stream).unwrap();

    let mut maxrel = 0f32;
    let mut bad = 0;
    for i in 0..NH * HD {
        let rel = (o[i] - oref[i]).abs() / oref[i].abs().max(1e-3);
        if rel > 1e-4 {
            bad += 1;
        }
        if rel > maxrel {
            maxrel = rel;
        }
    }
    println!("=== Llama attn-decode heads={NH} kv={NKV} head_dim={HD} seq={SEQ} ===");
    println!("o[0..4]  ={:?}", &o[..4]);
    println!("ref[0..4]={:?}", &oref[..4]);
    println!("max_rel_err={maxrel:.2e}, bad(>1e-4)={bad}/{}", NH * HD);
    if bad == 0 {
        println!("\u{2713} SUCCESS: attn-decode matches CPU reference");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
