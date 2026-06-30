/*
 * Track A — Llama decode kernel #1: RMSNorm (single token).
 *   y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
 * One block; block-reduce sum-of-squares (warp shuffle + shared partials). CPU-parity gated.
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn rmsnorm(dim: u32, eps: f32, x: &[f32], weight: &[f32], mut y: DisjointSlice<f32>) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT; // one slot per warp (<=32)

        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let d = dim as usize;
        let lane = tid % 32;
        let warp = tid / 32;
        let nwarps = bdim / 32;

        // local sum of squares
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < d {
            let v = x[i];
            ss += v * v;
            i += bdim;
        }
        // warp reduce
        ss += warp::shuffle_xor_f32(ss, 16);
        ss += warp::shuffle_xor_f32(ss, 8);
        ss += warp::shuffle_xor_f32(ss, 4);
        ss += warp::shuffle_xor_f32(ss, 2);
        ss += warp::shuffle_xor_f32(ss, 1);
        if lane == 0 {
            unsafe { RED[warp] = ss };
        }
        thread::sync_threads();
        // every thread sums the per-warp partials -> total
        let mut total = 0.0f32;
        let mut wi = 0usize;
        while wi < nwarps {
            total += unsafe { RED[wi] };
            wi += 1;
        }
        let rms = 1.0f32 / (total / d as f32 + eps).sqrt();

        let mut i = tid;
        while i < d {
            unsafe { *y.as_mut_ptr().add(i) = x[i] * rms * weight[i] };
            i += bdim;
        }
    }
}

fn main() {
    const DIM: usize = 2048; // Llama-1B hidden dim
    const EPS: f32 = 1e-5;

    let mut s = 0x9e37_79b9u32;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let x: Vec<f32> = (0..DIM)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 500.0)
        .collect();
    let weight: Vec<f32> = (0..DIM).map(|_| 0.5 + (rng() % 1000) as f32 / 1000.0).collect();

    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (ss / DIM as f32 + EPS).sqrt();
    let yref: Vec<f32> = (0..DIM).map(|i| x[i] * rms * weight[i]).collect();

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let x_d = DeviceBuffer::from_host(&stream, &x).unwrap();
    let w_d = DeviceBuffer::from_host(&stream, &weight).unwrap();
    let mut y_d = DeviceBuffer::<f32>::zeroed(&stream, DIM).unwrap();
    let module = kernels::load(&ctx).expect("load");
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .rmsnorm(&stream, cfg, DIM as u32, EPS, &x_d, &w_d, &mut y_d)
        .expect("launch");
    let y = y_d.to_host_vec(&stream).unwrap();

    let mut maxrel = 0f32;
    let mut bad = 0;
    for i in 0..DIM {
        let rel = (y[i] - yref[i]).abs() / yref[i].abs().max(1e-3);
        if rel > 1e-4 {
            bad += 1;
        }
        if rel > maxrel {
            maxrel = rel;
        }
    }
    println!("=== Llama RMSNorm dim={DIM} ===");
    println!("y[0..4]  ={:?}", &y[..4]);
    println!("ref[0..4]={:?}", &yref[..4]);
    println!("max_rel_err={maxrel:.2e}, bad(>1e-4)={bad}/{DIM}");
    if bad == 0 {
        println!("\u{2713} SUCCESS: RMSNorm matches CPU reference");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
