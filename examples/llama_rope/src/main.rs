/*
 * Track A — Llama decode kernel #2: RoPE (rotary position embedding), rotate-half (HF/Llama).
 * In-place on a [num_heads * head_dim] vector for one token at a fixed position.
 * cos/sin are precomputed per-position (length head_dim/2) and passed in — exactly like a
 * real inference rotary cache, and keeps the kernel free of device transcendentals.
 *   j < d/2:  out[j]      = x[j]*cos_j - x[j+d/2]*sin_j
 *             out[j+d/2]  = x[j+d/2]*cos_j + x[j]*sin_j
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn rope(
        num_heads: u32,
        head_dim: u32,
        cos: &[f32], // len head_dim/2
        sin: &[f32], // len head_dim/2
        mut q: DisjointSlice<f32>,
    ) {
        let gid = thread::index_1d().get();
        let hd = head_dim as usize;
        let half = hd / 2;
        let total = num_heads as usize * half;
        if gid >= total {
            return;
        }
        let head = gid / half;
        let j = gid % half;
        let base = head * hd;
        let c = cos[j];
        let s = sin[j];
        let p = q.as_mut_ptr();
        let x0 = unsafe { *p.add(base + j) };
        let x1 = unsafe { *p.add(base + j + half) };
        unsafe {
            *p.add(base + j) = x0 * c - x1 * s;
            *p.add(base + j + half) = x1 * c + x0 * s;
        }
    }
}

fn main() {
    const HEADS: usize = 32; // Llama-1B query heads
    const HD: usize = 64; // head_dim
    const HALF: usize = HD / 2;
    const POS: f32 = 37.0;
    const THETA: f64 = 500000.0;

    let mut s = 0x1357_9bdfu32;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let q: Vec<f32> = (0..HEADS * HD)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 500.0)
        .collect();

    // rotary cache for this position
    let mut cos = vec![0f32; HALF];
    let mut sin = vec![0f32; HALF];
    for j in 0..HALF {
        let freq = 1.0 / THETA.powf(2.0 * j as f64 / HD as f64);
        let ang = POS as f64 * freq;
        cos[j] = ang.cos() as f32;
        sin[j] = ang.sin() as f32;
    }

    // cpu reference (rotate-half)
    let mut qref = q.clone();
    for h in 0..HEADS {
        let base = h * HD;
        for j in 0..HALF {
            let x0 = q[base + j];
            let x1 = q[base + j + HALF];
            qref[base + j] = x0 * cos[j] - x1 * sin[j];
            qref[base + j + HALF] = x1 * cos[j] + x0 * sin[j];
        }
    }

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let cos_d = DeviceBuffer::from_host(&stream, &cos).unwrap();
    let sin_d = DeviceBuffer::from_host(&stream, &sin).unwrap();
    let mut q_d = DeviceBuffer::from_host(&stream, &q).unwrap();
    let module = kernels::load(&ctx).expect("load");
    let total = (HEADS * HALF) as u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .rope(&stream, cfg, HEADS as u32, HD as u32, &cos_d, &sin_d, &mut q_d)
        .expect("launch");
    let out = q_d.to_host_vec(&stream).unwrap();

    let mut maxrel = 0f32;
    let mut bad = 0;
    for i in 0..HEADS * HD {
        let rel = (out[i] - qref[i]).abs() / qref[i].abs().max(1e-3);
        if rel > 1e-4 {
            bad += 1;
        }
        if rel > maxrel {
            maxrel = rel;
        }
    }
    println!("=== Llama RoPE heads={HEADS} head_dim={HD} pos={POS} ===");
    println!("out[0..4]={:?}", &out[..4]);
    println!("ref[0..4]={:?}", &qref[..4]);
    println!("max_rel_err={maxrel:.2e}, bad(>1e-4)={bad}/{}", HEADS * HD);
    if bad == 0 {
        println!("\u{2713} SUCCESS: RoPE matches CPU reference");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
