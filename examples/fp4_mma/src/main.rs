/*
 * Track B — native FP4 block-scaled tensor-core MMA in pure Rust via cuda-oxide.
 * Single-warp MXFP4 m16n8k64, all-ones validation:
 *   A = all 1.0 (E2M1 = 0x2), B = all 1.0, scale = 1.0 (ue8m0 = 0x7F)
 *   => every D element must equal sum_{k=0..63} 1*1 = 64.0  (permutation-invariant)
 *
 * REQUIRES `--arch sm_120a`: plain sm_120 rejects mma...block_scale.
 * ptx_asm! allows at most one `out`, so the MMA + 4 stores live in one memory-only
 * asm block writing through a per-lane pointer; outputs go to global memory.
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fp4_mma_allones(mut out: DisjointSlice<f32>) {
        let lane = thread::index_1d().get();
        // Each lane writes its 4 accumulator regs to 4 contiguous f32 (disjoint across lanes).
        let p = unsafe { out.as_mut_ptr().add(lane * 4) };

        let a0: u32 = 0x2222_2222; // eight E2M1 1.0 values packed per b32
        let a1: u32 = 0x2222_2222;
        let a2: u32 = 0x2222_2222;
        let a3: u32 = 0x2222_2222;
        let b0: u32 = 0x2222_2222;
        let b1: u32 = 0x2222_2222;
        let sa: u32 = 0x7F7F_7F7F; // ue8m0 1.0 = 0x7F per byte
        let sb: u32 = 0x7F7F_7F7F;

        unsafe {
            ptx_asm!(
                "{ .reg .f32 fz, dr0, dr1, dr2, dr3; \
                   mov.f32 fz, 0f00000000; \
                   mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {dr0,dr1,dr2,dr3}, {%0,%1,%2,%3}, {%4,%5}, {fz,fz,fz,fz}, %6, {0,0}, %7, {0,0}; \
                   st.global.f32 [%8], dr0; \
                   st.global.f32 [%8+4], dr1; \
                   st.global.f32 [%8+8], dr2; \
                   st.global.f32 [%8+12], dr3; }",
                in("r") a0,
                in("r") a1,
                in("r") a2,
                in("r") a3,
                in("r") b0,
                in("r") b1,
                in("r") sa,
                in("r") sb,
                in("l") p,
                clobber("memory"),
            );
        }
    }
}

fn main() {
    println!("=== cuda-oxide FP4 MXFP4 m16n8k64 MMA (all-ones) ===");
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 128).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .fp4_mma_allones(&stream, cfg, &mut out_dev)
        .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    let bad = out.iter().filter(|&&v| v != 64.0).count();
    let (mn, mx) = out
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("D[0..8]: {:?}", &out[..8]);
    println!("min={mn} max={mx}  mismatches(!=64)={bad}/128");
    if bad == 0 {
        println!("\u{2713} SUCCESS: FP4 MMA numerically correct through cuda-oxide (all 64.0)");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
