/*
 * Track A — end-to-end Llama-1B decode forward pass, pure Rust via cuda-oxide.
 * All projections are FP4 (Track B fp4_gemv); norms/attention/elementwise in f32.
 * Each decode kernel is independently CPU-parity validated (see llama_rmsnorm/rope/attn_decode
 * + fp4_gemv examples). This harness wires them into an autoregressive loop and measures tok/s.
 * Weights are random (perf harness); sanity-checks finite logits + valid argmax.
 *
 * Config: Llama-3.2-1B — dim 2048, 16 layers, 32 q / 8 kv heads, head_dim 64, ffn 8192,
 * vocab 128256, rope theta 5e5.
 */
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    const MAXK: usize = 8192; // fp4_gemv shared-x capacity (down-proj input)
    const MAXSEQ: usize = 1024;
    const HD: usize = 64;

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
        w: &[u32],
        scales: &[f32],
        x: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        static mut XS: SharedArray<f32, MAXK> = SharedArray::UNINIT;
        let kk = k as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let mut i = tid;
        while i < kk {
            unsafe { XS[i] = x[i] };
            i += bdim;
        }
        thread::sync_threads();
        let lane = tid % 32;
        let row = thread::index_1d().get() / 32;
        if row >= n as usize {
            return;
        }
        let kdiv8 = kk / 8;
        let wbase = row * kdiv8;
        let sbase = row * (kk / 32);
        let mut acc = 0.0f32;
        let mut u = lane;
        while u < kdiv8 {
            let word = w[wbase + u];
            let base_k = u * 8;
            let sc = scales[sbase + base_k / 32];
            let mut blk = 0.0f32;
            let mut p = 0usize;
            while p < 8 {
                blk += fp4_val((word >> (p * 4)) & 0xF) * unsafe { XS[base_k + p] };
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
            unsafe { *y.as_mut_ptr().add(row) = acc };
        }
    }

    #[kernel]
    pub fn rmsnorm(dim: u32, eps: f32, x: &[f32], weight: &[f32], mut y: DisjointSlice<f32>) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let d = dim as usize;
        let lane = tid % 32;
        let warp = tid / 32;
        let nwarps = bdim / 32;
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < d {
            let v = x[i];
            ss += v * v;
            i += bdim;
        }
        ss += warp::shuffle_xor_f32(ss, 16);
        ss += warp::shuffle_xor_f32(ss, 8);
        ss += warp::shuffle_xor_f32(ss, 4);
        ss += warp::shuffle_xor_f32(ss, 2);
        ss += warp::shuffle_xor_f32(ss, 1);
        if lane == 0 {
            unsafe { RED[warp] = ss };
        }
        thread::sync_threads();
        let mut total = 0.0f32;
        let mut w = 0usize;
        while w < nwarps {
            total += unsafe { RED[w] };
            w += 1;
        }
        let rms = 1.0f32 / (total / d as f32 + eps).sqrt();
        let mut i = tid;
        while i < d {
            unsafe { *y.as_mut_ptr().add(i) = x[i] * rms * weight[i] };
            i += bdim;
        }
    }

    #[kernel]
    pub fn rope(
        num_heads: u32,
        head_dim: u32,
        cos_base: u32,
        cos: &[f32],
        sin: &[f32],
        mut q: DisjointSlice<f32>,
    ) {
        let gid = thread::index_1d().get();
        let hd = head_dim as usize;
        let half = hd / 2;
        if gid >= num_heads as usize * half {
            return;
        }
        let head = gid / half;
        let j = gid % half;
        let base = head * hd;
        let c = cos[cos_base as usize + j];
        let s = sin[cos_base as usize + j];
        let p = q.as_mut_ptr();
        let x0 = unsafe { *p.add(base + j) };
        let x1 = unsafe { *p.add(base + j + half) };
        unsafe {
            *p.add(base + j) = x0 * c - x1 * s;
            *p.add(base + j + half) = x1 * c + x0 * s;
        }
    }

    #[kernel]
    pub fn copy_into(n: u32, off: u32, src: &[f32], mut dst: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < n as usize {
            unsafe { *dst.as_mut_ptr().add(off as usize + i) = src[i] };
        }
    }

    #[kernel]
    pub fn attn_decode(
        n_heads: u32,
        n_kv_heads: u32,
        seq: u32,
        q: &[f32],
        kc: &[f32],
        vc: &[f32],
        mut out: DisjointSlice<f32>,
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
        thread::sync_threads();
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
        thread::sync_threads();
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

    #[kernel]
    pub fn swiglu(n: u32, gate: &[f32], up: &[f32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < n as usize {
            let g = gate[i];
            let s = g / (1.0f32 + (-g).exp());
            unsafe { *out.as_mut_ptr().add(i) = s * up[i] };
        }
    }

    #[kernel]
    pub fn residual(n: u32, y: &[f32], mut x: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i < n as usize {
            let p = x.as_mut_ptr();
            unsafe { *p.add(i) = *p.add(i) + y[i] };
        }
    }
}

// ---- host ----
const DIM: usize = 2048;
const NL: usize = 16;
const NH: usize = 32;
const NKV: usize = 8;
const HD: usize = 64;
const FFN: usize = 8192;
const VOCAB: usize = 128256;
const HALF: usize = HD / 2;
const THETA: f64 = 500000.0;
const EPS: f32 = 1e-5;

fn nxt(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

fn mk_fp4(
    stream: &CudaStream,
    s: &mut u32,
    outd: usize,
    ind: usize,
) -> (DeviceBuffer<u32>, DeviceBuffer<f32>) {
    let mut w = vec![0u32; outd * ind / 8];
    for x in w.iter_mut() {
        *x = nxt(s);
    }
    let mut sc = vec![0f32; outd * (ind / 32)];
    for x in sc.iter_mut() {
        *x = 0.25 + (nxt(s) % 8) as f32 * 0.03125;
    }
    (
        DeviceBuffer::from_host(stream, &w).unwrap(),
        DeviceBuffer::from_host(stream, &sc).unwrap(),
    )
}

struct Layer {
    wq: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wk: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wv: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wo: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wg: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wu: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    wd: (DeviceBuffer<u32>, DeviceBuffer<f32>),
    an: DeviceBuffer<f32>,
    fnw: DeviceBuffer<f32>,
    kc: DeviceBuffer<f32>,
    vc: DeviceBuffer<f32>,
}

fn elem_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn gemv_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n / 8) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn main() {
    const P: usize = 512; // prior context length
    const WARM: usize = 4;
    const GEN: usize = 32;
    const SEQ_MAX: usize = P + WARM + GEN + 8;

    println!("=== Llama-1B decode (pure Rust / cuda-oxide, FP4 weights) ===");
    println!(
        "dim={DIM} layers={NL} heads={NH}/{NKV} head_dim={HD} ffn={FFN} vocab={VOCAB} ctx={P}"
    );

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let mut s: u32 = 0xdead_beef;

    print!("allocating + uploading FP4 weights... ");
    let t_load = Instant::now();
    let mut layers: Vec<Layer> = Vec::with_capacity(NL);
    for _ in 0..NL {
        let kc_init = vec![0.01f32; SEQ_MAX * NKV * HD];
        let vc_init = vec![0.01f32; SEQ_MAX * NKV * HD];
        // randomize prior context lightly
        layers.push(Layer {
            wq: mk_fp4(&stream, &mut s, NH * HD, DIM),
            wk: mk_fp4(&stream, &mut s, NKV * HD, DIM),
            wv: mk_fp4(&stream, &mut s, NKV * HD, DIM),
            wo: mk_fp4(&stream, &mut s, DIM, NH * HD),
            wg: mk_fp4(&stream, &mut s, FFN, DIM),
            wu: mk_fp4(&stream, &mut s, FFN, DIM),
            wd: mk_fp4(&stream, &mut s, DIM, FFN),
            an: DeviceBuffer::from_host(&stream, &vec![1.0f32; DIM]).unwrap(),
            fnw: DeviceBuffer::from_host(&stream, &vec![1.0f32; DIM]).unwrap(),
            kc: DeviceBuffer::from_host(&stream, &kc_init).unwrap(),
            vc: DeviceBuffer::from_host(&stream, &vc_init).unwrap(),
        });
    }
    let final_norm = DeviceBuffer::from_host(&stream, &vec![1.0f32; DIM]).unwrap();
    let lmhead = mk_fp4(&stream, &mut s, VOCAB, DIM);

    // rotary cache for all positions
    let mut cosv = vec![0f32; SEQ_MAX * HALF];
    let mut sinv = vec![0f32; SEQ_MAX * HALF];
    for pos in 0..SEQ_MAX {
        for j in 0..HALF {
            let freq = 1.0 / THETA.powf(2.0 * j as f64 / HD as f64);
            let a = pos as f64 * freq;
            cosv[pos * HALF + j] = a.cos() as f32;
            sinv[pos * HALF + j] = a.sin() as f32;
        }
    }
    let cos_d = DeviceBuffer::from_host(&stream, &cosv).unwrap();
    let sin_d = DeviceBuffer::from_host(&stream, &sinv).unwrap();

    // scratch
    let xinit: Vec<f32> = (0..DIM).map(|_| (nxt(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
    let mut x = DeviceBuffer::from_host(&stream, &xinit).unwrap();
    let mut h = DeviceBuffer::<f32>::zeroed(&stream, DIM).unwrap();
    let mut q = DeviceBuffer::<f32>::zeroed(&stream, NH * HD).unwrap();
    let mut kk = DeviceBuffer::<f32>::zeroed(&stream, NKV * HD).unwrap();
    let mut vv = DeviceBuffer::<f32>::zeroed(&stream, NKV * HD).unwrap();
    let mut a = DeviceBuffer::<f32>::zeroed(&stream, NH * HD).unwrap();
    let mut ao = DeviceBuffer::<f32>::zeroed(&stream, DIM).unwrap();
    let mut h2 = DeviceBuffer::<f32>::zeroed(&stream, DIM).unwrap();
    let mut g = DeviceBuffer::<f32>::zeroed(&stream, FFN).unwrap();
    let mut u = DeviceBuffer::<f32>::zeroed(&stream, FFN).unwrap();
    let mut sw = DeviceBuffer::<f32>::zeroed(&stream, FFN).unwrap();
    let mut dproj = DeviceBuffer::<f32>::zeroed(&stream, DIM).unwrap();
    let mut logits = DeviceBuffer::<f32>::zeroed(&stream, VOCAB).unwrap();
    let m = kernels::load(&ctx).expect("load");
    stream.synchronize().unwrap();
    println!("done ({:.1}s)", t_load.elapsed().as_secs_f32());

    let decode_step = |pos: usize,
                           x: &mut DeviceBuffer<f32>,
                           h: &mut DeviceBuffer<f32>,
                           q: &mut DeviceBuffer<f32>,
                           kk: &mut DeviceBuffer<f32>,
                           vv: &mut DeviceBuffer<f32>,
                           a: &mut DeviceBuffer<f32>,
                           ao: &mut DeviceBuffer<f32>,
                           h2: &mut DeviceBuffer<f32>,
                           g: &mut DeviceBuffer<f32>,
                           u: &mut DeviceBuffer<f32>,
                           sw: &mut DeviceBuffer<f32>,
                           dproj: &mut DeviceBuffer<f32>,
                           logits: &mut DeviceBuffer<f32>,
                           layers: &mut [Layer]|
     -> usize {
        let cos_base = (pos * HALF) as u32;
        for l in layers.iter_mut() {
            m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &*x, &l.an, h).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(NH * HD), (NH * HD) as u32, DIM as u32, &l.wq.0, &l.wq.1, &*h, q).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(NKV * HD), (NKV * HD) as u32, DIM as u32, &l.wk.0, &l.wk.1, &*h, kk).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(NKV * HD), (NKV * HD) as u32, DIM as u32, &l.wv.0, &l.wv.1, &*h, vv).unwrap();
            m.rope(&stream, elem_cfg(NH * HALF), NH as u32, HD as u32, cos_base, &cos_d, &sin_d, q).unwrap();
            m.rope(&stream, elem_cfg(NKV * HALF), NKV as u32, HD as u32, cos_base, &cos_d, &sin_d, kk).unwrap();
            let off = (pos * NKV * HD) as u32;
            m.copy_into(&stream, elem_cfg(NKV * HD), (NKV * HD) as u32, off, &*kk, &mut l.kc).unwrap();
            m.copy_into(&stream, elem_cfg(NKV * HD), (NKV * HD) as u32, off, &*vv, &mut l.vc).unwrap();
            m.attn_decode(&stream, LaunchConfig { grid_dim: (NH as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }, NH as u32, NKV as u32, (pos + 1) as u32, &*q, &l.kc, &l.vc, a).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(DIM), DIM as u32, (NH * HD) as u32, &l.wo.0, &l.wo.1, &*a, ao).unwrap();
            m.residual(&stream, elem_cfg(DIM), DIM as u32, &*ao, x).unwrap();
            m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &*x, &l.fnw, h2).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(FFN), FFN as u32, DIM as u32, &l.wg.0, &l.wg.1, &*h2, g).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(FFN), FFN as u32, DIM as u32, &l.wu.0, &l.wu.1, &*h2, u).unwrap();
            m.swiglu(&stream, elem_cfg(FFN), FFN as u32, &*g, &*u, sw).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(DIM), DIM as u32, FFN as u32, &l.wd.0, &l.wd.1, &*sw, dproj).unwrap();
            m.residual(&stream, elem_cfg(DIM), DIM as u32, &*dproj, x).unwrap();
        }
        m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &*x, &final_norm, h).unwrap();
        m.fp4_gemv(&stream, gemv_cfg(VOCAB), VOCAB as u32, DIM as u32, &lmhead.0, &lmhead.1, &*h, logits).unwrap();
        let lv = logits.to_host_vec(&stream).unwrap();
        let mut best = 0usize;
        let mut bv = f32::MIN;
        for (i, &v) in lv.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best
    };

    let mut pos = P;
    let mut last = 0usize;
    for _ in 0..WARM {
        last = decode_step(pos, &mut x, &mut h, &mut q, &mut kk, &mut vv, &mut a, &mut ao, &mut h2, &mut g, &mut u, &mut sw, &mut dproj, &mut logits, &mut layers);
        pos += 1;
    }
    stream.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..GEN {
        last = decode_step(pos, &mut x, &mut h, &mut q, &mut kk, &mut vv, &mut a, &mut ao, &mut h2, &mut g, &mut u, &mut sw, &mut dproj, &mut logits, &mut layers);
        pos += 1;
    }
    let dt = t.elapsed().as_secs_f64();
    let toks = GEN as f64 / dt;
    let lv = logits.to_host_vec(&stream).unwrap();
    let finite = lv.iter().all(|v| v.is_finite());
    println!("\nlast argmax token id = {last}  (logits finite: {finite})");
    println!(
        "decode: {:.3} ms/token, {:.1} tokens/sec  ({} layers, ctx {}, {} tokens timed)",
        dt * 1000.0 / GEN as f64,
        toks,
        NL,
        P,
        GEN
    );
    if finite && last < VOCAB {
        println!("\u{2713} SUCCESS: end-to-end Rust decode runs; finite logits, valid argmax");
    } else {
        eprintln!("\u{2717} FAIL");
        std::process::exit(1);
    }
}
