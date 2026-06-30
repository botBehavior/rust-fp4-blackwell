/*
 * Track A/B capstone — run a REAL Llama model (TinyLlama-1.1B) end-to-end in pure Rust via
 * cuda-oxide, with all projection weights quantized to MXFP4 (E2M1 + per-32 f32 scale).
 * Parses safetensors, quantizes on host, embedding lookup, prefill + greedy generation,
 * device argmax. Prints generated token ids (decode with the `tok` helper).
 *
 * Usage: cargo oxide run llama_run -- <id0> <id1> ...    (token ids from `tok encode`)
 */
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    const MAXK: usize = 8192;
    const MAXSEQ: usize = 2048;
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
    pub fn fp4_gemv(n: u32, k: u32, w: &[u32], scales: &[f32], x: &[f32], mut y: DisjointSlice<f32>) {
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
    pub fn rope(num_heads: u32, head_dim: u32, cos_base: u32, cos: &[f32], sin: &[f32], mut q: DisjointSlice<f32>) {
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
    pub fn attn_decode(n_heads: u32, n_kv_heads: u32, seq: u32, q: &[f32], kc: &[f32], vc: &[f32], mut out: DisjointSlice<f32>) {
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

    #[kernel]
    pub fn argmax(n: u32, logits: &[f32], mut out: DisjointSlice<u32>) {
        static mut MV: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut MI: SharedArray<u32, 32> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let lane = tid % 32;
        let warp = tid / 32;
        let nwarps = bdim / 32;
        let mut bv = -1e30f32;
        let mut bi = 0u32;
        let mut i = tid;
        while i < n as usize {
            let v = logits[i];
            if v > bv {
                bv = v;
                bi = i as u32;
            }
            i += bdim;
        }
        let mut off = 16u32;
        while off >= 1 {
            let ov = warp::shuffle_xor_f32(bv, off);
            let oi = warp::shuffle_xor(bi, off);
            if ov > bv || (ov == bv && oi < bi) {
                bv = ov;
                bi = oi;
            }
            off >>= 1;
        }
        if lane == 0 {
            unsafe {
                MV[warp] = bv;
                MI[warp] = bi;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let mut gv = -1e30f32;
            let mut gi = 0u32;
            let mut w = 0;
            while w < nwarps {
                let v = unsafe { MV[w] };
                let ix = unsafe { MI[w] };
                if v > gv || (v == gv && ix < gi) {
                    gv = v;
                    gi = ix;
                }
                w += 1;
            }
            unsafe { *out.as_mut_ptr() = gi };
        }
    }
}

// ---- TinyLlama-1.1B config ----
const DIM: usize = 2048;
const NL: usize = 22;
const NH: usize = 32;
const NKV: usize = 4;
const HD: usize = 64;
const FFN: usize = 5632;
const VOCAB: usize = 32000;
const HALF: usize = HD / 2;
const THETA: f64 = 10000.0;
const EPS: f32 = 1e-5;
const EOS: u32 = 2;
const MODEL: &str = "./models/tinyllama/model.safetensors";

const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn quant_fp4(w: &[f32], outd: usize, ind: usize) -> (Vec<u32>, Vec<f32>) {
    let nb = ind / 32;
    let mut codes = vec![0u8; outd * ind];
    let mut scales = vec![0f32; outd * nb];
    for r in 0..outd {
        for b in 0..nb {
            let base = r * ind + b * 32;
            let mut amax = 0f32;
            for i in 0..32 {
                amax = amax.max(w[base + i].abs());
            }
            let scale = if amax > 0.0 { amax / 6.0 } else { 1.0 };
            scales[r * nb + b] = scale;
            for i in 0..32 {
                let v = w[base + i] / scale;
                let a = v.abs();
                let mut bc = 0usize;
                let mut bd = f32::MAX;
                for (ci, &lv) in LEVELS.iter().enumerate() {
                    let d = (a - lv).abs();
                    if d < bd {
                        bd = d;
                        bc = ci;
                    }
                }
                let mut code = bc as u8;
                if v < 0.0 {
                    code |= 8;
                }
                codes[base + i] = code;
            }
        }
    }
    let mut packed = vec![0u32; outd * ind / 8];
    for c in 0..packed.len() {
        let mut x = 0u32;
        for p in 0..8 {
            x |= (codes[c * 8 + p] as u32) << (p * 4);
        }
        packed[c] = x;
    }
    (packed, scales)
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
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
fn gemv_cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n / 8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn main() {
    let mut prompt: Vec<u32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    // `cargo oxide run` does not forward args; allow PROMPT_IDS="1 390 ..." as an input path.
    if prompt.is_empty() {
        if let Ok(s) = std::env::var("PROMPT_IDS") {
            prompt = s.split_whitespace().map(|x| x.parse().unwrap()).collect();
        }
    }
    let prompt = if prompt.is_empty() { vec![1u32, 15043] } else { prompt }; // BOS + "Hello" fallback
    const MAX_NEW: usize = 200;

    println!("=== TinyLlama-1.1B in pure Rust / cuda-oxide (MXFP4 weights) ===");
    print!("reading + quantizing safetensors... ");
    let t0 = Instant::now();
    let raw = std::fs::read(MODEL).expect("read model");
    let hlen = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
    let data = &raw[8 + hlen..];
    let get = |name: &str| -> Vec<f32> {
        let m = &header[name];
        let off = m["data_offsets"].as_array().unwrap();
        let st = off[0].as_u64().unwrap() as usize;
        let en = off[1].as_u64().unwrap() as usize;
        let dt = m["dtype"].as_str().unwrap();
        let b = &data[st..en];
        match dt {
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => b.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            d => panic!("dtype {d}"),
        }
    };

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let qup = |name: &str, outd: usize, ind: usize| -> (DeviceBuffer<u32>, DeviceBuffer<f32>) {
        let (p, s) = quant_fp4(&get(name), outd, ind);
        (DeviceBuffer::from_host(&stream, &p).unwrap(), DeviceBuffer::from_host(&stream, &s).unwrap())
    };
    let fup = |name: &str| -> DeviceBuffer<f32> { DeviceBuffer::from_host(&stream, &get(name)).unwrap() };

    const MAXSEQ: usize = 2048;
    let mut layers: Vec<Layer> = Vec::with_capacity(NL);
    for i in 0..NL {
        let p = format!("model.layers.{i}.");
        layers.push(Layer {
            wq: qup(&format!("{p}self_attn.q_proj.weight"), NH * HD, DIM),
            wk: qup(&format!("{p}self_attn.k_proj.weight"), NKV * HD, DIM),
            wv: qup(&format!("{p}self_attn.v_proj.weight"), NKV * HD, DIM),
            wo: qup(&format!("{p}self_attn.o_proj.weight"), DIM, NH * HD),
            wg: qup(&format!("{p}mlp.gate_proj.weight"), FFN, DIM),
            wu: qup(&format!("{p}mlp.up_proj.weight"), FFN, DIM),
            wd: qup(&format!("{p}mlp.down_proj.weight"), DIM, FFN),
            an: fup(&format!("{p}input_layernorm.weight")),
            fnw: fup(&format!("{p}post_attention_layernorm.weight")),
            kc: DeviceBuffer::<f32>::zeroed(&stream, MAXSEQ * NKV * HD).unwrap(),
            vc: DeviceBuffer::<f32>::zeroed(&stream, MAXSEQ * NKV * HD).unwrap(),
        });
    }
    let final_norm = fup("model.norm.weight");
    let lmhead = qup("lm_head.weight", VOCAB, DIM);
    let embed = get("model.embed_tokens.weight"); // host f32 [VOCAB*DIM]

    let mut cosv = vec![0f32; MAXSEQ * HALF];
    let mut sinv = vec![0f32; MAXSEQ * HALF];
    for pos in 0..MAXSEQ {
        for j in 0..HALF {
            let freq = 1.0 / THETA.powf(2.0 * j as f64 / HD as f64);
            let a = pos as f64 * freq;
            cosv[pos * HALF + j] = a.cos() as f32;
            sinv[pos * HALF + j] = a.sin() as f32;
        }
    }
    let cos_d = DeviceBuffer::from_host(&stream, &cosv).unwrap();
    let sin_d = DeviceBuffer::from_host(&stream, &sinv).unwrap();

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
    let mut idx_d = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let m = kernels::load(&ctx).expect("load");
    stream.synchronize().unwrap();
    println!("done ({:.1}s)", t0.elapsed().as_secs_f32());

    let forward = |tok: u32,
                   pos: usize,
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
                   idx_d: &mut DeviceBuffer<u32>,
                   layers: &mut [Layer]|
     -> u32 {
        let xrow = &embed[tok as usize * DIM..tok as usize * DIM + DIM];
        let mut x = DeviceBuffer::from_host(&stream, xrow).unwrap();
        let cos_base = (pos * HALF) as u32;
        for l in layers.iter_mut() {
            m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &x, &l.an, h).unwrap();
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
            m.residual(&stream, elem_cfg(DIM), DIM as u32, &*ao, &mut x).unwrap();
            m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &x, &l.fnw, h2).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(FFN), FFN as u32, DIM as u32, &l.wg.0, &l.wg.1, &*h2, g).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(FFN), FFN as u32, DIM as u32, &l.wu.0, &l.wu.1, &*h2, u).unwrap();
            m.swiglu(&stream, elem_cfg(FFN), FFN as u32, &*g, &*u, sw).unwrap();
            m.fp4_gemv(&stream, gemv_cfg(DIM), DIM as u32, FFN as u32, &l.wd.0, &l.wd.1, &*sw, dproj).unwrap();
            m.residual(&stream, elem_cfg(DIM), DIM as u32, &*dproj, &mut x).unwrap();
        }
        m.rmsnorm(&stream, elem_cfg(DIM), DIM as u32, EPS, &x, &final_norm, h).unwrap();
        m.fp4_gemv(&stream, gemv_cfg(VOCAB), VOCAB as u32, DIM as u32, &lmhead.0, &lmhead.1, &*h, logits).unwrap();
        m.argmax(&stream, LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, VOCAB as u32, &*logits, idx_d).unwrap();
        idx_d.to_host_vec(&stream).unwrap()[0]
    };

    print!("prompt ids: ");
    for t in &prompt {
        print!("{t} ");
    }
    println!();

    let mut ids = prompt.clone();
    let mut pos = 0usize;
    let mut gen_tokens = 0usize;
    let t_gen = Instant::now();
    let mut timed_tokens = 0usize;
    let mut t_decode = Instant::now();
    loop {
        let tok = ids[pos];
        let next = forward(tok, pos, &mut h, &mut q, &mut kk, &mut vv, &mut a, &mut ao, &mut h2, &mut g, &mut u, &mut sw, &mut dproj, &mut logits, &mut idx_d, &mut layers);
        pos += 1;
        if pos == prompt.len() {
            t_decode = Instant::now(); // start timing real generation after prefill
        }
        if pos >= prompt.len() {
            ids.push(next);
            gen_tokens += 1;
            if pos > prompt.len() {
                timed_tokens += 1;
            }
            if next == EOS || gen_tokens >= MAX_NEW {
                break;
            }
        }
        if pos >= MAXSEQ - 1 {
            break;
        }
    }
    let dt = t_decode.elapsed().as_secs_f64();
    let _ = t_gen;

    print!("\noutput ids: ");
    for t in &ids[prompt.len()..] {
        print!("{t} ");
    }
    println!();
    if timed_tokens > 0 {
        println!(
            "\ngen: {:.2} ms/token, {:.1} tokens/sec ({} tokens, ctx grows from {})",
            dt * 1000.0 / timed_tokens as f64,
            timed_tokens as f64 / dt,
            timed_tokens,
            prompt.len()
        );
    }
    println!("\u{2713} decode `tok decode <output ids>` to read the text");
}

fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut e = -14i32;
            let mut m = mant as u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(f)
}
