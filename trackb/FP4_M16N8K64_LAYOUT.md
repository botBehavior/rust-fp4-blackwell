# FP4 m16n8k64 block-scaled MMA — complete validated layout (sm_120a)

All empirically validated on RTX 5070 Ti against CPU reference, 0 mismatches.
`g = lane>>2` (0..7), `t = lane&3` (0..3). Low nibble = lowest K. 8 E2M1 values per `.b32`.
Requires `.target sm_120a` (plain sm_120 rejects block_scale).

## Instruction (MXFP4 shown; NVFP4 = kind::mxf4nvf4, scale_vec::4X, .ue4m3)
```
mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
  {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}, sa, {0,0}, sb, {0,0};
```
Fragments: A=4×.b32, B=2×.b32, C/D=4×.f32, sa/sb=.b32. byte-id/thread-id = immediates {0,0}.

## Element layout (validated: fp4_mapping_test.cu)
- **A** (16×64 row-major), a0..a3: `row=(r&1)?g+8:g`, `k=t*16+(r>>1)*8+p` (p=nibble 0..7)
- **B** (64×8 col-major), b0,b1: `col=g`, `k=t*16+r*8+p`
- **D** (16×8), d0..d3: `(g,2t) (g,2t+1) (g+8,2t) (g+8,2t+1)`

## Scale layout (validated: fp4_scale_test.cu, HYP 1)
scale_vec::2X, K=64 → 2 blocks of 32. ue8m0 byte = 127 + log2(scale).
- **sa** (.b32, 4 bytes) = `[ SFA[g][blk0], SFA[g][blk1], SFA[g+8][blk0], SFA[g+8][blk1] ]`
- **sb** (.b32, low 2 bytes used, replicate to hi) = `[ SFB[blk0][g], SFB[blk1][g], (rpt), (rpt) ]`
- All 4 threads in a quad carry the same sa/sb; hardware addresses via the {0,0} immediates.

## E2M1 codes
mag = {0,0.5,1,1.5,2,3,4,6} for code&7; bit3 = sign. 1.0 = 0x2.

## Status
- Element mapping: PINNED. Scale layout: PINNED. → full FP4 GEMM with per-block scales is now buildable.
- FP4 MMA also runs in pure Rust via cuda-oxide: examples/fp4_mma (--arch sm_120a).
- Decode path (bandwidth-bound, our thesis lane): examples/fp4_gemv — dequant+FMA, NOT MMA;
  231 GB/s = 25.8% of 896 (naive baseline, optimization headroom).
