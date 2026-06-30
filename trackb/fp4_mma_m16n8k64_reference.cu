#include <cstdio>
#include <cuda_runtime.h>

// Single-warp MXFP4 m16n8k64 MMA. A=all 1.0, B=all 1.0, scale=1.0 (ue8m0).
// Permutation-invariant: every D element must == sum_{k=0..63} 1*1 = 64.0
__global__ void fp4mma(float* out) {
  unsigned a0=0x22222222,a1=0x22222222,a2=0x22222222,a3=0x22222222; // eight 1.0 (e2m1=0x2) each
  unsigned b0=0x22222222,b1=0x22222222;
  unsigned sa=0x7F7F7F7F, sb=0x7F7F7F7F; // ue8m0 1.0 = exponent 127 = 0x7F
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
  int lane = threadIdx.x & 31;
  int row = lane>>2, col=(lane&3)*2;
  out[(row)*8 + col]     = d0;
  out[(row)*8 + col+1]   = d1;
  out[(row+8)*8 + col]   = d2;
  out[(row+8)*8 + col+1] = d3;
}
int main(){
  float *d, h[128];
  cudaMalloc(&d, 128*sizeof(float));
  cudaMemset(d, 0, 128*sizeof(float));
  fp4mma<<<1,32>>>(d);
  cudaError_t e=cudaDeviceSynchronize();
  if(e){printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1;}
  cudaMemcpy(h, d, 128*sizeof(float), cudaMemcpyDeviceToHost);
  int bad=0; float mn=1e9,mx=-1e9;
  for(int i=0;i<128;i++){ if(h[i]!=64.0f) bad++; if(h[i]<mn)mn=h[i]; if(h[i]>mx)mx=h[i]; }
  printf("D[0..7]: "); for(int i=0;i<8;i++) printf("%.1f ", h[i]); printf("\n");
  printf("min=%.3f max=%.3f  mismatches(!=64)=%d/128\n", mn,mx,bad);
  printf(bad==0 ? "\xE2\x9C\x93 SUCCESS: FP4 m16n8k64 MMA numerically correct (all 64.0)\n"
                : "\xE2\x9C\x97 FAIL\n");
  return bad?1:0;
}
