// Pin the m16n8k64 FP4 (e2m1) fragment mapping empirically.
// Host packs logical A[16][64], B[64][8] into per-lane registers using a HYPOTHESIZED
// layout; GPU runs the MMA; we compare D to a CPU reference computed from the logical
// arrays (layout-independent ground truth). Match => hypothesis correct.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

static const float FP4V[8] = {0,0.5f,1,1.5f,2,3,4,6};
uint8_t e2m1(float v){ float a=fabsf(v); uint8_t c=0;
  for(int i=0;i<8;i++) if(a==FP4V[i]){c=i;break;}
  if(v<0) c|=0x8; return c; }
float fp4dec(uint8_t c){ float m=FP4V[c&7]; return (c&8)? -m : m; }

// ---- layout hypothesis (groupID g=lane>>2, tid t=lane&3) ----
// A[16x64] row-major: regs a0..a3; row = (r&1)?g+8:g ; khalf=(r>>1); k=t*16+khalf*8+p
// B[64x8] col-major:  regs b0,b1; col=g ; k=t*16+r*8+p
// D[16x8]:            d0:(g, t*2) d1:(g, t*2+1) d2:(g+8, t*2) d3:(g+8, t*2+1)

__global__ void mma(const uint32_t* pA, const uint32_t* pB, float* out){
  int lane = threadIdx.x & 31;
  uint32_t a0=pA[lane*4+0],a1=pA[lane*4+1],a2=pA[lane*4+2],a3=pA[lane*4+3];
  uint32_t b0=pB[lane*2+0],b1=pB[lane*2+1];
  uint32_t sa=0x7F7F7F7F, sb=0x7F7F7F7F;
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
  out[lane*4+0]=d0; out[lane*4+1]=d1; out[lane*4+2]=d2; out[lane*4+3]=d3;
}

int main(){
  float A[16][64], B[64][8], Dref[16][8]={};
  // fp4-exact non-uniform data
  for(int i=0;i<16;i++) for(int k=0;k<64;k++) A[i][k]=FP4V[(i*7+k*3)%8];
  for(int k=0;k<64;k++) for(int j=0;j<8;j++)  B[k][j]=FP4V[(k*5+j*2)%8];
  for(int i=0;i<16;i++) for(int j=0;j<8;j++){ float s=0; for(int k=0;k<64;k++) s+=A[i][k]*B[k][j]; Dref[i][j]=s; }

  uint32_t pA[32*4]={0}, pB[32*2]={0};
  for(int lane=0; lane<32; lane++){
    int g=lane>>2, t=lane&3;
    for(int r=0;r<4;r++){ int row=(r&1)?g+8:g, kh=r>>1; uint32_t w=0;
      for(int p=0;p<8;p++){ int k=t*16+kh*8+p; w |= (uint32_t)e2m1(A[row][k])<<(p*4); }
      pA[lane*4+r]=w; }
    for(int r=0;r<2;r++){ int col=g; uint32_t w=0;
      for(int p=0;p<8;p++){ int k=t*16+r*8+p; w |= (uint32_t)e2m1(B[k][col])<<(p*4); }
      pB[lane*2+r]=w; }
  }

  uint32_t *dA,*dB; float *dO, hO[128];
  cudaMalloc(&dA,sizeof(pA)); cudaMalloc(&dB,sizeof(pB)); cudaMalloc(&dO,128*4);
  cudaMemcpy(dA,pA,sizeof(pA),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,pB,sizeof(pB),cudaMemcpyHostToDevice);
  mma<<<1,32>>>(dA,dB,dO);
  cudaError_t e=cudaDeviceSynchronize(); if(e){printf("CUDA: %s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(hO,dO,128*4,cudaMemcpyDeviceToHost);

  // unpack via D-layout, compare
  float Dg[16][8]; int bad=0;
  for(int lane=0;lane<32;lane++){ int g=lane>>2,t=lane&3;
    Dg[g][t*2]     = hO[lane*4+0];
    Dg[g][t*2+1]   = hO[lane*4+1];
    Dg[g+8][t*2]   = hO[lane*4+2];
    Dg[g+8][t*2+1] = hO[lane*4+3];
  }
  for(int i=0;i<16;i++) for(int j=0;j<8;j++){ if(fabsf(Dg[i][j]-Dref[i][j])>1e-3f){ if(bad<6) printf("  mismatch [%d][%d] gpu=%.1f ref=%.1f\n",i,j,Dg[i][j],Dref[i][j]); bad++; } }
  printf("Dref[0][0..3]=%.1f %.1f %.1f %.1f  gpu=%.1f %.1f %.1f %.1f\n",
    Dref[0][0],Dref[0][1],Dref[0][2],Dref[0][3], Dg[0][0],Dg[0][1],Dg[0][2],Dg[0][3]);
  printf("mismatches=%d/128\n", bad);
  printf(bad==0? "\xE2\x9C\x93 MAPPING PINNED: hypothesis matches reference\n" : "\xE2\x9C\x97 hypothesis wrong, adjust pack\n");
  return bad?1:0;
}
