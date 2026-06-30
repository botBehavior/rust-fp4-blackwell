// Pin the SCALE-FACTOR layout for MXFP4 m16n8k64 (scale_vec::2X, ue8m0).
// Element mapping already validated; now add non-uniform per-block scales.
// K=64 => 2 blocks of 32 along K. SFA[16][2], SFB[2][8]. ue8m0: byte = 127 + log2(scale).
// Dequant: A_real[i][k]=A_fp4*SFA[i][k/32]; B_real[k][j]=B_fp4*SFB[k/32][j].
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

static const float FP4V[8] = {0,0.5f,1,1.5f,2,3,4,6};
uint8_t e2m1(float v){ float a=fabsf(v); uint8_t c=0; for(int i=0;i<8;i++) if(a==FP4V[i]){c=i;break;} if(v<0)c|=0x8; return c; }
uint8_t ue8m0(float s){ int e=(int)lroundf(log2f(s)); return (uint8_t)(127+e); } // power-of-two scales

__global__ void mma(const uint32_t* pA,const uint32_t* pB,const uint32_t* pSA,const uint32_t* pSB,float* out){
  int lane=threadIdx.x&31;
  uint32_t a0=pA[lane*4+0],a1=pA[lane*4+1],a2=pA[lane*4+2],a3=pA[lane*4+3];
  uint32_t b0=pB[lane*2+0],b1=pB[lane*2+1];
  uint32_t sa=pSA[lane], sb=pSB[lane];
  float d0=0,d1=0,d2=0,d3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
  out[lane*4+0]=d0; out[lane*4+1]=d1; out[lane*4+2]=d2; out[lane*4+3]=d3;
}

int main(int argc, char** argv){
  int HYP = argc>1 ? atoi(argv[1]) : 1;
  float A[16][64],B[64][8],SFA[16][2],SFB[2][8],Dref[16][8]={};
  for(int i=0;i<16;i++)for(int k=0;k<64;k++)A[i][k]=FP4V[(i*7+k*3)%8];
  for(int k=0;k<64;k++)for(int j=0;j<8;j++)B[k][j]=FP4V[(k*5+j*2)%8];
  // non-uniform power-of-two scales (0.5,1,2,4)
  float SV[4]={0.5f,1,2,4};
  for(int i=0;i<16;i++)for(int b=0;b<2;b++)SFA[i][b]=SV[(i+b)%4];
  for(int b=0;b<2;b++)for(int j=0;j<8;j++)SFB[b][j]=SV[(j+b*2)%4];
  for(int i=0;i<16;i++)for(int j=0;j<8;j++){float s=0;for(int k=0;k<64;k++)s+=A[i][k]*SFA[i][k/32]*B[k][j]*SFB[k/32][j];Dref[i][j]=s;}

  uint32_t pA[128]={0},pB[64]={0},pSA[32]={0},pSB[32]={0};
  for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;
    for(int r=0;r<4;r++){int row=(r&1)?g+8:g,kh=r>>1;uint32_t w=0;for(int p=0;p<8;p++){int k=t*16+kh*8+p;w|=(uint32_t)e2m1(A[row][k])<<(p*4);}pA[lane*4+r]=w;}
    for(int r=0;r<2;r++){int col=g;uint32_t w=0;for(int p=0;p<8;p++){int k=t*16+r*8+p;w|=(uint32_t)e2m1(B[k][col])<<(p*4);}pB[lane*2+r]=w;}
    // ---- scale packing hypotheses ----
    uint32_t saw=0, sbw=0;
    if(HYP==1){ // sa bytes = [SFA[g][0],SFA[g][1],SFA[g+8][0],SFA[g+8][1]]; sb=[SFB[0][g],SFB[1][g],..]
      saw = ue8m0(SFA[g][0]) | (ue8m0(SFA[g][1])<<8) | (ue8m0(SFA[g+8][0])<<16) | (ue8m0(SFA[g+8][1])<<24);
      sbw = ue8m0(SFB[0][g]) | (ue8m0(SFB[1][g])<<8) | (ue8m0(SFB[0][g])<<16) | (ue8m0(SFB[1][g])<<24);
    } else if(HYP==2){ // interleave blocks in low 2 bytes only
      saw = ue8m0(SFA[g][0]) | (ue8m0(SFA[g+8][0])<<8) | (ue8m0(SFA[g][1])<<16) | (ue8m0(SFA[g+8][1])<<24);
      sbw = ue8m0(SFB[0][g]) | (ue8m0(SFB[1][g])<<8);
    } else if(HYP==3){ // all 4 bytes = same block layout, only tid 0/1 matter
      saw = ue8m0(SFA[g][t&1]) | (ue8m0(SFA[g+8][t&1])<<8) | (ue8m0(SFA[g][t&1])<<16) | (ue8m0(SFA[g+8][t&1])<<24);
      sbw = ue8m0(SFB[0][g]) | (ue8m0(SFB[1][g])<<8) | (ue8m0(SFB[0][g])<<16) | (ue8m0(SFB[1][g])<<24);
    }
    pSA[lane]=saw; pSB[lane]=sbw;
  }

  uint32_t *dA,*dB,*dSA,*dSB; float *dO,hO[128];
  cudaMalloc(&dA,512);cudaMalloc(&dB,256);cudaMalloc(&dSA,128);cudaMalloc(&dSB,128);cudaMalloc(&dO,512);
  cudaMemcpy(dA,pA,512,cudaMemcpyHostToDevice);cudaMemcpy(dB,pB,256,cudaMemcpyHostToDevice);
  cudaMemcpy(dSA,pSA,128,cudaMemcpyHostToDevice);cudaMemcpy(dSB,pSB,128,cudaMemcpyHostToDevice);
  mma<<<1,32>>>(dA,dB,dSA,dSB,dO);
  cudaError_t e=cudaDeviceSynchronize();if(e){printf("CUDA:%s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(hO,dO,512,cudaMemcpyDeviceToHost);
  float Dg[16][8];int bad=0;
  for(int lane=0;lane<32;lane++){int g=lane>>2,t=lane&3;Dg[g][t*2]=hO[lane*4+0];Dg[g][t*2+1]=hO[lane*4+1];Dg[g+8][t*2]=hO[lane*4+2];Dg[g+8][t*2+1]=hO[lane*4+3];}
  for(int i=0;i<16;i++)for(int j=0;j<8;j++){if(fabsf(Dg[i][j]-Dref[i][j])>1e-2f){if(bad<8)printf("  [%d][%d] gpu=%.1f ref=%.1f\n",i,j,Dg[i][j],Dref[i][j]);bad++;}}
  printf("HYP=%d  Dref[0][0..3]=%.1f %.1f %.1f %.1f  gpu=%.1f %.1f %.1f %.1f\n",HYP,Dref[0][0],Dref[0][1],Dref[0][2],Dref[0][3],Dg[0][0],Dg[0][1],Dg[0][2],Dg[0][3]);
  printf("HYP=%d mismatches=%d/128 %s\n",HYP,bad,bad==0?"\xE2\x9C\x93 SCALE LAYOUT PINNED":"");
  return bad?1:0;
}
