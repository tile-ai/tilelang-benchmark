
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <math.h>

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

#define htanh(x) __float2half_rn(tanh(__half2float(x)))
#define htan(x) __float2half_rn(tanf(__half2float(x)))
#define hatan(x) __float2half_rn(atanf(__half2float(x)))
#define herf(x) __float2half_rn(erff(__half2float(x)))
#define hexp(x) __float2half_rn(expf(__half2float(x)))

#define HIPRT_INF_F        __int_as_float(0x7f800000)
#define HIPRT_NAN_F        __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F   __int_as_float(0x80000000)
#define HIPRT_ZERO_F       0.0f
#define HIPRT_ONE_F        1.0f

/* double precision constants */
#define HIPRT_INF          __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN          __hiloint2double(0xfff80000, 0x00000000)

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

using int32x4
 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4
 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16
 = __attribute__((__vector_size__(16 * sizeof(float)))) float;


#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define half _Float16
#define __float2half_rn(x) half(x)

#define htanh tanhf
#define htan tanf
#define hatan atanf
#define herf erff

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16
#define hexp __ocml_exp_f16

// Pack two half values.
inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// There is no make_int8 in cuda, but TVM codegen seem to use it
inline __device__ longlong4 make_int8(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7) {
  int2 i0 = make_int2(x0, x1);
  int2 i1 = make_int2(x2, x3);
  int2 i2 = make_int2(x4, x5);
  int2 i3 = make_int2(x6, x7);
  long long l0 = *(long long*)&i0;
  long long l1 = *(long long*)&i1;
  long long l2 = *(long long*)&i2;
  long long l3 = *(long long*)&i3;
  return make_longlong4(l0, l1, l2, l3);
}

using float16_t = _Float16;
using float16x2
 = __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4
 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8
 = __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16
 = __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;
using int32x4
 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4
 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16
 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using bfloat16_t = hip_bfloat16;
using bfloat16x4
 = __attribute__((__vector_size__(4 * sizeof(bfloat16_t)))) float16_t;
__global__ void __launch_bounds__(256) Fused(half* __restrict__ input0, half* __restrict__ input1, half* __restrict__ output0) {
  
  float mediate0_warp[128];
  __shared__ half input0_shared[8192];
  __shared__ half input1_shared[4096];
  half input0_shared_warp[32];
  half input1_shared_warp[16];

  const int MAX_BLOCK_N = 10;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int i_2_init = 0; i_2_init < 8; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        mediate0_warp[(((i_2_init * 16) + (j_2_init * 4)) + local_id)] = 0.000000e+00f;
      }
    }
  }
  for (int k_0 = 0; k_0 < 1792; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 4; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(uint4*)(input0_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(input0 + ((((((((int)blockIdx.y) * 14680064) + (ax0_ax1_ax2_ax3_0_fused_0 * 3670016)) + (((int)threadIdx.y) * 1835008)) + (((int)threadIdx.z) * 917504)) + (k_0 * 512)) + (((int)threadIdx.x) * 8)));
    }
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 2; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
      *(uint4*)(input1_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 2048) + (((int)threadIdx.y) * 1024)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(input1 + ((((((((int)blockIdx.x) * 7340032) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 3670016)) + (((int)threadIdx.y) * 1835008)) + (((int)threadIdx.z) * 917504)) + (k_0 * 512)) + (((int)threadIdx.x) * 8)));
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
          input0_shared_warp[((ax0 * 4) + local_id_1)] = input0_shared[(((((((int)threadIdx.y) * 4096) + (ax0 * 512)) + (k_1 * 256)) + (((int)threadIdx.x) * 4)) + local_id_1)];
        }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {
        for (int local_id_2 = 0; local_id_2 < 4; ++local_id_2) {
          input1_shared_warp[((ax0_1 * 4) + local_id_2)] = input1_shared[(((((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (k_1 * 256)) + (((int)threadIdx.x) * 4)) + local_id_2)];
        }
      }
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          {
    *(((float32x4*)mediate0_warp) + ((i_2 * 4) + j_2)) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)input0_shared_warp) + i_2),
                  *(((float16x4*)input1_shared_warp) + j_2),
                  *(((float32x4*)mediate0_warp) + ((i_2 * 4) + j_2)), 0, 0, 0);
  };
        }
      }
    }
  }
  for (int ax0_2 = 0; ax0_2 < 8; ++ax0_2) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int local_id = 0; local_id < 4; ++local_id) {
(&(output0[((((((((int)blockIdx.y) * 3670016) + (((int)threadIdx.y) * 1835008)) + (ax0_2 * 229376)) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax1 * 256))]))[(((((threadIdx.x / 16) * 4) + local_id) * 16) + (threadIdx.x % 16))] = (half)mediate0_warp[((ax0_2 * 16) + (ax1 * 4)) + local_id];
}
;
    }
  }
}



extern "C" void call(half* args0, half* args1, half* args2) {
    Fused<<<dim3(112, 16, 1), dim3(64, 2, 2)>>>(args0, args1, args2);
}

extern "C" float profile(half* args0, half* args1, half* args2) {
    float ms;
    hipEvent_t start, stop;
    hipEventCreateWithFlags(&start, hipEventDefault);
    hipEventCreateWithFlags(&stop, hipEventDefault);
    hipEventRecord(start, 0);
    Fused<<<dim3(112, 16, 1), dim3(64, 2, 2)>>>(args0, args1, args2);
    if (hipEventRecord(stop, 0) != hipSuccess) return -1;
    if (hipEventSynchronize(stop) != hipSuccess) return -1;
    if (hipGetLastError() != hipSuccess) return -1;
    hipEventElapsedTime(&ms, start, stop);
    int repeats = int(ceil(100.0 / ms));
    if (repeats <= 3) repeats = 5;
    hipEventRecord(start, 0);
    for (int _ = 0; _ < repeats; _++)
        Fused<<<dim3(112, 16, 1), dim3(64, 2, 2)>>>(args0, args1, args2);
    if (hipEventRecord(stop, 0) != hipSuccess) return -1;
    if (hipEventSynchronize(stop) != hipSuccess) return -1;
    if (hipGetLastError() != hipSuccess) return -1;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return ms / repeats;
}
