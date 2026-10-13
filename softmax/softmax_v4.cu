#include <math_constants.h>
#include <cstdint>

#include "common.hpp"


namespace
{

template <uint32_t TB_SIZE>
__global__
void softmax_kernel(const float* in, float* out, int N, int H)
{
    extern __shared__ float vec[];
    const uint32_t tid = threadIdx.x;

    constexpr uint32_t WARPS_PER_BLOCK = TB_SIZE / WARP_SIZE;

    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;

    float* max_vals = vec;
    float* sum_vals = vec + WARPS_PER_BLOCK;

    const uint32_t H_f4 = H / 4;
    const float4* in_begin = reinterpret_cast<const float4*>(in + blockIdx.x * H);

    float max_val = -CUDART_INF_F;
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        max_val = fmaxf(max_val, fmaxf(fmaxf(f4.x, f4.y), fmaxf(f4.z, f4.w)));
    }

    max_val = warp_reduce_max<false>(max_val);
    if (lane_id == 0) {
        max_vals[warp_id] = max_val;
    }
    __syncthreads();

    max_val = tid < WARPS_PER_BLOCK ? max_vals[tid] : -CUDART_INF_F;
    max_val = warp_reduce_max<false>(max_val);
    if (tid == 0) {
        max_vals[0] = max_val;
    }
    __syncthreads();
    max_val = max_vals[0];

    float sum_val = 0.f;
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        sum_val += expf(f4.x - max_val)
                 + expf(f4.y - max_val)
                 + expf(f4.z - max_val)
                 + expf(f4.w - max_val);
    }

    sum_val = warp_reduce_sum<false>(sum_val);
    if (lane_id == 0) {
        sum_vals[warp_id] = sum_val;
    }
    __syncthreads();
    
    sum_val = tid < WARPS_PER_BLOCK ? sum_vals[tid] : 0.f;
    sum_val = warp_reduce_sum<false>(sum_val);
    if (tid == 0) {
        sum_vals[0] = sum_val;
    }
    __syncthreads();

    float4* out_begin = reinterpret_cast<float4*>(out + blockIdx.x * H);
    const float norm = 1.f / sum_vals[0];
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        out_begin[i] = make_float4(
            expf(f4.x - max_val) * norm,
            expf(f4.y - max_val) * norm,
            expf(f4.z - max_val) * norm,
            expf(f4.w - max_val) * norm
        );
    }
}

}

// no bound check
void softmax_v4(const float* in, float* out, int N, int H)
{
    constexpr uint32_t TB_SIZE = 256;
    constexpr uint32_t WARPS_PER_TBLOCK = TB_SIZE / WARP_SIZE;
    
    constexpr uint32_t SMEM_BYTE_SIZE = 2 * WARPS_PER_TBLOCK * sizeof(float);
    softmax_kernel<TB_SIZE>
        <<<N, TB_SIZE, SMEM_BYTE_SIZE>>>(in, out, N, H);
}
