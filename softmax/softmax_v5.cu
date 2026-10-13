#include <cstdint>
#include <cassert>

#include <math_constants.h>

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

    const uint32_t lane_id = tid & (WARP_SIZE - 1);
    const uint32_t warp_id = tid >> 5;

    float* max_vals = vec;
    float* sum_vals = vec + WARPS_PER_BLOCK;

    const uint32_t H_f4 = H / 4;
    const float4* in_begin = reinterpret_cast<const float4*>(in + blockIdx.x * H);

    // float max_val = -CUDART_INF_F;
    float max_val0 = -CUDART_INF_F;
    float max_val1 = -CUDART_INF_F;
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        // max_val = fmaxf(max_val, fmaxf(fmaxf(f4.x, f4.y), fmaxf(f4.z, f4.w)));
        max_val0 = fmaxf(max_val0, fmaxf(f4.x, f4.y));
        max_val1 = fmaxf(max_val1, fmaxf(f4.z, f4.w));
    }

    max_val0 = warp_reduce_max<false>(fmaxf(max_val0, max_val1));
    if (lane_id == 0) {
        max_vals[warp_id] = max_val0;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val0 = tid < WARPS_PER_BLOCK ? max_vals[tid] : -CUDART_INF_F;
        max_val0 = warp_reduce_max<false>(max_val0);
        if (lane_id == 0) {
            max_vals[0] = max_val0;
        }
    }
    __syncthreads();
    max_val0 = max_vals[0];

    float sum_val0 = 0.f;
    float sum_val1 = 0.f;
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        sum_val0 += __expf(f4.x - max_val0)
                 + __expf(f4.y - max_val0);
        sum_val1 += __expf(f4.z - max_val0)
                 + __expf(f4.w - max_val0);
    }

    sum_val0 = warp_reduce_sum<false>(sum_val0 + sum_val1);
    if (lane_id == 0) {
        sum_vals[warp_id] = sum_val0;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum_val0 = tid < WARPS_PER_BLOCK ? sum_vals[tid] : 0.f;
        sum_val0 = warp_reduce_sum<false>(sum_val0);
        if (tid == 0) {
            sum_vals[0] = __frcp_rn(sum_val0);
        }
    }
    __syncthreads();

    float4* out_begin = reinterpret_cast<float4*>(out + blockIdx.x * H);
    const float norm = sum_vals[0];
    for (uint32_t i = tid; i < H_f4; i += TB_SIZE) {
        const float4 f4 = in_begin[i];
        out_begin[i] = make_float4(
            __expf(f4.x - max_val0) * norm,
            __expf(f4.y - max_val0) * norm,
            __expf(f4.z - max_val0) * norm,
            __expf(f4.w - max_val0) * norm
        );
    }
}

template <uint32_t TB_SIZE, uint32_t FL4S_PER_THREAD>
__launch_bounds__(TB_SIZE)
__global__
void softmax_kernel_h4096(const float* in, float* out, int N)
{
    constexpr uint32_t H = 4096;

    extern __shared__ float vec[];
    const uint32_t tid = threadIdx.x;

    constexpr uint32_t WARPS_PER_BLOCK = TB_SIZE / WARP_SIZE;

    const uint32_t lane_id = tid & (WARP_SIZE - 1);
    const uint32_t warp_id = tid >> 5;

    float* max_vals = vec;
    float* sum_vals = vec + WARPS_PER_BLOCK;

    constexpr uint32_t H_f4 = H / 4;
    const float4* in_begin = reinterpret_cast<const float4*>(in + blockIdx.x * H);

    float4 reg[FL4S_PER_THREAD];

    float max_val0 = -CUDART_INF_F;
    float max_val1 = -CUDART_INF_F;
    #pragma unroll
    for (uint32_t i = 0; i < FL4S_PER_THREAD; i++) {
        const uint32_t idx = tid + i * TB_SIZE;
        const float4& f4 = reg[i] = in_begin[idx];
        max_val0 = fmaxf(max_val0, fmaxf(f4.x, f4.y));
        max_val1 = fmaxf(max_val1, fmaxf(f4.z, f4.w));
    }

    max_val0 = warp_reduce_max<false>(fmaxf(max_val0, max_val1));
    if (lane_id == 0) {
        max_vals[warp_id] = max_val0;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val0 = tid < WARPS_PER_BLOCK ? max_vals[tid] : -CUDART_INF_F;
        max_val0 = warp_reduce_max<false>(max_val0);
        if (lane_id == 0) {
            max_vals[0] = max_val0;
        }
    }
    __syncthreads();
    max_val0 = max_vals[0];

    float sum_val0 = 0.f;
    float sum_val1 = 0.f;
    #pragma unroll
    for (uint32_t i = 0; i < FL4S_PER_THREAD; i++) {
        float4& f4 = reg[i];
        f4.x = __expf(f4.x - max_val0);
        f4.y = __expf(f4.y - max_val0);
        f4.z = __expf(f4.z - max_val0);
        f4.w = __expf(f4.w - max_val0);
        sum_val0 += f4.x + f4.y;
        sum_val1 += f4.z + f4.w;
    }

    sum_val0 = warp_reduce_sum<false>(sum_val0 + sum_val1);
    if (lane_id == 0) {
        sum_vals[warp_id] = sum_val0;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum_val0 = tid < WARPS_PER_BLOCK ? sum_vals[tid] : 0.f;
        sum_val0 = warp_reduce_sum<false>(sum_val0);
        if (lane_id == 0) {
            sum_vals[0] = __frcp_rn(sum_val0);
        }
    }
    __syncthreads();

    float4* out_begin = reinterpret_cast<float4*>(out + blockIdx.x * H);
    const float norm = sum_vals[0];
    #pragma unroll
    for (uint32_t i = 0; i < FL4S_PER_THREAD; i++) {
        const uint32_t idx = tid + i * TB_SIZE;
        float4& f4 = reg[i];
        f4.x *= norm;
        f4.y *= norm;
        f4.z *= norm;
        f4.w *= norm;
        out_begin[idx] = f4;
    }
}

}

void softmax_v5(const float* in, float* out, int N, int H)
{
    constexpr uint32_t TB_SIZE = 256;
    constexpr uint32_t WARPS_PER_TBLOCK = TB_SIZE / WARP_SIZE;
    
    constexpr uint32_t SMEM_BYTE_SIZE = 2 * WARPS_PER_TBLOCK * sizeof(float);

    assert(H % 4 == 0);

    if (H == 4096) {
        static_assert(1024 % TB_SIZE == 0);
        softmax_kernel_h4096<TB_SIZE, 4096 / 4 / TB_SIZE>
            <<<N, TB_SIZE, SMEM_BYTE_SIZE>>>(in, out, N);
    } else {
        softmax_kernel<TB_SIZE>
            <<<N, TB_SIZE, SMEM_BYTE_SIZE>>>(in, out, N, H);
    }
}
