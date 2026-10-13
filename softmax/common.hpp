#pragma once

#include <cstdint>

#include <cuda_runtime.h>


inline constexpr uint32_t WARP_SIZE = 32;

template <bool kButterFly>
__device__ __forceinline__
float warp_reduce_max(float val)
{
    if constexpr (kButterFly) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    } else {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    }
    return val;
}

template <bool kButterFly>
__device__ __forceinline__
float warp_reduce_sum(float val)
{
    if constexpr (kButterFly) {
        val += __shfl_xor_sync(0xffffffff, val, 16);
        val += __shfl_xor_sync(0xffffffff, val, 8);
        val += __shfl_xor_sync(0xffffffff, val, 4);
        val += __shfl_xor_sync(0xffffffff, val, 2);
        val += __shfl_xor_sync(0xffffffff, val, 1);
    } else {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    return val;
}