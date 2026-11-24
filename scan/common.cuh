#pragma once

#include <cstdint>


inline constexpr uint32_t kWarpSize = 32u;


template <typename T>
T cdiv(T a, T b) {
  return (a + b - 1) / b;
}

template <uint32_t kWidth = 32>
requires (kWidth <= 32 && kWidth != 0 && (kWidth & (kWidth - 1)) == 0)
__device__ __forceinline__
int warp_inclusive_scan(int val) {
  const uint32_t lane_id = threadIdx.x % 32u;
#pragma unroll
  for (uint32_t stride = 1; stride < kWidth; stride <<= 1) {
    const int t = __shfl_up_sync(0xffffffff, val, stride);
    if (lane_id >= stride) {
      val += t;
    }
  }
  return val;
}

template <uint32_t kWidth = 32>
__device__ __forceinline__
int warp_exclusive_scan(int val) {
  return warp_inclusive_scan<kWidth>(val) - val;
}

template <typename ToType, typename FromType>
__device__ __forceinline__
ToType& as(FromType* p) {
  return *reinterpret_cast<ToType*>(p);
}
