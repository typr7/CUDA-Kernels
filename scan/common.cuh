#pragma once

#include <cstdint>


__device__ __forceinline__
uint32_t get_lane_id() {
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

/*
// warp scan
// Hillis-Steele
__device__ __forceinline__
int warp_inclusive_sum(int val) {
  const uint32_t lane_id = get_lane_id();
#pragma unroll
  for (uint32_t stride = 1; stride < 32; stride <<= 1) {
    const int t = __shfl_up_sync(0xffffffff, val, stride);
    if (lane_id >= stride) {
      val += t;
    }
  }
  
  return val;
}
*/

template <uint32_t kWidth = 32>
requires (kWidth <= 32 && kWidth != 0 && (kWidth & (kWidth - 1)) == 0)
__device__ __forceinline__
int warp_inclusive_sum(int val) {
  const uint32_t lane_id = get_lane_id();
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
int warp_exclusive_sum(int val) {
  return warp_inclusive_sum<kWidth>(val) - val;
}

