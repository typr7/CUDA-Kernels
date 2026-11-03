#include <cstdint>

#include "common.cuh"

/*
// Hillis Steele and Blelloch example

// single block kernel, n <= kNumThreads
// depth optimal O(logN)
// work O(NlogN)
template <uint32_t kNumThreads>
__global__
void scan_hillis_steele_inclusive(const int* __restrict__ in, int* __restrict__ out, int n) {
  // double buffer to avoid rw data race
  __shared__ int smem[2][kNumThreads];
  const uint32_t tid = threadIdx.x;
  int r = 0;
  smem[r][tid] = tid < n ? in[tid] : 0;

#pragma unroll
  for (uint32_t stride = 1; stride < kNumThreads; stride <<= 1) {
    const int w = r ^ 0b1;
    __syncthreads();
    int val = smem[r][tid];
    if (tid >= stride) {
      val += smem[r][tid - stride];
    }
    smem[w][tid] = val;
    r = w;
  }

  __syncthreads();
  if (tid < n) {
    out[tid] = smem[r][tid];
  }
}

// single block kernel, n <= kNumThreads
// depth optimal O(logN)
// work optimal O(N)
template <uint32_t kNumThreads>
__global__
void scan_blelloch_exclusive(const int* __restrict__ in,
                             int* __restrict__ out,
                             int n,
                             int* __restrict__ block_sum = nullptr) {
  __shared__ int smem[kNumThreads];
  const uint32_t tid = threadIdx.x;
  smem[tid] = tid < n ? in[tid] : 0;

#pragma unroll
  for (uint32_t stride = 1; stride < kNumThreads; stride <<= 1) {
    const int offset = stride << 1;
    __syncthreads();
    if ((tid + 1) % offset == 0) {
      smem[tid] += smem[tid - stride];
    }
  }

  __syncthreads();
  if (tid == kNumThreads - 1) {
    if (block_sum) {
      *block_sum = smem[tid];
    }
    smem[tid] = 0;
  }

#pragma unroll
  for (uint32_t stride = kNumThreads >> 1; stride > 0; stride >>= 1) {
    const int offset = stride << 1;
    __syncthreads();
    if ((tid + 1) % offset == 0) {
      const int tmp = smem[tid - stride];
      smem[tid - stride] = smem[tid];
      smem[tid] += tmp;
    }
  }

  __syncthreads();
  if (tid < n) {
    out[tid] = smem[tid];
  }
}
*/


// block scan

// n <= kNumThreads
template <uint32_t kNumThreads, bool kInclusive>
requires (kNumThreads >= 32 && kNumThreads != 0 && (kNumThreads & (kNumThreads - 1)) == 0) // is power of two
__global__ __launch_bounds__(kNumThreads)
void scan_block(const int* __restrict__ in,
                int* __restrict__ out,
                int n,
                int* __restrict__ block_sum = nullptr) {
  constexpr uint32_t kNumThreadsPerWarp = 32u;
  constexpr uint32_t kNumWarps = (kNumThreads + kNumThreadsPerWarp - 1) / kNumThreadsPerWarp;

  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid / kNumThreadsPerWarp;
  const uint32_t lane_id = tid % kNumThreadsPerWarp;

  __shared__ int smem[kNumWarps];

  const int x = tid < n ? in[tid] : 0;
  int val = warp_inclusive_sum(x);

  if (lane_id == kNumThreadsPerWarp - 1) {
    smem[warp_id] = val;
  }

  __syncthreads();

  if (warp_id == 0) {
    int t = lane_id < kNumWarps ? smem[lane_id] : 0;
    t = warp_exclusive_sum(t);
    if (lane_id < kNumWarps) {
      smem[lane_id] = t;
    }
  }
  __syncthreads();

  int warp_prefix = smem[warp_id];

  if (tid < n) {
    if constexpr (kInclusive) {
      out[tid] = val + warp_prefix;
    } else {
      out[tid] = val + warp_prefix - x;
    }
  }

  if (tid == kNumThreads - 1 && block_sum) {
    *block_sum = val + warp_prefix;
  }
}

template <uint32_t kNumThreads, uint32_t kItemsPerThread, bool kInclusive>
requires (kNumThreads >= 32 && kNumThreads != 0 && (kNumThreads & (kNumThreads - 1)) == 0)
__device__
void block_sum(const int (&in)[kItemsPerThread],
               int (&out)[kItemsPerThread],
               int* __restrict__ block_sum = nullptr) {
  constexpr uint32_t kNumThreadsPerWarp = 32u;
  constexpr uint32_t kNumWarps = (kNumThreads + kNumThreadsPerWarp - 1) / kNumThreadsPerWarp;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % kNumThreadsPerWarp;

  if constexpr (kNumWarps == 1) {
    int ws[kItemsPerThread];
    
    int sum_val = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      ws[i] = sum_val = in[i] + sum_val;
    }

    const int thread_prefix = warp_exclusive_sum(sum_val);

#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      if constexpr (kInclusive) {
        out[i] = thread_prefix + ws[i];
      } else {
        out[i] = thread_prefix + (i == 0 ? 0 : ws[i - 1]);
      }
    }

    if (tid == kNumThreads - 1 && block_sum) {
      *block_sum = ws[kItemsPerThread - 1] + thread_prefix;
    }

  } else {
    const uint32_t warp_id = tid / kNumThreadsPerWarp;

    __shared__ int smem[kNumWarps];

    int ws[kItemsPerThread];

    int sum_val = 0;
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      ws[i] = sum_val = in[i] + sum_val;
    }

    const int thread_prefix = warp_exclusive_sum(sum_val);

    if (lane_id == kNumThreadsPerWarp - 1) {
      smem[warp_id] = thread_prefix + sum_val;
    }
    __syncthreads();
    
    int prefix = lane_id < kNumWarps ? smem[lane_id] : 0;
    prefix = warp_exclusive_sum<kNumWarps>(prefix);
    prefix = __shfl_sync(0xffffffff, prefix, warp_id) + thread_prefix;

#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      if constexpr (kInclusive) {
        out[i] = prefix + ws[i];
      } else {
        out[i] = prefix + (i == 0 ? 0 : ws[i - 1]);
      }
    }

    if (tid == kNumThreads - 1 && block_sum) {
      *block_sum = ws[kItemsPerThread - 1] + prefix;
    }
  }
}
