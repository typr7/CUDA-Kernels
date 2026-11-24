#include <cstdint>
#include <cassert>
#include <cstddef>

#include "scan.cuh"
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

namespace {
  constexpr uint32_t kNumThreads      = 256;
  constexpr uint32_t kItemsPerThread  = 16;
  constexpr uint32_t kItemsPerBlock   = kNumThreads * kItemsPerThread;

  constexpr uint32_t kPass2NumThreads = 1024;
  constexpr int64_t kMaxNumElems      = static_cast<int64_t>(kItemsPerBlock) * kPass2NumThreads;
}

// block scan

template <uint32_t kNumThreads, uint32_t kItemsPerThread, bool kInclusive>
requires (kNumThreads >= 32 && kNumThreads != 0 && (kNumThreads & (kNumThreads - 1)) == 0)
__device__
void block_scan(const int (&in)[kItemsPerThread],
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

    const int thread_prefix = warp_exclusive_scan(sum_val);

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

    const int thread_prefix = warp_exclusive_scan(sum_val);

    if (lane_id == kNumThreadsPerWarp - 1) {
      smem[warp_id] = thread_prefix + sum_val;
    }
    __syncthreads();
    
    int prefix = lane_id < kNumWarps ? smem[lane_id] : 0;
    prefix = warp_exclusive_scan<kNumWarps>(prefix);
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

/*
template <uint32_t kNumThreads, uint32_t kItemsPerThread, bool kInclusive>
__global__ __launch_bounds__(kNumThreads)
void device_sum_pass1(const int* in, int* out, int* __restrict__ temp, int n)
{
  constexpr uint32_t kItemsPerBlock = kNumThreads * kItemsPerThread;
  int ws[kItemsPerThread];
  const uint32_t offset = blockIdx.x * kItemsPerBlock + threadIdx.x * kItemsPerThread;
  if constexpr (kItemsPerThread % 4) {
    uint32_t i = 0;
#pragma unroll
    for (; i < kItemsPerThread; i += 4) {
      if (offset + i + 3 >= n) {
        break;
      }
      as<int4>(&ws[i]) = as<int4>(in + offset);
    }
#pragma unroll
    for (; i < kItemsPerThread; i++) {
      ws[i] = offset + i < n ? in[offset + i] : 0;
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      ws[i] = offset + i < n ? in[offset + i] : 0;
    }
  }

  block_sum<kNumThreads, kItemsPerThread, kInclusive>(ws, ws, temp + blockIdx.x);

  if (blockIdx.x != gridDim.x - 1) {
    if constexpr (kItemsPerThread % 4) {
#pragma unroll
      for (uint32_t i = 0; i < kItemsPerThread; i += 4) {
        as<int4>(in + offset) = as<int4>(&ws[i]);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kItemsPerThread; i++) {
        in[offset + i] = ws[i];
      }
    }
  } else {
    if constexpr (kItemsPerThread % 4) {
      uint32_t i = 0;
#pragma unroll
      for (; i < kItemsPerThread; i += 4) {
        if (offset + i + 3 >= n) {
          break;
        }
        as<int4>(in + offset) = as<int4>(&ws[i]);
      }
#pragma unroll
      for (; i < kItemsPerThread && offset + i < n; i++) {
        in[offset + i] = ws[i];
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kItemsPerThread && offset + i < n; i++) {
        in[offset + i] = ws[i];
      }
    }
  }
}
*/

template <uint32_t kNumThreads, uint32_t kItemsPerThread>
struct TxSharedStorage {
  static constexpr uint32_t kItemsPerBlock = kNumThreads * kItemsPerThread;
  static constexpr bool kNeedPadding = kItemsPerThread > 4 && (kItemsPerThread & (kItemsPerThread - 1)) == 0;
  static constexpr uint32_t kSize = kItemsPerBlock + (kNeedPadding ? kItemsPerBlock / 32 : 0);
  alignas(16) int buff[kSize];
};

template <uint32_t kItemsPerThread>
requires (kItemsPerThread > 0)
__device__ __forceinline__
void warp_striped_to_blocked(int (&ws)[kItemsPerThread], int* __restrict__ tx_buff) {
  if constexpr (kItemsPerThread > 1) {
    constexpr bool kNeedPadding = kItemsPerThread > 4 && (kItemsPerThread & (kItemsPerThread - 1)) == 0;
    const uint32_t lane_id = threadIdx.x % kWarpSize;
    const uint32_t warp_offset = (threadIdx.x / kWarpSize) * kWarpSize * kItemsPerThread;

#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      uint32_t offset = warp_offset + i * kWarpSize + lane_id;
      if constexpr (kNeedPadding) {
        offset += offset >> 5u;
      }
      tx_buff[offset] = ws[i];
    }
    __syncwarp();

#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      uint32_t offset = warp_offset + lane_id * kItemsPerThread + i;
      if constexpr (kNeedPadding) {
        offset += offset >> 5u;
      }
      ws[i] = tx_buff[offset];
    }
    __syncwarp();
  }
}

template <uint32_t kItemsPerThread>
requires (kItemsPerThread > 0)
__device__ __forceinline__
void blocked_to_warp_striped(int (&ws)[kItemsPerThread], int* __restrict__ tx_buff) {
  if constexpr (kItemsPerThread > 1) {
    constexpr bool kNeedPadding = kItemsPerThread > 4 && (kItemsPerThread & (kItemsPerThread - 1)) == 0;
    const uint32_t lane_id = threadIdx.x % kWarpSize;
    const uint32_t warp_offset = (threadIdx.x / kWarpSize) * kWarpSize * kItemsPerThread;

#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      uint32_t offset = warp_offset + lane_id * kItemsPerThread + i;
      if constexpr (kNeedPadding) {
        offset += offset >> 5u;
      }
      tx_buff[offset] = ws[i];
    }
    __syncwarp();
  
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      uint32_t offset = warp_offset + i * 32u + lane_id;
      if constexpr (kNeedPadding) {
        offset += offset >> 5u;
      }
      ws[i] = tx_buff[offset];
    }
    __syncwarp();
  }
}

template <uint32_t kNumThreads, uint32_t kItemsPerThread, bool kInclusive, bool kIsLastBlock>
__device__ __forceinline__
void consume_tile(const int* tile_in,
                  int* tile_out,
                  int* __restrict__ block_sum,
                  int* __restrict__ tx_buff,
                  int num_valid_in_tile) {
  const uint32_t lane_id = threadIdx.x % 32u;
  const uint32_t warp_offset = (threadIdx.x / 32u) * 32u * kItemsPerThread;
  int ws[kItemsPerThread];

  if constexpr (kIsLastBlock) {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      const int pos = static_cast<int>(warp_offset + lane_id + i * 32u);
      ws[i] = (pos < num_valid_in_tile) ? tile_in[pos] : 0;
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      ws[i] = tile_in[warp_offset + lane_id + i * 32u];
    }
  }

  warp_striped_to_blocked(ws, tx_buff);
  block_scan<kNumThreads, kItemsPerThread, kInclusive>(ws, ws, block_sum);
  blocked_to_warp_striped(ws, tx_buff);

  if constexpr (kIsLastBlock) {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      const int pos = static_cast<int>(warp_offset + lane_id + i * 32u);
      if (pos < num_valid_in_tile) {
        tile_out[pos] = ws[i];
      }
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      tile_out[warp_offset + lane_id + i * 32u] = ws[i];
    }
  }
}

template <uint32_t kNumThreads, uint32_t kItemsPerThread, bool kInclusive>
__global__ __launch_bounds__(kNumThreads)
void scan_pass1_kernel(const int* in,
                             int* out,
                             int* __restrict__ block_sums,
                             int n) {
  constexpr int kItemsPerBlock = static_cast<int>(kNumThreads * kItemsPerThread);
  const int tile_offset = static_cast<int>(blockIdx.x * kItemsPerBlock);
  const int num_remaining = n - tile_offset;

  assert(num_remaining > 0);

  using Tx = TxSharedStorage<kNumThreads, kItemsPerThread>;
  __shared__ Tx tx;

  if (num_remaining >= kItemsPerBlock) {
    consume_tile<kNumThreads, kItemsPerThread, kInclusive, false>(in + tile_offset,
                                                                  out + tile_offset,
                                                                  block_sums + blockIdx.x,
                                                                  tx.buff,
                                                                  kItemsPerBlock);
  } else {
    consume_tile<kNumThreads, kItemsPerThread, kInclusive, true>(in + tile_offset,
                                                                 out + tile_offset,
                                                                 block_sums + blockIdx.x,
                                                                 tx.buff,
                                                                 num_remaining);
  }
}

template <uint32_t kNumThreads>
requires (kNumThreads >= 32 && (kNumThreads & (kNumThreads - 1)) == 0)
__global__ __launch_bounds__(kNumThreads)
void scan_pass2_kernel(int* __restrict__ block_sums, int num_tiles) {
  constexpr uint32_t kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % kWarpSize;
  const uint32_t warp_id = tid / kWarpSize;

  if constexpr (kNumWarps == 1) {
    const int val = tid < num_tiles ? block_sums[tid] : 0;
    const int prefix_in_warp = warp_exclusive_scan(val);

    if (tid < num_tiles) {
      block_sums[tid] = prefix_in_warp;
    }
  } else {
    __shared__ int smem[kNumWarps];

    const int val = tid < num_tiles ? block_sums[tid] : 0;
    const int prefix_in_warp = warp_exclusive_scan(val);

    if (lane_id == kWarpSize - 1) {
      smem[warp_id] = prefix_in_warp + val;
    }
    __syncthreads();

    if (warp_id == 0) {
      int warp_prefix = lane_id < kNumWarps ? smem[lane_id] : 0;
      warp_prefix = warp_exclusive_scan(warp_prefix);
      if (lane_id < kNumWarps) {
        smem[lane_id] = warp_prefix;
      }
    }
    __syncthreads();

    if (tid < num_tiles) {
      block_sums[tid] = prefix_in_warp + smem[warp_id];
    }
  }
}

template <uint32_t kNumThreads, uint32_t kItemsPerThread>
requires (kNumThreads >= 32 && (kNumThreads & (kNumThreads - 1)) == 0)
__global__ __launch_bounds__(kNumThreads)
void scan_pass3_kernel(int* __restrict__ out, const int* __restrict__ block_sums, int n) {
  constexpr uint32_t kItemsPerBlock = kNumThreads * kItemsPerThread;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane_id = tid % kWarpSize;
  const uint32_t warp_id = tid / kWarpSize;

  int ws[kItemsPerThread];

  const int block_prefix = block_sums[blockIdx.x];

  const int tile_offset = blockIdx.x * kItemsPerBlock;
  const int num_remaining = n - tile_offset;

  const int warp_offset = warp_id * kWarpSize * kItemsPerThread;
  int* tile_out = out + tile_offset;

  if (num_remaining >= kItemsPerBlock) {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      ws[i] = tile_out[warp_offset + i * kWarpSize + lane_id] + block_prefix;
    }
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      tile_out[warp_offset + i * kWarpSize + lane_id] = ws[i];
    }
  } else {
#pragma unroll
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      const int pos = warp_offset + i * kWarpSize + lane_id;
      ws[i] = pos < num_remaining ? tile_out[pos] + block_prefix : 0;
    }
    for (uint32_t i = 0; i < kItemsPerThread; i++) {
      const int pos = warp_offset + i * kWarpSize + lane_id;
      if (pos < num_remaining) {
        tile_out[pos] = ws[i];
      }
    }
  }
}

size_t get_workspace_size(int num_elems) {
  assert(num_elems >= 0 && num_elems <= kMaxNumElems);
  return static_cast<size_t>(cdiv<int>(num_elems, kItemsPerBlock)) * sizeof(int);
}

template <bool kInclusive>
void scan(const int* in, int* out, void* workspace, int num_elems) {
  if (num_elems <= 0) {
    return;
  }

  assert(in);
  assert(out);
  assert(workspace);
  assert(num_elems >= 0 && num_elems <= kMaxNumElems);

  const int num_tiles = cdiv<int>(num_elems, kItemsPerBlock);
  int* block_sums = static_cast<int*>(workspace);

  const dim3 p13_grid_size(num_tiles);
  const dim3 p13_block_size(kNumThreads);
  const dim3 p2_grid_size(1);
  const dim3 p2_block_size(kPass2NumThreads);

  scan_pass1_kernel<kNumThreads, kItemsPerThread, kInclusive>
    <<<p13_grid_size, p13_block_size>>>(in, out, block_sums, num_elems);

  scan_pass2_kernel<kPass2NumThreads><<<p2_grid_size, p2_block_size>>>(block_sums, num_tiles);

  scan_pass3_kernel<kNumThreads, kItemsPerThread>
    <<<p13_grid_size, p13_block_size>>>(out, block_sums, num_elems);
}

void inclusive_scan(const int* in, int* out, void* workspace, int num_elems) {
  scan<true>(in, out, workspace, num_elems);
}

void exclusive_scan(const int* in, int* out, void* workspace, int num_elems) {
  scan<false>(in, out, workspace, num_elems);
}
