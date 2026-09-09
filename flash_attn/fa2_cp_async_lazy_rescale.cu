#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>

#include "common.hpp"


namespace
{

// m16n8k16
constexpr uint32_t kMmaM = 16;
constexpr uint32_t kMmaN = 8;
constexpr uint32_t kMmaK = 16;

template <typename T, uint32_t kBr, uint32_t kBc, uint32_t kHeadDim>
struct SharedStorage
{
    union
    {
        struct
        {
            union
            {
                alignas(128) T smem_q[kBr][kHeadDim];
                alignas(128) T smem_k[kBc][kHeadDim];
            };
            alignas(128) T smem_v[kBc][kHeadDim];
        };

        alignas(128) T smem_o[kBr][kHeadDim];
    };
};

// Same 128-byte swizzle as the ldmatrix consumers; zero-fill OOB rows like TMA.
template <uint32_t kNumThreads, typename T, uint32_t kNumRows, uint32_t kNumCols>
__device__ __forceinline__
void load_tile_to_smem_async(
    const T* __restrict__ src,
    T (&smem)[kNumRows][kNumCols],
    uint32_t src_stride,
    uint32_t src_rows
)
{
    constexpr uint32_t kNumColsU4 = kNumCols / kNumBf16sPerVector;

#pragma unroll
    for (uint32_t i = threadIdx.x; i < kNumRows * kNumColsU4; i += kNumThreads) {
        const uint32_t y = i / kNumColsU4;
        const uint32_t x = (i % kNumColsU4) * kNumBf16sPerVector;
        const uint32_t src_size = y < src_rows ? 16 : 0;
        cp_async_cg_16(src + y * src_stride + x, &smem[y][swizzle_tma_128b(y, x)], src_size);
    }
    cp_async_commit();
}

template <
    typename T,
    uint32_t kNumTiles,
    uint32_t kNumRegs,
    uint32_t kNumRows,
    uint32_t kNumCols
>
__device__ __forceinline__
void load_mma_tile_q_s2r(
    uint32_t (&reg)[kNumTiles][kNumRegs],
    const T (&smem)[kNumRows][kNumCols]
)
{
    const uint32_t warp_id = threadIdx.x / kNumThreadsPerWarp;
    const uint32_t lane_id = threadIdx.x % kNumThreadsPerWarp;
    const uint32_t y = warp_id * kMmaM + lane_id % kMmaM;
    const uint32_t x_offset = lane_id / kMmaM * (kMmaK / 2);
#pragma unroll
    for (uint32_t i = 0; i < kNumTiles; i++) {
        const uint32_t x = i * kMmaK + x_offset;
        ldmatrix_x4(reg[i], &smem[y][swizzle_tma_128b(y, x)]);
    }
}

template <
    typename T,
    uint32_t kNumQWarps,
    uint32_t kNumKWarps,
    uint32_t kNumQMmaTilesPerWarp,
    uint32_t kNumKMmaTilesPerWarp,
    uint32_t kHeadDim
> requires (std::is_same_v<T, __nv_bfloat16>)
__global__ __launch_bounds__(kNumQWarps * kNumKWarps * kNumThreadsPerWarp)
void fa2_cp_async_lazy_rescale(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O, // [batch_size, q_seq_len, q_head_num, head_dim]
    uint32_t seq_len,
    uint32_t q_head_num,
    uint32_t kv_head_num
)
{
    constexpr uint32_t kNumThreads = kNumQWarps * kNumKWarps * kNumThreadsPerWarp;
    constexpr uint32_t kBr = kNumQWarps * kNumQMmaTilesPerWarp * kMmaM;
    constexpr uint32_t kBc = kNumKWarps * kNumKMmaTilesPerWarp * kMmaN;
    constexpr uint32_t kNumQRegsPerThread =
        kMmaM * kMmaK * sizeof(T) / sizeof(uint32_t) / kNumThreadsPerWarp;
    constexpr uint32_t kNumAccRegsPerThread = kMmaM * kMmaN / kNumThreadsPerWarp;
    constexpr uint32_t kHeadDimDivMmaK = kHeadDim / kMmaK;
    constexpr uint32_t kHeadDimDivMmaN = kHeadDim / kMmaN;
    constexpr uint32_t kBcDivMmaN = kBc / kMmaN;
    constexpr uint32_t kBcDivMmaK = kBc / kMmaK;
    // constexpr float lazy_scale_threshold = 0x1p-8f;
    constexpr float kMaxExponentGap = 8.f;
    const float kScale = rsqrt(static_cast<float>(kHeadDim));
    const float kScaleLog2e = kScale * 1.44269504f;

    using Shared = SharedStorage<T, kBr, kBc, kHeadDim>;

    extern __shared__ Shared smem[];
    Shared& shared = smem[0];

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / kNumThreadsPerWarp;
    const uint32_t lane_id = tid % kNumThreadsPerWarp;
    const uint32_t q_tile_idx = blockIdx.x;
    const uint32_t q_head_idx = blockIdx.y;
    const uint32_t batch_idx = blockIdx.z;

    const uint32_t group_size = q_head_num / kv_head_num;
    const uint32_t kv_head_idx = q_head_idx / group_size;

    const uint32_t q_start_idx = q_tile_idx * kBr;
    const uint32_t warp_start_idx = q_start_idx + warp_id * kMmaM;

    const uint32_t q_stride = q_head_num * kHeadDim;
    const uint32_t kv_stride = kv_head_num * kHeadDim;
    const uint32_t batch_start_idx = batch_idx * seq_len;
    const uint64_t kv_offset =
        static_cast<uint64_t>(batch_start_idx) * kv_stride + kv_head_idx * kHeadDim;

    Q += (batch_start_idx + q_start_idx) * static_cast<uint64_t>(q_stride) + q_head_idx * kHeadDim;
    K += kv_offset;
    V += kv_offset;

    load_tile_to_smem_async<kNumThreads>(Q, shared.smem_q, q_stride, seq_len - q_start_idx);
    cp_async_wait_group<0>();
    __syncthreads();

    // load Q tile from smem to reg
    uint32_t q_reg[kHeadDim / kMmaK][kNumQRegsPerThread];
    load_mma_tile_q_s2r(q_reg, shared.smem_q);

    float m_ref[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.f, 0.f};

    float acc_o[kHeadDimDivMmaN][kNumAccRegsPerThread] = {0.f};
    const uint32_t kv_len = min(seq_len, q_start_idx + kBr);

    __syncthreads();
    for (uint32_t kv_start_idx = 0; kv_start_idx < kv_len; kv_start_idx += kBc) {
        load_tile_to_smem_async<kNumThreads>(
            K + kv_start_idx * kv_stride, shared.smem_k, kv_stride, seq_len - kv_start_idx
        );
        load_tile_to_smem_async<kNumThreads>(
            V + kv_start_idx * kv_stride, shared.smem_v, kv_stride, seq_len - kv_start_idx
        );
        // K is ready while V can overlap with QK^T and softmax.
        cp_async_wait_group<1>();
        __syncthreads();

        // S = QK^T
        float acc_s[kBcDivMmaN][kNumAccRegsPerThread] = {0.f};
#pragma unroll
        for (uint32_t k = 0; k < kHeadDimDivMmaK; k++) {
            const uint32_t x = k * kMmaK + ((lane_id / 8) & 0b1) * 8;
#pragma unroll
            for (uint32_t n = 0; n < kBcDivMmaN; n += 2) {
                uint32_t k_reg[4];
                const uint32_t y = n * kMmaN + (lane_id / 16) * 8 + lane_id % 8;
                ldmatrix_x4(k_reg, &shared.smem_k[y][swizzle_tma_128b(y, x)]);
                mma_m16n8k16(q_reg[k], k_reg, acc_s[n]);
                mma_m16n8k16(q_reg[k], k_reg + 2, acc_s[n + 1]);
            }
        }

        const uint32_t row0 = warp_start_idx + lane_id / 4;
        const uint32_t row1 = row0 + 8;
        const uint32_t col_base = kv_start_idx + (lane_id % 4) * 2;

        // online softmax
        float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};

        const bool need_causal_mask = warp_start_idx < kv_start_idx + kBc;

#pragma unroll
        for (uint32_t i = 0; i < kBcDivMmaN; i++) {
            const uint32_t col = col_base + i * kMmaN;
            float* s = acc_s[i];
            if (need_causal_mask) {
                s[0] = row0 >= col ? s[0] : -CUDART_INF_F;
                s[1] = row0 >= col + 1 ? s[1] : -CUDART_INF_F;
                s[2] = row1 >= col ? s[2] : -CUDART_INF_F;
                s[3] = row1 >= col + 1 ? s[3] : -CUDART_INF_F;
            }
            tile_max[0] = fmaxf(tile_max[0], fmaxf(s[0], s[1]));
            tile_max[1] = fmaxf(tile_max[1], fmaxf(s[2], s[3]));
        }

        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 1));
        tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], 2));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 1));
        tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], 2));

        tile_max[0] *= kScaleLog2e;
        tile_max[1] *= kScaleLog2e;

        for (uint32_t row = 0; row < 2; row++) {
            const float new_max = tile_max[row];

            if (kv_start_idx == 0) {
                m_ref[row] = new_max;
            } else if (new_max > m_ref[row] + kMaxExponentGap) {
                const float scale = exp2f(m_ref[row] - new_max);

#pragma unroll
                for (uint32_t n = 0; n < kHeadDimDivMmaN; n++) {
                    acc_o[n][row * 2] *= scale;
                    acc_o[n][row * 2 + 1] *= scale;
                }

                l[row] *= scale;
                m_ref[row] = new_max;
            }
        }

#pragma unroll
        for (uint32_t i = 0; i < kBcDivMmaN; i++) {
            float* s = acc_s[i];
            s[0] = exp2f(fmaf(s[0], kScaleLog2e, -m_ref[0]));
            s[1] = exp2f(fmaf(s[1], kScaleLog2e, -m_ref[0]));
            s[2] = exp2f(fmaf(s[2], kScaleLog2e, -m_ref[1]));
            s[3] = exp2f(fmaf(s[3], kScaleLog2e, -m_ref[1]));

            l[0] += s[0] + s[1];
            l[1] += s[2] + s[3];
        }

        cp_async_wait_group<0>();
        __syncthreads();

#pragma unroll
        for (uint32_t k = 0; k < kBcDivMmaK; k++) {
            const float* s0 = acc_s[k * 2]; // row00, row01, row10, row11
            const float* s1 = acc_s[k * 2 + 1];
            uint32_t p_reg[4] = {
                pack_float2(s0[0], s0[1]),
                pack_float2(s0[2], s0[3]),
                pack_float2(s1[0], s1[1]),
                pack_float2(s1[2], s1[3])
            };
            const uint32_t y = k * kMmaK + lane_id % kMmaK;
            const uint32_t x_offset = (lane_id / 16) * kMmaN;
#pragma unroll
            for (uint32_t n = 0; n < kHeadDimDivMmaN; n += 2) {
                uint32_t v_reg[4];
                const uint32_t x = n * kMmaN + x_offset;
                ldmatrix_x4_trans(v_reg, &shared.smem_v[y][swizzle_tma_128b(y, x)]);
                mma_m16n8k16(p_reg, v_reg, acc_o[n]);
                mma_m16n8k16(p_reg, v_reg + 2, acc_o[n + 1]);
            }
        }

        __syncthreads();
    }

    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);

    // write back
    const float inv_l[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
    const uint32_t row0 = warp_id * kMmaM + lane_id / 4;
    const uint32_t row1 = row0 + 8;
    const uint32_t col_base = (lane_id % 4) * 2;

#pragma unroll
    for (uint32_t i = 0; i < kHeadDimDivMmaN; i++) {
        const uint32_t col = col_base + i * kMmaN;
        float* o = acc_o[i];
        o[0] *= inv_l[0];
        o[1] *= inv_l[0];
        o[2] *= inv_l[1];
        o[3] *= inv_l[1];
        as<__nv_bfloat162>(&shared.smem_o[row0][swizzle_tma_128b(row0, col)]) =
            __float22bfloat162_rn(make_float2(o[0], o[1]));
        as<__nv_bfloat162>(&shared.smem_o[row1][swizzle_tma_128b(row1, col)]) =
            __float22bfloat162_rn(make_float2(o[2], o[3]));
    }
    __syncthreads();

    constexpr uint32_t kHeadDimU4 = kHeadDim / kNumBf16sPerVector;
    const uint32_t g_y_base = batch_idx * seq_len + q_start_idx;
    const uint32_t g_x_base = q_head_idx * kHeadDim;

#pragma unroll
    for (uint32_t i = tid; i < kBr * kHeadDimU4; i += kNumThreads) {
        const uint32_t s_y = i / kHeadDimU4;
        const uint32_t s_x = (i % kHeadDimU4) * kNumBf16sPerVector;
        const uint32_t g_y = g_y_base + s_y;
        const uint32_t g_x = g_x_base + s_x;
        if (q_start_idx + s_y < seq_len) {
            as<uint4>(O + static_cast<uint64_t>(g_y) * q_stride + g_x) =
                as<uint4>(&shared.smem_o[s_y][swizzle_tma_128b(s_y, s_x)]);
        }
    }
}

}

// MHA/GQA, contiguous Q/K/V shape: [batch_size, seq_len, head_num, head_dim]
// Tail rows in Q/K/V tiles are zero-filled via cp.async src_size.
template <uint32_t kHeadDim>
requires (kHeadDim == 128)
void launch_fa2_cp_async_lazy_rescale(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O)
{
    constexpr uint32_t kNumQWarps = 4;
    constexpr uint32_t kNumKWarps = 1;
    constexpr uint32_t kNumQMmaTilesPerWarp = 1;
    constexpr uint32_t kNumKMmaTilesPerWarp = 8;
    constexpr uint32_t kBr = kNumQWarps * kNumQMmaTilesPerWarp * kMmaM;
    constexpr uint32_t kBc = kNumKWarps * kNumKMmaTilesPerWarp * kMmaN;
    constexpr uint32_t kNumThreads = kNumQWarps * kNumKWarps * kNumThreadsPerWarp;

    const uint32_t batch_size = Q.size(0);
    const uint32_t head_dim = Q.size(3);

    const uint32_t q_seq_len = Q.size(1);
    const uint32_t q_head_num = Q.size(2);

    const uint32_t kv_seq_len = K.size(1);
    const uint32_t kv_head_num = K.size(2);

    TORCH_CHECK(kHeadDim == head_dim, "Unmatched head dim.");
    TORCH_CHECK(q_seq_len == kv_seq_len, "Only support equavalent Q/KV seq_len");

    const dim3 grid_dim(cdiv(q_seq_len, kBr), q_head_num, batch_size);
    const dim3 block_dim(kNumThreads);

    constexpr uint32_t kSmemBytes = sizeof(SharedStorage<__nv_bfloat16, kBr, kBc, 128>);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto kernel = fa2_cp_async_lazy_rescale<
        __nv_bfloat16,
        kNumQWarps,
        kNumKWarps,
        kNumQMmaTilesPerWarp,
        kNumKMmaTilesPerWarp,
        kHeadDim
    >;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    kernel<<<grid_dim, block_dim, kSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        q_seq_len,
        q_head_num,
        kv_head_num
    );
}

template void launch_fa2_cp_async_lazy_rescale<128>(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O
);
