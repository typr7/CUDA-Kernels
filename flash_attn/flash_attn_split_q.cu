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
    alignas(128) union
    {
        alignas(128) struct
        {
            alignas(128) T smem_q[2][kBr][kHeadDim / 2];
            alignas(128) T smem_k[2][kBc][kHeadDim / 2];
            alignas(128) T smem_v[2][kBc][kHeadDim / 2];
        } mainloop;

        alignas(128) T smem_o[kBr][kHeadDim];
    } tile;

    alignas(16) uint64_t barrier_q;
    alignas(16) uint64_t barrier_k;
    alignas(16) uint64_t barrier_v;
};

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
    const T (&smem)[2][kNumRows][kNumCols]
)
{
    const uint32_t warp_id = threadIdx.x / kNumThreadsPerWarp;
    const uint32_t lane_id = threadIdx.x % kNumThreadsPerWarp;
    const uint32_t y = warp_id * kMmaM + lane_id % kMmaM;
    const uint32_t x_offset = lane_id / kMmaM * (kMmaK / 2);
#pragma unroll
    for (uint32_t j = 0; j < 2; j++) {
#pragma unroll
        for (uint32_t i = 0; i < kNumTiles / 2; i++) {
            const uint32_t x = i * kMmaK + x_offset;
            ldmatrix_x4(reg[j * (kNumTiles / 2) + i], &smem[j][y][swizzle_tma_128b(y, x)]);
        }
    }
}

template <uint32_t kNumTiles>
__device__ __forceinline__
void rescale(
    float (&acc_o)[kNumTiles][4],
    float l[2],
    float row_scale[2],
    uint32_t row_idx
)
{
    const float scale = row_scale[row_idx];
#pragma unroll
    for (uint32_t i = 0; i < kNumTiles; i++) {
        acc_o[i][row_idx * 2] *= scale;
        acc_o[i][row_idx * 2 + 1] *= scale;
    }
    l[row_idx] *= scale;
    row_scale[row_idx] = 1.f;
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
void flash_attn_split_q(
    const __grid_constant__ CUtensorMap Q,
    const __grid_constant__ CUtensorMap K,
    const __grid_constant__ CUtensorMap V,
    T* __restrict__ O, // [batch_size, q_seq_len, q_head_num, head_dim]
    uint32_t q_seq_len,
    uint32_t q_head_num,
    uint32_t kv_seq_len,
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
    constexpr float lazy_scale_threshold = 0x1p-8f;
    const float kScale = rsqrt(static_cast<float>(kHeadDim));
    const float kScaleLog2e = kScale * 1.44269504f;

    using Shared = SharedStorage<T, kBr, kBc, kHeadDim>;

    extern __shared__ Shared smem[];
    Shared& shared = smem[0];
    auto& mainloop = shared.tile.mainloop;

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

    if (tid == 0) {
        mbarrier_init(shared.barrier_q, 1);
        mbarrier_init(shared.barrier_k, 1);
        mbarrier_init(shared.barrier_v, 1);

        fence_proxy_async();
        
        mbarrier_expect_tx(shared.barrier_q, kBr * kHeadDim * sizeof(T));

        cp_async_bulk_tensor_4d(
            mainloop.smem_q[0],
            Q,
            shared.barrier_q,
            0,
            q_head_idx,
            q_start_idx,
            batch_idx
        );
        cp_async_bulk_tensor_4d(
            mainloop.smem_q[1],
            Q,
            shared.barrier_q,
            kHeadDim / 2,
            q_head_idx,
            q_start_idx,
            batch_idx
        );
    }
    __syncthreads();
    mbarrier_wait(shared.barrier_q, 0);

    // load Q tile from smem to reg
    uint32_t q_reg[kHeadDim / kMmaK][kNumQRegsPerThread];
    load_mma_tile_q_s2r(q_reg, mainloop.smem_q);

    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float l[2] = {0.f, 0.f};
    float row_scale[2] = {1.f, 1.f};

    uint32_t phase_k = 0;
    uint32_t phase_v = 0;

    float acc_o[kHeadDimDivMmaN][kNumAccRegsPerThread] = {0.f};
    const uint32_t kv_len = min(kv_seq_len, q_start_idx + kBr);
    for (uint32_t kv_start_idx = 0; kv_start_idx < kv_len; kv_start_idx += kBc) {
        if (tid == 0) {
            mbarrier_expect_tx(shared.barrier_k, kBc * kHeadDim * sizeof(T));
            mbarrier_expect_tx(shared.barrier_v, kBc * kHeadDim * sizeof(T));
            cp_async_bulk_tensor_4d(
                mainloop.smem_k[0],
                K,
                shared.barrier_k,
                0,
                kv_head_idx,
                kv_start_idx,
                batch_idx
            );
            cp_async_bulk_tensor_4d(
                mainloop.smem_k[1],
                K,
                shared.barrier_k,
                kHeadDim / 2,
                kv_head_idx,
                kv_start_idx,
                batch_idx
            );
            cp_async_bulk_tensor_4d(
                mainloop.smem_v[0],
                V,
                shared.barrier_v,
                0,
                kv_head_idx,
                kv_start_idx,
                batch_idx
            );
            cp_async_bulk_tensor_4d(
                mainloop.smem_v[1],
                V,
                shared.barrier_v,
                kHeadDim / 2,
                kv_head_idx,
                kv_start_idx,
                batch_idx
            );
        }
        __syncthreads();
        mbarrier_wait(shared.barrier_k, phase_k);
        phase_k ^= 1;

        // S = QK^T
        float acc_s[kBcDivMmaN][kNumAccRegsPerThread] = {0.f};
#pragma unroll
        for (uint32_t chunk = 0; chunk < 2; chunk++) {
#pragma unroll
            for (uint32_t k = 0; k < (kHeadDimDivMmaK / 2); k++) {
                const uint32_t x = k * kMmaK + ((lane_id / 8) & 0b1) * 8;
#pragma unroll
                for (uint32_t n = 0; n < kBcDivMmaN; n += 2) {
                    uint32_t k_reg[4];
                    const uint32_t y = n * kMmaN + (lane_id / 16) * 8 + lane_id % 8;
                    ldmatrix_x4(k_reg, &mainloop.smem_k[chunk][y][swizzle_tma_128b(y, x)]);
                    mma_m16n8k16(q_reg[chunk * (kHeadDimDivMmaK / 2) + k], k_reg, acc_s[n]);
                    mma_m16n8k16(q_reg[chunk * (kHeadDimDivMmaK / 2) + k], k_reg + 2, acc_s[n + 1]);
                }
            }
        }

        const uint32_t row0 = warp_start_idx + lane_id / 4;
        const uint32_t row1 = row0 + 8;
        const uint32_t col_base = kv_start_idx + (lane_id % 4) * 2;

        // online softmax
        float m_prev[2] = {m[0], m[1]};
        float m_curr[2] = {-CUDART_INF_F, -CUDART_INF_F}; // row0, row1
        float l_i[2] = {0.f, 0.f};

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
            m_curr[0] = fmaxf(m_curr[0], fmaxf(s[0], s[1]));
            m_curr[1] = fmaxf(m_curr[1], fmaxf(s[2], s[3]));
        }

        m_curr[0] = fmaxf(m_curr[0], __shfl_xor_sync(0xffffffff, m_curr[0], 1));
        m_curr[0] = fmaxf(m_curr[0], __shfl_xor_sync(0xffffffff, m_curr[0], 2));
        m_curr[1] = fmaxf(m_curr[1], __shfl_xor_sync(0xffffffff, m_curr[1], 1));
        m_curr[1] = fmaxf(m_curr[1], __shfl_xor_sync(0xffffffff, m_curr[1], 2));

        m_curr[0] = m_curr[0] == -CUDART_INF_F ? -CUDART_INF_F : m_curr[0] * kScaleLog2e;
        m_curr[1] = m_curr[1] == -CUDART_INF_F ? -CUDART_INF_F : m_curr[1] * kScaleLog2e;

        m[0] = fmaxf(m[0], m_curr[0]);
        m[1] = fmaxf(m[1], m_curr[1]);

        float alpha[2] = {
            m_prev[0] == -CUDART_INF_F ? 1.f : exp2f(m_prev[0] - m[0]),
            m_prev[1] == -CUDART_INF_F ? 1.f : exp2f(m_prev[1] - m[1])
        };

        row_scale[0] *= alpha[0];
        row_scale[1] *= alpha[1];

        if (row_scale[0] < lazy_scale_threshold) {
            rescale(acc_o, l, row_scale, 0);
        }
        if (row_scale[1] < lazy_scale_threshold) {
            rescale(acc_o, l, row_scale, 1);
        }

        float inv_row_scale[2] = {__frcp_rn(row_scale[0]), __frcp_rn(row_scale[1])};

#pragma unroll
        for (uint32_t i = 0; i < kBcDivMmaN; i++) {
            float* s = acc_s[i];
            s[0] = exp2f(fmaf(s[0], kScaleLog2e, -m[0])) * inv_row_scale[0];
            s[1] = exp2f(fmaf(s[1], kScaleLog2e, -m[0])) * inv_row_scale[0];
            s[2] = exp2f(fmaf(s[2], kScaleLog2e, -m[1])) * inv_row_scale[1];
            s[3] = exp2f(fmaf(s[3], kScaleLog2e, -m[1])) * inv_row_scale[1];

            l_i[0] += s[0] + s[1];
            l_i[1] += s[2] + s[3];
        }

        l_i[0] += __shfl_xor_sync(0xffffffff, l_i[0], 1);
        l_i[0] += __shfl_xor_sync(0xffffffff, l_i[0], 2);
        l_i[1] += __shfl_xor_sync(0xffffffff, l_i[1], 1);
        l_i[1] += __shfl_xor_sync(0xffffffff, l_i[1], 2);

        l[0] += l_i[0];
        l[1] += l_i[1];

        mbarrier_wait(shared.barrier_v, phase_v);
        phase_v ^= 1;
        
#pragma unroll
        for (uint32_t k = 0; k < kBcDivMmaK; k++) {
            const float* s0 = acc_s[k * 2]; // row00, row01, row10, row11
            const float* s1 = acc_s[k * 2 + 1];
            uint32_t p_reg[4] = {
                pack_float2({s0[0], s0[1]}),
                pack_float2({s0[2], s0[3]}),
                pack_float2({s1[0], s1[1]}),
                pack_float2({s1[2], s1[3]})
            };
            const uint32_t y = k * kMmaK + lane_id % kMmaK;
            const uint32_t x_offset = (lane_id / 16) * kMmaN;
#pragma unroll
            for (uint32_t chunk = 0; chunk < 2; chunk++) {
#pragma unroll
                for (uint32_t n = 0; n < kHeadDimDivMmaN / 2; n += 2) {
                    uint32_t v_reg[4];
                    const uint32_t x = n * kMmaN + x_offset;
                    ldmatrix_x4_trans(v_reg, &mainloop.smem_v[chunk][y][swizzle_tma_128b(y, x)]);
                    mma_m16n8k16(p_reg, v_reg, acc_o[chunk * (kHeadDimDivMmaN / 2) + n]);
                    mma_m16n8k16(p_reg, v_reg + 2, acc_o[chunk * (kHeadDimDivMmaN / 2) + n + 1]);
                }
            }
        }

        __syncthreads();
    }

    // write back
    const float inv_l[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
    const uint32_t row0 = warp_id * kMmaM + lane_id / 4;
    const uint32_t row1 = row0 + 8;
    const uint32_t col_base = (lane_id % 4) * 2;
    auto& smem_o = shared.tile.smem_o;

#pragma unroll
    for (uint32_t i = 0; i < kHeadDimDivMmaN; i++) {
        const uint32_t col = col_base + i * kMmaN;
        float* o = acc_o[i];
        o[0] *= inv_l[0];
        o[1] *= inv_l[0];
        o[2] *= inv_l[1];
        o[3] *= inv_l[1];
        as<__nv_bfloat162>(&smem_o[row0][swizzle_tma_128b(row0, col)]) =
            __float22bfloat162_rn(make_float2(o[0], o[1]));
        as<__nv_bfloat162>(&smem_o[row1][swizzle_tma_128b(row1, col)]) =
            __float22bfloat162_rn(make_float2(o[2], o[3]));
    }
    __syncthreads();

    constexpr uint32_t kHeadDimU4 = kHeadDim / kNumBf16sPerVector;
    const uint32_t g_y_base = batch_idx * q_seq_len + q_start_idx;
    const uint32_t g_x_base = q_head_idx * kHeadDim;
    const uint32_t stride = q_head_num * kHeadDim;

#pragma unroll
    for (uint32_t i = tid; i < kBr * kHeadDimU4; i += kNumThreads) {
        const uint32_t s_y = i / kHeadDimU4;
        const uint32_t s_x = (i % kHeadDimU4) * kNumBf16sPerVector;
        const uint32_t g_y = g_y_base + s_y;
        const uint32_t g_x = g_x_base + s_x;
        if (q_start_idx + s_y < q_seq_len) {
            as<uint4>(O + static_cast<uint64_t>(g_y) * stride + g_x) =
                as<uint4>(&smem_o[s_y][swizzle_tma_128b(s_y, s_x)]);
        }
    }
}

}

// MHA, Q/K/V shape: [batch_size, seq_len, head_num, head_dim]
template <uint32_t kHeadDim>
requires (kHeadDim == 128)
void launch_flash_attn_split_q(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O)
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

    TORCH_CHECK(kHeadDim == head_dim, "Unmatched head dim.");

    const uint32_t q_seq_len = Q.size(1);
    const uint32_t q_head_num = Q.size(2);

    const uint32_t kv_seq_len = K.size(1);
    const uint32_t kv_head_num = K.size(2);

    CUtensorMap tmq = create_tensor_map_4d(
        reinterpret_cast<__nv_bfloat16*>(Q.data_ptr()),
        head_dim,
        q_head_num,
        q_seq_len,
        batch_size,
        Q.stride(2),
        Q.stride(1),
        Q.stride(0),
        head_dim / 2,
        kBr
    );

    CUtensorMap tmk = create_tensor_map_4d(
        reinterpret_cast<__nv_bfloat16*>(K.data_ptr()),
        head_dim,
        kv_head_num,
        kv_seq_len,
        batch_size,
        K.stride(2),
        K.stride(1),
        K.stride(0),
        head_dim / 2,
        kBc
    );

    CUtensorMap tmv = create_tensor_map_4d(
        reinterpret_cast<__nv_bfloat16*>(V.data_ptr()),
        head_dim,
        kv_head_num,
        kv_seq_len,
        batch_size,
        V.stride(2),
        V.stride(1),
        V.stride(0),
        head_dim / 2,
        kBc
    );

    const dim3 grid_dim(cdiv(q_seq_len, kBr), q_head_num, batch_size);
    const dim3 block_dim(kNumThreads);

    constexpr uint32_t kSmemBytes = sizeof(SharedStorage<__nv_bfloat16, kBr, kBc, 128>); // QKV Tile + 3 barriers
                                  // + 3 * sizeof(__mbarrier_t); // barrier
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto kernel = flash_attn_split_q<
        __nv_bfloat16,
        kNumQWarps,
        kNumKWarps,
        kNumQMmaTilesPerWarp,
        kNumKMmaTilesPerWarp,
        kHeadDim
    >;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);

    kernel<<<grid_dim, block_dim, kSmemBytes, stream>>>(
        tmq,
        tmk,
        tmv,
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        q_seq_len,
        q_head_num,
        kv_seq_len,
        kv_head_num
    );
}

template void launch_flash_attn_split_q<128>(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O
);
