#include <cstdint>
#include <cassert>

#include <cuda_bf16.h>

#include "common.hpp"


namespace
{

template <uint32_t TILE_N_U4>
__device__ __forceinline__
uint32_t swizzle_u4(uint32_t y, uint32_t x_u4)
{
    return x_u4 ^ (y & (TILE_N_U4 - 1));
}

__device__ __forceinline__
uint32_t swizzle_128b(uint32_t row, uint32_t col)
{
    return col ^ ((row & 0b111) << 3);
}

template <uint32_t TB_SIZE, uint32_t TILE_M, uint32_t TILE_N>
__device__ __forceinline__
void load_tile_to_smem_async(
    const nv_bfloat16* __restrict__ src,
    nv_bfloat16* __restrict__ smem,
    uint32_t src_stride
) {
    constexpr uint32_t TILE_N_U4 = TILE_N / BF16_NUM_PER_U4;
    const uint32_t src_stride_u4 = src_stride / BF16_NUM_PER_U4;

    const auto* __restrict__ src_u4 = reinterpret_cast<const uint4*>(src);
    auto* __restrict__ smem_u4 = reinterpret_cast<uint4*>(smem);

    for (uint32_t i = threadIdx.x; i < TILE_M * TILE_N_U4; i += TB_SIZE) {
        const uint32_t y = i / TILE_N_U4;
        const uint32_t x_u4 = i % TILE_N_U4;
        const uint32_t x_u4_swizzled = swizzle_u4<TILE_N_U4>(y, x_u4);
        // smem_u4[y * TILE_N_U4 + x_u4_swizzled] = src_u4[y * src_stride_u4 + x_u4];
        cp_async_cg_16(
            src_u4 + y * src_stride_u4 + x_u4,
            smem_u4 + y * TILE_N_U4 + x_u4_swizzled
        );
    }
}

template <
    uint32_t TB_SIZE,
    uint32_t CTA_TILE_M, uint32_t CTA_TILE_N, uint32_t CTA_TILE_K,
    uint32_t WARP_TILE_M, uint32_t WARP_TILE_N
> __launch_bounds__(TB_SIZE) __global__
void matmul_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;

    const uint32_t cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const uint32_t cta_tile_offset_n = blockIdx.x * CTA_TILE_N;

    constexpr uint32_t WARP_TILES_N = CTA_TILE_N / WARP_TILE_N;

    const uint32_t warp_tile_y = warp_id / WARP_TILES_N;
    const uint32_t warp_tile_x = warp_id % WARP_TILES_N;

    const uint32_t warp_tile_offset_m = warp_tile_y * WARP_TILE_M;
    const uint32_t warp_tile_offset_n = warp_tile_x * WARP_TILE_N;

    // const uint32_t offset_m = cta_tile_offset_m + warp_tile_offset_m;
    // const uint32_t offset_n = cta_tile_offset_n + warp_tile_offset_n;

    constexpr uint32_t MMA_TILES_M = WARP_TILE_M / MMA_M;
    constexpr uint32_t MMA_TILES_N = WARP_TILE_N / MMA_N;

    constexpr uint32_t ACC_REGS_PER_THREAD = MMA_M * MMA_N / WARP_SIZE;
    constexpr uint32_t A_REGS_PER_THREAD
        = MMA_M * MMA_K * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);
    constexpr uint32_t B_REGS_PER_THREAD
        = MMA_K * MMA_N * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n * K;
    // C += offset_m * N + offset_n;

    extern __shared__ nv_bfloat16 smem[];

    // buffer_0        buffer_1
    // [A_smem][B_smem][A_smem][B_smem]
    constexpr uint32_t A_STAGE = CTA_TILE_M * CTA_TILE_K;
    constexpr uint32_t B_STAGE = CTA_TILE_N * CTA_TILE_K;
    constexpr uint32_t STAGE_SIZE = A_STAGE + B_STAGE;

    const uint32_t A_smem_offset_m = warp_tile_offset_m + ((lane_id / 8) & 0b1) * 8 + lane_id % 8;
    const uint32_t A_smem_offset_k = lane_id / 16 * 8;

    const uint32_t B_smem_offset_n = warp_tile_offset_n + (lane_id / 16) * 8 + lane_id % 8;
    const uint32_t B_smem_offset_k = ((lane_id / 8) & 0b1) * 8;

    float acc_reg[MMA_TILES_M][MMA_TILES_N][ACC_REGS_PER_THREAD] = {0.f};

    load_tile_to_smem_async<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A, smem, K);
    load_tile_to_smem_async<TB_SIZE, CTA_TILE_N, CTA_TILE_K>(B, smem + A_STAGE, K);
    cp_async_wait_all();
    __syncthreads();

    for (uint32_t cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        const uint32_t curr = (cta_tile_offset_k / CTA_TILE_K) & 0b1;
        const uint32_t next = curr ^ 0b1;

        nv_bfloat16* A_smem_curr = smem + curr * STAGE_SIZE;
        nv_bfloat16* B_smem_curr = A_smem_curr + A_STAGE;
        nv_bfloat16* A_smem_next = smem + next * STAGE_SIZE;
        nv_bfloat16* B_smem_next = A_smem_next + A_STAGE;

        const uint32_t next_k = cta_tile_offset_k + CTA_TILE_K;
        if (next_k < K) {
            load_tile_to_smem_async<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A + next_k, A_smem_next, K);
            load_tile_to_smem_async<TB_SIZE, CTA_TILE_N, CTA_TILE_K>(B + next_k, B_smem_next, K);
        }

        for (uint32_t k = 0; k < CTA_TILE_K; k += MMA_K) {
            uint32_t B_reg[MMA_TILES_N][B_REGS_PER_THREAD];

            // (16x8)
            for (uint32_t n = 0; n < MMA_TILES_N; n += 2) {
                /*
                const uint32_t ldmatrix_lane = lane_id % 16;
                const uint32_t smem_n = warp_tile_offset_n + n * MMA_N + ldmatrix_lane % 8;
                const uint32_t smem_k = k + (ldmatrix_lane / 8) * 8;
                const uint32_t smem_k_swizzled = swizzle_u4<CTA_TILE_K / BF16_NUM_PER_U4>(
                    smem_n, smem_k / BF16_NUM_PER_U4
                ) * BF16_NUM_PER_U4;
                ldmatrix_x2(B_reg[n], cvta_shared(
                    B_smem[curr] + smem_n * CTA_TILE_K + smem_k_swizzled
                ));
                */
                const uint32_t smem_n = B_smem_offset_n + n * MMA_N;
                const uint32_t smem_k = B_smem_offset_k + k;
                const uint32_t smem_k_swz = swizzle_u4<CTA_TILE_K / BF16_NUM_PER_U4>(
                    smem_n, smem_k / BF16_NUM_PER_U4
                ) * BF16_NUM_PER_U4;
                
                ldmatrix_x4(B_reg[n], cvta_shared(
                    B_smem_curr + smem_n * CTA_TILE_K + smem_k_swz
                ));
            }

            // (16x16)
            for (uint32_t m = 0; m < MMA_TILES_M; m++) {
                uint32_t A_reg[A_REGS_PER_THREAD];

                /*
                const uint32_t matrix_id = lane_id / 8;
                const uint32_t smem_m = warp_tile_offset_m + m * MMA_M + (matrix_id & 0b1) * 8 + lane_id % 8;
                const uint32_t smem_k = k + (lane_id / 16) * 8;
                const uint32_t smem_k_swizzled = swizzle_u4<CTA_TILE_K / BF16_NUM_PER_U4>(
                    smem_m, smem_k / BF16_NUM_PER_U4
                ) * BF16_NUM_PER_U4;
                ldmatrix_x4(A_reg, cvta_shared(
                    A_smem[curr] + smem_m * CTA_TILE_K + smem_k_swizzled
                ));
                */
                const uint32_t smem_m = A_smem_offset_m + m * MMA_M;
                const uint32_t smem_k = A_smem_offset_k + k;
                const uint32_t smem_k_swz = swizzle_u4<CTA_TILE_K / BF16_NUM_PER_U4>(
                    smem_m, smem_k / BF16_NUM_PER_U4
                ) * BF16_NUM_PER_U4;
                ldmatrix_x4(A_reg, cvta_shared(
                    A_smem_curr + smem_m * CTA_TILE_K + smem_k_swz
                ));

                for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                    mma_m16n8k16(A_reg, B_reg[n], acc_reg[m][n]);
                }
            }
        }

        if (next_k < K) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    /*
    for (uint32_t m = 0; m < MMA_TILES_M; m++) {
        for (uint32_t n = 0; n < MMA_TILES_N; n++) {
            const uint32_t y = m * MMA_M + lane_id / 4;
            const uint32_t x = n * MMA_N + (lane_id % 4) * 2;

            const float* reg = acc_reg[m][n];
            reinterpret_cast<nv_bfloat162*>(C + y * N + x)[0] =
                __float22bfloat162_rn(make_float2(reg[0], reg[1]));
            reinterpret_cast<nv_bfloat162*>(C + (y + 8) * N + x)[0] =
                __float22bfloat162_rn(make_float2(reg[2], reg[3]));
        }
    }
    */
    const uint32_t s_y_base = warp_tile_y * WARP_TILE_M;
    const uint32_t s_x_base = warp_tile_x * WARP_TILE_N;
    nv_bfloat16* smem_o = smem;
    for (uint32_t m = 0; m < MMA_TILES_M; m++) {
        const uint32_t s_y0 = s_y_base + m * MMA_M + lane_id / 4;
        const uint32_t s_y1 = s_y0 + 8;
        for (uint32_t n = 0; n < MMA_TILES_N; n++) {
            const uint32_t s_x = s_x_base + n * MMA_N + (lane_id % 4) * 2;
            const float* reg = acc_reg[m][n];
            as<nv_bfloat162>(smem_o + s_y0 * CTA_TILE_N + swizzle_128b(s_y0, s_x)) =
                __float22bfloat162_rn(make_float2(reg[0], reg[1]));
            as<nv_bfloat162>(smem_o + s_y1 * CTA_TILE_N + swizzle_128b(s_y1, s_x)) =
                __float22bfloat162_rn(make_float2(reg[2], reg[3]));
        }
    }

    constexpr uint32_t CTA_TILE_N_U8 = CTA_TILE_N / BF16_NUM_PER_U4;

    for (uint32_t i = tid; i < CTA_TILE_M * CTA_TILE_N_U8; i += TB_SIZE) {
        const uint32_t s_y = i / CTA_TILE_N_U8;
        const uint32_t s_x = (i % CTA_TILE_N_U8) * BF16_NUM_PER_U4;
        const uint32_t g_y = cta_tile_offset_m + s_y;
        const uint32_t g_x = cta_tile_offset_n + s_x;
        as<uint4>(C + g_y * N + g_x) =
            as<uint4>(smem_o + s_y * CTA_TILE_N + swizzle_128b(s_y, s_x));
    }
}

}

// no bound check
void matmul_v6(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M, int N, int K
) {
    constexpr uint32_t CTA_TILE_M = 128;
    constexpr uint32_t CTA_TILE_N = 128;
    constexpr uint32_t CTA_TILE_K = 64;

    constexpr uint32_t WARP_TILE_M = 64;
    constexpr uint32_t WARP_TILE_N = 64;

    constexpr uint32_t WARP_TILES_M = CTA_TILE_M / WARP_TILE_M;
    constexpr uint32_t WARP_TILES_N = CTA_TILE_N / WARP_TILE_N;

    constexpr uint32_t TB_SIZE = WARP_TILES_M * WARP_TILES_N * WARP_SIZE;

    // double buffer
    constexpr uint32_t SMEM_BYTE_SIZE = 2 * (CTA_TILE_M + CTA_TILE_N) * CTA_TILE_K * sizeof(nv_bfloat16);

    static_assert((CTA_TILE_M % WARP_TILE_M == 0) && (CTA_TILE_N % WARP_TILE_N == 0));
    assert((N % CTA_TILE_N == 0) && (M % CTA_TILE_M == 0) && (K % CTA_TILE_K == 0));

    constexpr auto kernel = matmul_kernel<TB_SIZE, CTA_TILE_M, CTA_TILE_N, CTA_TILE_K, WARP_TILE_M, WARP_TILE_N>;

    const dim3 grid_size(N / CTA_TILE_N, M / CTA_TILE_M);
    
    launch_kernel<kernel>(grid_size, TB_SIZE, SMEM_BYTE_SIZE, A, B, C, M, N, K);
}
