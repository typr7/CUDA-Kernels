#include <cassert>
#include <cstdint>

#include <cuda_bf16.h>

#include "common.hpp"

namespace
{

constexpr uint32_t BF16_PER_UINT4 = sizeof(uint4) / sizeof(nv_bfloat16);

template <uint32_t TILE_N_U4>
__device__ __forceinline__
uint32_t swizzle_smem_col_u4(uint32_t y, uint32_t x_u4)
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
void load_tile_to_smem(
    const nv_bfloat16* __restrict__ src,
    nv_bfloat16* __restrict__ smem,
    uint32_t src_stride
) {
    constexpr uint32_t TILE_N_U4 = TILE_N / BF16_PER_UINT4;
    const uint32_t src_stride_u4 = src_stride / BF16_PER_UINT4;

    const uint4* __restrict__ src_u4 = reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ smem_u4 = reinterpret_cast<uint4*>(smem);

    for (uint32_t i = threadIdx.x; i < TILE_M * TILE_N_U4; i += TB_SIZE) {
        const uint32_t y = i / TILE_N_U4;
        const uint32_t x_u4 = i % TILE_N_U4;
        const uint32_t smem_x_u4 = swizzle_smem_col_u4<TILE_N_U4>(y, x_u4);

        smem_u4[y * TILE_N_U4 + smem_x_u4] = src_u4[y * src_stride_u4 + x_u4];
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

    const uint32_t cta_tile_offset_m = blockIdx.y * CTA_TILE_M;
    const uint32_t cta_tile_offset_n = blockIdx.x * CTA_TILE_N;

    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;

    constexpr uint32_t WARP_TILES_N = CTA_TILE_N / WARP_TILE_N;
    const uint32_t warp_tile_y = warp_id / WARP_TILES_N;
    const uint32_t warp_tile_x = warp_id % WARP_TILES_N;

    const uint32_t warp_tile_offset_m = warp_tile_y * WARP_TILE_M;
    const uint32_t warp_tile_offset_n = warp_tile_x * WARP_TILE_N;

    constexpr uint32_t MMA_TILES_M = WARP_TILE_M / MMA_M;
    constexpr uint32_t MMA_TILES_N = WARP_TILE_N / MMA_N;

    constexpr uint32_t ACC_REGS_PER_THREAD = MMA_M * MMA_N / WARP_SIZE;
    constexpr uint32_t A_REGS_PER_THREAD = MMA_M * MMA_K * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);
    constexpr uint32_t B_REGS_PER_THREAD = MMA_K * MMA_N * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);

    // const uint32_t offset_m = cta_tile_offset_m + warp_tile_offset_m;
    // const uint32_t offset_n = cta_tile_offset_n + warp_tile_offset_n;

    A += cta_tile_offset_m * K;
    B += cta_tile_offset_n * K;

    extern __shared__ nv_bfloat16 smem[];
    // __shared__ nv_bfloat16 A_smem[CTA_TILE_M][CTA_TILE_K];
    // __shared__ nv_bfloat16 B_smem[CTA_TILE_N][CTA_TILE_K]; // column-major

    // auto* A_smem_p = reinterpret_cast<nv_bfloat16*>(A_smem);
    nv_bfloat16* A_smem_p = smem;
    // auto* B_smem_p = reinterpret_cast<nv_bfloat16*>(B_smem);
    nv_bfloat16* B_smem_p = smem + CTA_TILE_M * CTA_TILE_K;

    float acc_reg[MMA_TILES_M][MMA_TILES_N][ACC_REGS_PER_THREAD] = {0.f};
    for (uint32_t cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        load_tile_to_smem<TB_SIZE, CTA_TILE_M, CTA_TILE_K>(A, A_smem_p, K);
        load_tile_to_smem<TB_SIZE, CTA_TILE_N, CTA_TILE_K>(B, B_smem_p, K);
        __syncthreads();

        for (uint32_t k = 0; k < CTA_TILE_K; k += MMA_K) {
            uint32_t B_reg[MMA_TILES_N][B_REGS_PER_THREAD];

            for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                const uint32_t ldmatrix_lane = lane_id % 16;

                const uint32_t smem_n = warp_tile_offset_n + n * MMA_N + ldmatrix_lane % 8;
                const uint32_t smem_k = k + (ldmatrix_lane / 8) * 8;

                const uint32_t smem_k_u4 = smem_k / BF16_PER_UINT4;
                const uint32_t smem_k_u4_swizzled
                    = swizzle_smem_col_u4<CTA_TILE_K / BF16_PER_UINT4>(smem_n, smem_k_u4);
                const uint32_t smem_k_swizzled = smem_k_u4_swizzled * BF16_PER_UINT4;

                ldmatrix_x2(B_reg[n], cvta_shared(
                    B_smem_p + smem_n * CTA_TILE_K + smem_k_swizzled
                    // &B_smem[smem_n][smem_k_swizzled]
                ));
            }

            for (uint32_t m = 0; m < MMA_TILES_M; m++) {
                uint32_t A_reg[A_REGS_PER_THREAD];

                const uint32_t ldmatrix_group = lane_id / 8;
                const uint32_t smem_m = warp_tile_offset_m + m * MMA_M + (ldmatrix_group & 0b1) * 8 + lane_id % 8;
                const uint32_t smem_k = k + (ldmatrix_group / 2) * 8;

                const uint32_t smem_k_u4 = smem_k / BF16_PER_UINT4;
                const uint32_t smem_k_u4_swizzled
                    = swizzle_smem_col_u4<CTA_TILE_K / BF16_PER_UINT4>(smem_m, smem_k_u4);
                const uint32_t smem_k_swizzled = smem_k_u4_swizzled * BF16_PER_UINT4;

                ldmatrix_x4(A_reg, cvta_shared(
                    // Codegen-sensitive: the flat shared pointer form reduces
                    // short scoreboard stalls by about 33% vs. &A_smem[m][k].
                    A_smem_p + smem_m * CTA_TILE_K + smem_k_swizzled
                    // &A_smem[smem_m][smem_k_swizzled]
                ));

                for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                    mma_m16n8k16(A_reg, B_reg[n], acc_reg[m][n]);
                }
            }
        }
        __syncthreads();

        A += CTA_TILE_K;
        B += CTA_TILE_K;
    }

    /*
    for (uint32_t mma_tile_y = 0; mma_tile_y < MMA_TILES_M; mma_tile_y++) {
        for (uint32_t mma_tile_x = 0; mma_tile_x < MMA_TILES_N; mma_tile_x++) {
            const uint32_t y = mma_tile_y * MMA_M + lane_id / 4;
            const uint32_t x = mma_tile_x * MMA_N + (lane_id % 4) * 2;

            const float* reg = acc_reg[mma_tile_y][mma_tile_x];
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

// expect A in row-major, B in column-major, C in row-major
void matmul_v7(
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

    constexpr uint32_t TB_SIZE = (CTA_TILE_M / WARP_TILE_M) * (CTA_TILE_N / WARP_TILE_N) * WARP_SIZE;

    static_assert((CTA_TILE_M % WARP_TILE_M == 0) && (CTA_TILE_N % WARP_TILE_N == 0));
    assert((N % CTA_TILE_N == 0) && (M % CTA_TILE_M == 0) && (K % CTA_TILE_K == 0));

    constexpr uint32_t SMEM_BYTE_SIZE = (CTA_TILE_M + CTA_TILE_N) * CTA_TILE_K * sizeof(nv_bfloat16);
    constexpr auto kernel = matmul_kernel<TB_SIZE, CTA_TILE_M, CTA_TILE_N, CTA_TILE_K, WARP_TILE_M, WARP_TILE_N>;
    
    const dim3 grid_size(N / CTA_TILE_N, M / CTA_TILE_M);
    /*
    matmul_kernel<
        TB_SIZE,
        CTA_TILE_M, CTA_TILE_N, CTA_TILE_K,
        WARP_TILE_M, WARP_TILE_N
    ><<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
    */
    launch_kernel<kernel>(grid_size, TB_SIZE, SMEM_BYTE_SIZE, A, B, C, M, N, K);
}
