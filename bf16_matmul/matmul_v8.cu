#include <cassert>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>

#include "common.hpp"


namespace
{

__device__ __forceinline__
uint32_t swizzle_128b(uint32_t row, uint32_t col)
{
    return col ^ ((row & 0b111) << 3);
}

template <uint32_t CTA_TILE_M, uint32_t CTA_TILE_N, uint32_t CTA_TILE_K>
struct alignas(128) SharedStorage
{
    static constexpr uint32_t A_STAGE = CTA_TILE_M * CTA_TILE_K;
    static constexpr uint32_t B_STAGE = CTA_TILE_N * CTA_TILE_K;
    static constexpr uint32_t STAGE_SIZE = A_STAGE + B_STAGE;

    nv_bfloat16 tiles[2][STAGE_SIZE];
    alignas(16) uint64_t barriers[2];
};

template <
    uint32_t TB_SIZE,
    uint32_t CTA_TILE_M, uint32_t CTA_TILE_N, uint32_t CTA_TILE_K,
    uint32_t WARP_TILE_M, uint32_t WARP_TILE_N
> __launch_bounds__(TB_SIZE) __global__
void matmul_kernel(
    const __grid_constant__ CUtensorMap A,
    const __grid_constant__ CUtensorMap B,
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

    constexpr uint32_t MMA_TILES_M = WARP_TILE_M / MMA_M;
    constexpr uint32_t MMA_TILES_N = WARP_TILE_N / MMA_N;

    constexpr uint32_t ACC_REGS_PER_THREAD = MMA_M * MMA_N / WARP_SIZE;
    constexpr uint32_t A_REGS_PER_THREAD
        = MMA_M * MMA_K * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);
    constexpr uint32_t B_REGS_PER_THREAD
        = MMA_K * MMA_N * sizeof(nv_bfloat16) / WARP_SIZE / sizeof(uint32_t);

    using Shared = SharedStorage<CTA_TILE_M, CTA_TILE_N, CTA_TILE_K>;
    extern __shared__ Shared shared_storage[];
    Shared& shared = shared_storage[0];

    constexpr uint32_t A_STAGE = Shared::A_STAGE;
    constexpr uint32_t STAGE_BYTES = Shared::STAGE_SIZE * sizeof(nv_bfloat16);

    const uint32_t A_smem_offset_m = warp_tile_offset_m + ((lane_id / 8) & 0b1) * 8 + lane_id % 8;
    const uint32_t A_smem_offset_k = lane_id / 16 * 8;

    const uint32_t B_smem_offset_n = warp_tile_offset_n + (lane_id / 16) * 8 + lane_id % 8;
    const uint32_t B_smem_offset_k = ((lane_id / 8) & 0b1) * 8;

    float acc_reg[MMA_TILES_M][MMA_TILES_N][ACC_REGS_PER_THREAD] = {0.f};

    uint32_t phases[2] = {0, 0};
    if (tid == 0) {
        mbarrier_init(shared.barriers[0], 1);
        mbarrier_init(shared.barriers[1], 1);
        fence_proxy_async();

        mbarrier_expect_tx(shared.barriers[0], STAGE_BYTES);
        cp_async_bulk_tensor_2d(
            shared.tiles[0], A, shared.barriers[0], 0, cta_tile_offset_m
        );
        cp_async_bulk_tensor_2d(
            shared.tiles[0] + A_STAGE, B, shared.barriers[0], 0, cta_tile_offset_n
        );
    }
    __syncthreads();
    mbarrier_wait(shared.barriers[0], phases[0]);
    phases[0] ^= 1;

    for (uint32_t cta_tile_offset_k = 0; cta_tile_offset_k < K; cta_tile_offset_k += CTA_TILE_K) {
        const uint32_t curr = (cta_tile_offset_k / CTA_TILE_K) & 0b1;
        const uint32_t next = curr ^ 0b1;

        nv_bfloat16* A_smem_curr = shared.tiles[curr];
        nv_bfloat16* B_smem_curr = A_smem_curr + A_STAGE;

        const uint32_t next_k = cta_tile_offset_k + CTA_TILE_K;
        if (tid == 0 && next_k < K) {
            mbarrier_expect_tx(shared.barriers[next], STAGE_BYTES);
            cp_async_bulk_tensor_2d(
                shared.tiles[next], A, shared.barriers[next], next_k, cta_tile_offset_m
            );
            cp_async_bulk_tensor_2d(
                shared.tiles[next] + A_STAGE, B, shared.barriers[next], next_k, cta_tile_offset_n
            );
        }

        for (uint32_t k = 0; k < CTA_TILE_K; k += MMA_K) {
            uint32_t B_reg[MMA_TILES_N][B_REGS_PER_THREAD];

            for (uint32_t n = 0; n < MMA_TILES_N; n += 2) {
                const uint32_t smem_n = B_smem_offset_n + n * MMA_N;
                const uint32_t smem_k = B_smem_offset_k + k;
                const uint32_t smem_k_swz = swizzle_128b(smem_n, smem_k);

                ldmatrix_x4(B_reg[n], cvta_shared(
                    B_smem_curr + smem_n * CTA_TILE_K + smem_k_swz
                ));
            }

            for (uint32_t m = 0; m < MMA_TILES_M; m++) {
                uint32_t A_reg[A_REGS_PER_THREAD];

                const uint32_t smem_m = A_smem_offset_m + m * MMA_M;
                const uint32_t smem_k = A_smem_offset_k + k;
                const uint32_t smem_k_swz = swizzle_128b(smem_m, smem_k);
                ldmatrix_x4(A_reg, cvta_shared(
                    A_smem_curr + smem_m * CTA_TILE_K + smem_k_swz
                ));

                for (uint32_t n = 0; n < MMA_TILES_N; n++) {
                    mma_m16n8k16(A_reg, B_reg[n], acc_reg[m][n]);
                }
            }
        }

        __syncthreads();
        if (next_k < K) {
            mbarrier_wait(shared.barriers[next], phases[next]);
            phases[next] ^= 1;
        }
    }

    const uint32_t s_y_base = warp_tile_y * WARP_TILE_M;
    const uint32_t s_x_base = warp_tile_x * WARP_TILE_N;
    nv_bfloat16* smem_o = shared.tiles[0];
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
    __syncthreads();

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

CUtensorMap create_tensor_map_2d(
    const nv_bfloat16* device_ptr,
    uint64_t rows,
    uint64_t cols,
    uint32_t tile_rows,
    uint32_t tile_cols
) {
    CUtensorMap tensor_map;
    uint64_t global_dim[2] = {cols, rows};
    uint64_t global_stride[1] = {cols * sizeof(nv_bfloat16)};
    uint32_t box_dim[2] = {tile_cols, tile_rows};
    uint32_t element_stride[2] = {1, 1};

    const CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<nv_bfloat16*>(device_ptr),
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(result == CUDA_SUCCESS);
    return tensor_map;
}

}

// expect A in row-major, B in column-major, C in row-major; no bound check
void matmul_v8(
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

    using Shared = SharedStorage<CTA_TILE_M, CTA_TILE_N, CTA_TILE_K>;
    constexpr uint32_t SMEM_BYTE_SIZE = sizeof(Shared);

    static_assert(CTA_TILE_K * sizeof(nv_bfloat16) == 128);
    static_assert((CTA_TILE_M % WARP_TILE_M == 0) && (CTA_TILE_N % WARP_TILE_N == 0));
    assert((N % CTA_TILE_N == 0) && (M % CTA_TILE_M == 0) && (K % CTA_TILE_K == 0));

    const CUtensorMap tensor_map_a = create_tensor_map_2d(
        A, M, K, CTA_TILE_M, CTA_TILE_K
    );
    const CUtensorMap tensor_map_b = create_tensor_map_2d(
        B, N, K, CTA_TILE_N, CTA_TILE_K
    );

    constexpr auto kernel = matmul_kernel<
        TB_SIZE,
        CTA_TILE_M, CTA_TILE_N, CTA_TILE_K,
        WARP_TILE_M, WARP_TILE_N
    >;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTE_SIZE);

    const dim3 grid_size(N / CTA_TILE_N, M / CTA_TILE_M);
    kernel<<<grid_size, TB_SIZE, SMEM_BYTE_SIZE>>>(tensor_map_a, tensor_map_b, C, M, N, K);
}
