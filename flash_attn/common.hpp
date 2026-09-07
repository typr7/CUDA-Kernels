#pragma once

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <c10/core/Device.h>
#include <type_traits>
#include <cuda/ptx>

#include <cuda_bf16.h>


inline constexpr uint32_t kNumThreadsPerWarp = 32;

inline constexpr uint32_t kNumBf16sPerVector = sizeof(uint4) / sizeof(__nv_bfloat16);

// utility

template <typename ToType, typename FromType>
__device__ __forceinline__
ToType& as(FromType* p)
{
    return *reinterpret_cast<ToType*>(p);
}

// ret ((a + b - 1) // b)
__host__ __device__ __forceinline__
constexpr uint32_t cdiv(uint32_t a, uint32_t b)
{
    return ((a + b - 1) / b);
}

__device__ __forceinline__
uint32_t pack_float2(float2 f2)
{
    union {
        uint32_t packed;
        __nv_bfloat162 bf162;
    } pack;
    pack.bf162 = __float22bfloat162_rn(f2);
    return pack.packed;
}

__device__ __forceinline__
uint32_t cvta_shared(const void* smem)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem));
}

__device__ __forceinline__
void fence_proxy_async()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// cp.async

__device__ __forceinline__
void cp_async_cg_16(const void* gmem, void* smem, uint32_t src_size = 16)
{
    const uint32_t smem_addr = cvta_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem), "r"(src_size)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <uint32_t kPendingGroups>
__device__ __forceinline__
void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kPendingGroups) : "memory");
}

// mbarrier

__device__ __forceinline__
void mbarrier_init(const uint64_t& mbarrier, uint32_t expected_count)
{
    const uint32_t addr = cvta_shared(&mbarrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(expected_count));
}

__device__ __forceinline__
void mbarrier_expect_tx(const uint64_t& mbarrier, uint32_t tx_bytes)
{
    const uint32_t addr = cvta_shared(&mbarrier);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_arrive(const uint64_t& mbarrier)
{
    const uint32_t addr = cvta_shared(&mbarrier);
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(addr));
}

__device__ __forceinline__
void mbarrier_wait(const uint64_t& mbarrier, uint32_t phase)
{
    const uint32_t addr = cvta_shared(&mbarrier);
    const uint32_t ticks = 100'000'000;
    asm volatile(
        "{\n\t"
            ".reg .pred p;\n\t"
            "LAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
            "@p bra DONE;\n\t"
            "bra LAB_WAIT;\n\t"
            "DONE:\n\t"
        "}\n"
        :: "r"(addr), "r"(phase), "r"(ticks)
        : "memory"
    );
}

// ldmatrix
__device__ __forceinline__
void ldmatrix_x4(uint32_t reg[4], const void* smem)
{
    const uint32_t addr = cvta_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr)
    );
}

__device__ __forceinline__
void ldmatrix_x4_trans(uint32_t *reg, const void *smem)
{
    const uint32_t addr = cvta_shared(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr)
    );
}

// MMA
__device__ __forceinline__
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, " // D
        "{%4, %5, %6, %7}, " // A
        "{%8, %9}, " // B
        "{%0, %1, %2, %3};\n" // D
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
    );
}

// TMA

__device__ __forceinline__
uint32_t swizzle_tma_128b(uint32_t row, uint32_t col)
{
    return (col ^ ((row & 0b111) << 3));
}

__device__ __forceinline__
void cp_async_bulk_tensor_4d(
    const void* smem,
    const CUtensorMap& tmap,
    const uint64_t& mbarrier,
    uint32_t r0,
    uint32_t r1,
    uint32_t r2,
    uint32_t r3
)
{
    const uint32_t smem_addr = cvta_shared(smem);
    const uint32_t mbar_addr = cvta_shared(&mbarrier);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
        :: "r"(smem_addr),
           "l"(&tmap),
           "r"(r0),
           "r"(r1),
           "r"(r2),
           "r"(r3),
           "r"(mbar_addr)
        : "memory"
    );
}

// for shape: [B, S, H, D]
template <typename T, uint32_t kRowBytes = 128>
requires (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>)
inline CUtensorMap create_tensor_map_4d(
    T* device_ptr,
    uint64_t dim_d,
    uint64_t dim_h,
    uint64_t dim_s,
    uint64_t dim_b,
    uint64_t stride_h,
    uint64_t stride_s,
    uint64_t stride_b,
    uint32_t box_d,
    uint32_t box_s
)
{
    CUtensorMap tmap;
    CUtensorMapDataType dtype = std::is_same_v<T, __half>
                              ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                              : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = kRowBytes == 128
                               ? CU_TENSOR_MAP_SWIZZLE_128B
                               : CU_TENSOR_MAP_SWIZZLE_64B;
    
    uint64_t global_dim[4] = {dim_d, dim_h, dim_s, dim_b};
    uint64_t global_stride[3] = {stride_h * sizeof(T), stride_s * sizeof(T), stride_b * sizeof(T)};
    uint32_t box_dim[4] = {box_d, 1, box_s, 1};
    uint32_t element_stride[4] = {1, 1, 1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tmap,
        dtype,
        4,
        device_ptr,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(res == CUDA_SUCCESS, "Failed to create tensor map.");
    return tmap;
}