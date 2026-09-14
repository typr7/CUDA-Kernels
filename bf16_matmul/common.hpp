#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>


inline constexpr uint32_t BF16_NUM_PER_U4 = sizeof(uint4) / sizeof(nv_bfloat16);
inline constexpr uint32_t WARP_SIZE = 32;
inline constexpr uint32_t MMA_M = 16;
inline constexpr uint32_t MMA_N = 8;
inline constexpr uint32_t MMA_K = 16;

template <typename ToType, typename FromType>
__device__ __forceinline__
ToType& as(FromType* p)
{
    return *reinterpret_cast<ToType*>(p);
}

__device__ __host__ __forceinline__
constexpr uint32_t cdiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__
uint32_t cvta_shared(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void ldmatrix_x2(uint32_t reg[2], uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t reg[4], uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, " // D
        "{%4, %5, %6, %7}, " // A
        "{%8, %9}, " // B
        "{%0, %1, %2, %3};" // D
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__
void cp_async_cg_16(const void* gmem, void* smem)
{
    uint32_t smem_addr = cvta_shared(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(smem_addr), "l"(gmem)
    );
}

__device__ __forceinline__
void cp_async_cg_16(const void* gmem, uint32_t smem_addr)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(smem_addr), "l"(gmem)
    );
}

__device__ __forceinline__
void cp_async_commit()
{
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

__device__ __forceinline__
void cp_async_wait_all()
{
    asm volatile("cp.async.wait_all;" ::: "memory");
}

__device__ __forceinline__
void fence_proxy_async()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__
void mbarrier_init(const uint64_t& barrier, uint32_t expected_count)
{
    const uint32_t addr = cvta_shared(&barrier);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(expected_count));
}

__device__ __forceinline__
void mbarrier_expect_tx(const uint64_t& barrier, uint32_t tx_bytes)
{
    const uint32_t addr = cvta_shared(&barrier);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(addr), "r"(tx_bytes));
}

__device__ __forceinline__
void mbarrier_wait(const uint64_t& barrier, uint32_t phase)
{
    const uint32_t addr = cvta_shared(&barrier);
    constexpr uint32_t TICKS = 100'000'000;
    asm volatile(
        "{\n\t"
            ".reg .pred p;\n\t"
            "WAIT:\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
            "@p bra DONE;\n\t"
            "bra WAIT;\n\t"
            "DONE:\n\t"
        "}\n"
        :: "r"(addr), "r"(phase), "r"(TICKS)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_bulk_tensor_2d(
    void* smem,
    const CUtensorMap& tensor_map,
    const uint64_t& barrier,
    uint32_t x,
    uint32_t y
) {
    const uint32_t smem_addr = cvta_shared(smem);
    const uint32_t barrier_addr = cvta_shared(&barrier);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr),
           "l"(&tensor_map),
           "r"(x),
           "r"(y),
           "r"(barrier_addr)
        : "memory"
    );
}

using KernelFn = void(
    const nv_bfloat16*,
    const nv_bfloat16*,
    nv_bfloat16*,
    int, int, int
);

template <KernelFn KERNEL, typename... Args>
void launch_kernel(dim3 grid, dim3 block, uint32_t smem_byte_size, Args... args)
{
    if (smem_byte_size > 48 * 1024) {
        cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_byte_size);
    }
    KERNEL<<<grid, block, smem_byte_size>>>(args...);
}