import statistics
from collections.abc import Callable
from typing import Any, cast

import torch.nn.functional as F
import torch.utils.cpp_extension
from flash_attn import flash_attn_func
from triton.testing import do_bench


def benchmark(f: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    samples = [
        cast(float, do_bench(lambda: f(*args, **kwargs), return_mode="median"))
        for _ in range(7)
    ]
    return statistics.median(samples)

module: Any = torch.utils.cpp_extension.load(
    "module",
    sources=["flash_attn.cpp", "fa2_tma_lazy_rescale.cu"],
    extra_cflags=["-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo", "-Xptxas=-v"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

for seq_len in (512, 1024, 2048, 4096, 8192):
    shape = (1, seq_len, 32, 128)
    Q = torch.randn(shape, dtype=torch.bfloat16).cuda()
    K = torch.randn(shape, dtype=torch.bfloat16).cuda()
    V = torch.randn(shape, dtype=torch.bfloat16).cuda()
    Q_trans = Q.transpose(1, 2)
    K_trans = K.transpose(1, 2)
    V_trans = V.transpose(1, 2)

    output_ref = (
        F.scaled_dot_product_attention(
            Q_trans, K_trans, V_trans, is_causal=True
        )
        .transpose(1, 2)
        .contiguous()
    )
    output_fa2_tma = module.fa2_tma_lazy_rescale(Q, K, V)
    output_flash_attn = flash_attn_func(Q, K, V, causal=True)

    torch.testing.assert_close(output_fa2_tma, output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(output_flash_attn, output_ref, rtol=1e-2, atol=1e-2)

    print(f"shape: {shape}")
    print(
        f"F.sdpa: {benchmark(F.scaled_dot_product_attention, Q_trans, K_trans, V_trans, is_causal=True)}"
    )
    print(f"flash-attn: {benchmark(flash_attn_func, Q, K, V, causal=True)}")
    print(
        f"fa2_tma_lazy_rescale: {benchmark(module.fa2_tma_lazy_rescale, Q, K, V)}"
    )
