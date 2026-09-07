# CUDA Kernels

A collection of CUDA operator implementations developed and benchmarked against production baselines. 

## Operators

| Operator | Workload | Baseline | Best measured result |
| :--- | :--- | :--- | :--- |
| [FP32 Matmul](fp32_matmul/README.md) | FP32, M=N=K=4096 | cuBLAS | 39.06 TFLOPS on RTX 5090 |
| [BF16 Matmul](bf16_matmul/README.md) | BF16, M=N=K=4096 | cuBLAS | 203.22 TFLOPS, 92.51% of cuBLAS on RTX 5090 |
| [Softmax](softmax/README.md) | FP32, 4096x4096 | `torch.softmax` | 0.087552 ms, 1.047x speedup on RTX 5090 |
| [FlashAttention2](flash_attn/README.md) | [B, S, H, D]=[1, S, 32, 128]<br>S=512/1024/2048/4096/8192 | Dao-AILab/flash-attention | Up to 1.269x speedup on RTX 5090 |
