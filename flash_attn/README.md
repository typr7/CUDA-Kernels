# FlashAttention2

## Environment

- GPU: NVIDIA GeForce RTX 5090 (32 GB, SM120)
- NVIDIA driver: 595.58.03
- CUDA: 13.0
- PyTorch: 2.12.1+cu130
- flash-attn: 2.8.3.post1
- Input: BF16, causal attention, `(batch, seq_len, heads, head_dim) = (1, S, 32, 128)`

## Results

Each result is reported as **duration (ms) / speedup / TFLOPS**. Speedup is relative to [Dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention).

| seq_len | Dao-AILab/flash-attention | F.sdpa (PyTorch) | fa2_tma_lazy_rescale |
| ---: | ---: | ---: | ---: |
| 512 | **0.034848 / 1.000x / 61.62** | 0.036864 / 0.945x / 58.25 | 0.038912 / 0.896x / 55.19 |
| 1024 | 0.094208 / 1.000x / 91.18 | 0.094240 / 1.000x / 91.15 | **0.074240 / 1.269x / 115.70** |
| 2048 | 0.247840 / 1.000x / 138.64 | 0.250976 / 0.988x / 136.90 | **0.235728 / 1.051x / 145.76** |
| 4096 | 0.796592 / 1.000x / 172.53 | 0.796608 / 1.000x / 172.53 | **0.751680 / 1.060x / 182.84** |
| 8192 | 2.810848 / 1.000x / 195.58 | 2.847744 / 0.987x / 193.05 | **2.646528 / 1.062x / 207.73** |
