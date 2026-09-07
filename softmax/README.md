# Softmax

## Environment

- GPU: NVIDIA GeForce RTX 5090 (32 GB, SM120)
- NVIDIA driver: 595.58.03
- CUDA: 13.0
- PyTorch: 2.12.1+cu130
- Input: FP32, `shape = (4096, 4096)`, `dim = 1`

## Results

Speedup is relative to `torch.softmax`.

| Implementation | Duration (ms) | Speedup |
| :--- | ---: | ---: |
| torch.softmax | 0.091648 | 1.000x |
| softmax_v1 | 4.040704 | 0.023x |
| softmax_v2 | 0.095744 | 0.957x |
| softmax_v3a | 0.094192 | 0.973x |
| softmax_v3b | 0.094208 | 0.973x |
| softmax_v4 | 0.093184 | 0.984x |
| **softmax_v5** | **0.087552** | **1.047x** |
