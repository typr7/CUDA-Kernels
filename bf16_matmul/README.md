# BF16 Matmul

Fixed M = N = K = 4096. Compile with CUDA Toolkit 13.0 and report `duration; % of cuBLAS; TFLOPS (%SOL)`.

- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- Inputs use A row-major x B column-major.

GPU                           | Driver    | PyTorch        | Peak basis
------------------------------|-----------|----------------|-----------
NVIDIA GeForce RTX 5090       | 580.76.05 | 2.12.1+cu130   | 170 SMs, CUDA runtime device clock 2407 MHz; BF16 Tensor Core dense peak 419.01 TFLOPS
NVIDIA A100-PCIE-40GB         | 595.71.05 | 2.12.1+cu130   | 108 SMs, CUDA runtime device clock 1410 MHz; BF16 Tensor Core dense peak 311.87 TFLOPS

## Results

| Kernel name | RTX 5090 | A100-PCIE-40GB |
| :--- | :--- | :--- |
| cuBLAS 13.1 (via PyTorch 2.12.1) | 0.64 ms; 100.00%; 213.74 (51.01%) | 0.64 ms; 100.00%; 215.78 (69.19%) |
| v1 (tensor core MMA, hierarchical tiling) | 1.91 ms; 33.61%; 71.83 (17.14%) | 4.71 ms; 13.52%; 29.17 (9.35%) |
| v2 (vectorized memory copy) | 0.97 ms; 66.03%; 141.13 (33.68%) | 2.84 ms; 22.46%; 48.47 (15.54%) |
| v3 (swizzled shared memory) | 0.79 ms; 81.47%; 174.14 (41.56%) | 1.77 ms; 35.91%; 77.49 (24.85%) |
| v4 (flat shared-memory addressing) | 0.79 ms; 81.50%; 174.20 (41.57%) | 1.13 ms; 56.49%; 121.91 (39.09%) |
| v4 tuned | 0.74 ms; 87.47%; 186.96 (44.62%); CTA 64x128x64, warp 64x32 | 1.03 ms; 61.95%; 133.68 (42.87%); CTA 128x128x64, warp 64x32 |
| v5 (double-buffered async copy pipeline) | 0.80 ms; 80.40%; 171.85 (41.01%) | 0.83 ms; 76.41%; 164.89 (52.87%) |
| v5 tuned | 0.69 ms; 93.72%; 200.31 (47.81%); CTA 64x128x64, warp 32x64 | 0.83 ms; 76.60%; 165.29 (53.00%); CTA 128x128x64, warp 128x32 |
| v6 (v4 tuned + vectorized writeback) | 0.81 ms; 79.64%; 170.22 (40.62%) | — |
| v7 (v5 tuned + vectorized writeback) | 0.73 ms; 87.58%; 187.19 (44.67%) | — |
| v8 (TMA) | 0.74 ms; 86.74%; 185.39 (44.24%); CTA 128x64x64, warp 32x32 | — |
| v8 tuned | 0.64 ms; 99.90%; 213.52 (50.96%); CTA 128x64x64, warp 32x32 | — |
