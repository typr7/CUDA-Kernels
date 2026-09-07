# BF16 Matmul

Fixed M = N = K = 4096. Compile with CUDA Toolkit 13.0 and report `duration; % of cuBLAS; TFLOPS (%SOL)`.

- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- Inputs use A row-major x B column-major.

GPU                           | Driver    | PyTorch        | Peak basis
------------------------------|-----------|----------------|-----------
NVIDIA GeForce RTX 5090       | 595.58.03 | 2.12.1+cu130   | 170 SMs, CUDA runtime device clock 2407 MHz; BF16 Tensor Core dense peak 419.01 TFLOPS
NVIDIA A100-PCIE-40GB         | 595.71.05 | 2.12.1+cu130   | 108 SMs, CUDA runtime device clock 1410 MHz; BF16 Tensor Core dense peak 311.87 TFLOPS

## Results

| Kernel name | RTX 5090 | A100-PCIE-40GB |
| :--- | :--- | :--- |
| cuBLAS 13.1 (via PyTorch 2.12.1) | 0.63 ms; 100.00%; 219.67 (52.43%) | 0.64 ms; 100.00%; 215.78 (69.19%) |
| v1 (tensor core MMA, hierarchical tiling) | 1.89 ms; 33.12%; 72.75 (17.36%) | 4.71 ms; 13.52%; 29.17 (9.35%) |
| v2 (vectorized memory copy) | 0.96 ms; 65.49%; 143.86 (34.33%) | 2.84 ms; 22.46%; 48.47 (15.54%) |
| v3 (swizzled shared memory) | 0.78 ms; 80.45%; 176.72 (42.18%) | 1.77 ms; 35.91%; 77.49 (24.85%) |
| v4 (flat shared-memory addressing) | 0.78 ms; 80.24%; 176.26 (42.07%) | 1.13 ms; 56.49%; 121.91 (39.09%) |
| v4 tuned | 0.72 ms; 86.54%; 190.10 (45.37%); CTA 64x128x64, warp 64x32 | 1.03 ms; 61.95%; 133.68 (42.87%); CTA 128x128x64, warp 64x32 |
| v5 (double-buffered async copy pipeline) | 0.78 ms; 79.81%; 175.32 (41.84%) | 0.83 ms; 76.41%; 164.89 (52.87%) |
| v5 tuned | 0.68 ms; 92.51%; 203.22 (48.50%); CTA 64x128x64, warp 32x64 | 0.83 ms; 76.60%; 165.29 (53.00%); CTA 128x128x64, warp 128x32 |
