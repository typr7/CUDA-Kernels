# FP32 Matmul

Fixed M = N = K = 4096. Compile with CUDA Toolkit 13.0 and report `duration; % of cuBLAS; TFLOPS (%SOL)`.

- TFLOPS metric: `2 * M * N * K / duration`
- `%SOL` metric: `TFLOPS / theoretical_peak_TFLOPS * 100`
- Inputs use A row-major x B row-major.

GPU                           | Driver    | PyTorch        | Peak basis
------------------------------|-----------|----------------|-----------
NVIDIA GeForce RTX 5090       | 595.58.03 | 2.12.1+cu130   | 170 SMs, CUDA runtime device clock 2407 MHz; FP32 peak 104.75 TFLOPS
NVIDIA A100-PCIE-40GB         | 595.71.05 | 2.12.1+cu130   | 108 SMs, CUDA runtime device clock 1410 MHz; FP32 peak 19.49 TFLOPS

## Results

Kernel name                                             | RTX 5090                         | A100-PCIE-40GB
--------------------------------------------------------|----------------------------------|-------------------------------
cuBLAS 13.1 (via PyTorch 2.12.1)                        | 2.05 ms; 100.00%; 67.08 (64.03%) | 7.72 ms; 100.00%; 17.79 (91.29%)
v1 (naive one-thread-per-output)                        | 18.28 ms; 11.21%; 7.52 (7.18%)   | 45.79 ms; 16.87%; 3.00 (15.40%)
v2 (shared-memory CTA tiling)                           | 14.66 ms; 13.98%; 9.38 (8.95%)   | 26.73 ms; 28.89%; 5.14 (26.37%)
v3 (thread coarsening)                                  | 6.52 ms; 31.45%; 21.09 (20.14%)  | 27.35 ms; 28.24%; 5.02 (25.78%)
v4 (thread tiling with register blocking)               | 3.55 ms; 57.71%; 38.71 (36.95%)  | 10.96 ms; 70.48%; 12.54 (64.34%)
v5 (warp tiling, transposed/skewed SMEM, vectorized B)  | 3.52 ms; 58.24%; 39.06 (37.29%)  | 10.73 ms; 71.97%; 12.81 (65.70%)
