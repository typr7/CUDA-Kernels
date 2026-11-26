# Scan

## Environment

- NVIDIA A100-SXM4-40GB
- NVIDIA driver 620.70.06
- CUDA Toolkit 13.4 and CUB 3.4.3
- Out-of-place inclusive and exclusive scans of uniform random `int32` values in `[-100, 100]`

## Results

Durations are the median of three process runs. Each run reports the median of seven rounds, with 64 calls averaged per round. Speedup is CUB duration divided by scan duration for the same input size and mode.

| Elements | CUB inclusive (µs) | Scan inclusive (µs / speedup) | CUB exclusive (µs) | Scan exclusive (µs / speedup) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 11.06 | 14.43 / 0.766x | 11.12 | 14.53 / 0.765x |
| 4,096 | 12.08 | 15.10 / 0.800x | 12.13 | 15.18 / 0.799x |
| 16,384 | 11.97 | 15.20 / 0.787x | 11.94 | 15.25 / 0.783x |
| 65,536 | 12.93 | 15.90 / 0.813x | 12.91 | 15.92 / 0.811x |
| 262,144 | 14.16 | 17.34 / 0.816x | 14.21 | 17.39 / 0.817x |
| 1,048,576 | 19.76 | 22.02 / 0.898x | 19.81 | 22.03 / 0.899x |
| 2,097,152 | 27.87 | 30.14 / 0.925x | 27.92 | 30.18 / 0.925x |
