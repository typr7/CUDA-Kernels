#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "scan.cuh"

namespace {

constexpr std::array kSizes = {1024, 4096, 16384, 65536, 262144, 1048576, 2097152};
constexpr int kWarmupCalls = 20;
constexpr int kCallsPerRound = 64;
constexpr int kRounds = 7;

void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error for %s at %s:%d: %s\n",
                 expr, file, line, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

__global__ void scrub_cache(unsigned* buffer, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count;
       i += stride) {
    buffer[i] = buffer[i] * 1664525u + 1013904223u + static_cast<unsigned>(i);
  }
}

struct EventPair {
  cudaEvent_t start;
  cudaEvent_t stop;
};

template <typename Scan, typename Evict>
double time_round_us(Scan&& scan, Evict&& evict,
                     const std::array<EventPair, kCallsPerRound>& events) {
  for (const auto& [start, stop] : events) {
    evict();
    CUDA_CHECK(cudaEventRecord(start));
    scan();
    CUDA_CHECK(cudaEventRecord(stop));
  }
  CUDA_CHECK(cudaEventSynchronize(events.back().stop));
  CUDA_CHECK(cudaGetLastError());

  double total_ms = 0;
  for (const auto& [start, stop] : events) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }
  return total_ms * 1000 / events.size();
}

}  // namespace

int main() {
  int device = 0;
  int runtime = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  std::fprintf(stderr, "%s, %d SMs, compute capability %d.%d, L2 %.0f MiB\n",
               prop.name, prop.multiProcessorCount, prop.major, prop.minor,
               prop.l2CacheSize / (1024.0 * 1024.0));
  std::fprintf(stderr, "CUDA runtime %d.%d, CUB %d.%d.%d\n", runtime / 1000, (runtime / 10) % 100,
               CUB_MAJOR_VERSION, CUB_MINOR_VERSION, CUB_SUBMINOR_VERSION);

  constexpr int kMaxBenchmarkElems = kSizes.back();
  int* in = nullptr;
  int* out = nullptr;
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&in, kMaxBenchmarkElems * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out, kMaxBenchmarkElems * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&workspace, get_workspace_size(kMaxBenchmarkElems)));

  size_t cub_inc_bytes = 0;
  size_t cub_exc_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, cub_inc_bytes, in, out, kMaxBenchmarkElems));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, cub_exc_bytes, in, out, kMaxBenchmarkElems));
  const size_t cub_capacity = std::max(cub_inc_bytes, cub_exc_bytes);
  void* cub_workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&cub_workspace, cub_capacity));

  std::vector<int> input(kMaxBenchmarkElems);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> values(-100, 100);
  std::generate(input.begin(), input.end(), [&] { return values(rng); });
  CUDA_CHECK(cudaMemcpy(in, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));

  const size_t cache_bytes = std::max<size_t>(64ull * 1024 * 1024,
                                             4ull * prop.l2CacheSize);
  const size_t cache_words = cache_bytes / sizeof(unsigned);
  unsigned* cache_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&cache_buffer, cache_words * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(cache_buffer, 0, cache_words * sizeof(unsigned)));
  auto evict = [&] {
    scrub_cache<<<16 * prop.multiProcessorCount, 256>>>(cache_buffer, cache_words);
  };

  std::array<EventPair, kCallsPerRound> events;
  for (auto& [start, stop] : events) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }

  std::puts("n,implementation,mode,median_us");
  for (int n : kSizes) {
    auto scan = [&](int job) {
      size_t cub_bytes = cub_capacity;
      switch (job) {
        case 0:
          inclusive_scan(in, out, workspace, n);
          break;
        case 1:
          CUDA_CHECK(cub::DeviceScan::InclusiveSum(cub_workspace, cub_bytes, in, out, n));
          break;
        case 2:
          exclusive_scan(in, out, workspace, n);
          break;
        case 3:
          CUDA_CHECK(cub::DeviceScan::ExclusiveSum(cub_workspace, cub_bytes, in, out, n));
          break;
      }
    };

    constexpr std::array implementations = {"scan", "cub", "scan", "cub"};
    constexpr std::array modes = {"inclusive", "inclusive", "exclusive", "exclusive"};
    constexpr int kJobs = implementations.size();
    for (int i = 0; i < kWarmupCalls; ++i) {
      for (int job = 0; job < kJobs; ++job) {
        evict();
        scan(job);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::array<std::array<double, kRounds>, kJobs> timings{};
    for (int round = 0; round < kRounds; ++round) {
      for (int i = 0; i < kJobs; ++i) {
        const int job = (round + i) % kJobs;
        timings[job][round] = time_round_us([&] { scan(job); }, evict, events);
      }
    }

    for (int job = 0; job < kJobs; ++job) {
      auto& rounds = timings[job];
      std::sort(rounds.begin(), rounds.end());
      std::printf("%d,%s,%s,%.6f\n", n, implementations[job], modes[job], rounds[kRounds / 2]);
    }
    std::fflush(stdout);
  }

  for (const auto& [start, stop] : events) {
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  CUDA_CHECK(cudaFree(cache_buffer));
  CUDA_CHECK(cudaFree(cub_workspace));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(in));
}
