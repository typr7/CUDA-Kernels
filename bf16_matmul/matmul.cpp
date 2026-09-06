#include <torch/extension.h>
#include <cuda_bf16.h>


using MatmulFn = void(
    const nv_bfloat16* A,
    const nv_bfloat16* B,
    nv_bfloat16* C,
    int M, int N, int K
);

MatmulFn matmul_v1;
MatmulFn matmul_v2;
MatmulFn matmul_v3;
MatmulFn matmul_v4;
MatmulFn matmul_v5;
MatmulFn matmul_v6;
MatmulFn matmul_v7;

template <MatmulFn matmul_fn>
torch::Tensor matmul(torch::Tensor A, torch::Tensor B)
{
    using torch::BFloat16;
    const int M = A.size(0);
    const int N = B.size(1);
    const int K = A.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto A_p = A.data_ptr<BFloat16>();
    auto B_p = B.data_ptr<BFloat16>();
    auto C_p = C.data_ptr<BFloat16>();
    matmul_fn(
        reinterpret_cast<nv_bfloat16*>(A_p),
        reinterpret_cast<nv_bfloat16*>(B_p),
        reinterpret_cast<nv_bfloat16*>(C_p),
        M, N, K
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_v1", &matmul<matmul_v1>, "Matrix multiplication v1");
    m.def("matmul_v2", &matmul<matmul_v2>, "Matrix multiplication v2");
    m.def("matmul_v3", &matmul<matmul_v3>, "Matrix multiplication v3");
    m.def("matmul_v4", &matmul<matmul_v4>, "Matrix multiplication v4");
    m.def("matmul_v5", &matmul<matmul_v5>, "Matrix multiplication v5");
    m.def("matmul_v6", &matmul<matmul_v6>, "Matrix multiplication v6");
    m.def("matmul_v7", &matmul<matmul_v7>, "Matrix multiplication v7");
}