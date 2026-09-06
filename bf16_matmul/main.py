import torch
import torch.utils.cpp_extension
from triton.testing import do_bench
import statistics

def benchmark(f, *args, **kwargs):
    samples = [do_bench(lambda: f(*args, **kwargs), return_mode="median")
               for _ in range(7)]
    return statistics.median(samples)

# def benchmark(f, *args, **kwargs):
    # return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    'module',
    sources=['matmul.cpp', 'matmul_v1.cu', 'matmul_v2.cu', 'matmul_v3.cu', 'matmul_v4.cu', 'matmul_v5.cu', 'matmul_v6.cu', 'matmul_v7.cu'],
    extra_cuda_cflags=[
        '-O3',
        '-lineinfo',
        '-Xptxas=-v'
    ],
    verbose=True
)

input1 = torch.randn((4096, 4096), dtype=torch.bfloat16).cuda()
input2 = torch.randn((4096, 4096), dtype=torch.bfloat16).cuda()
input2_trans = input2.transpose(0, 1).contiguous()

output_ref = torch.matmul(input1, input2_trans)
output_v1  = module.matmul_v1(input1, input2)
output_v2  = module.matmul_v2(input1, input2)
output_v3  = module.matmul_v3(input1, input2)
output_v4  = module.matmul_v4(input1, input2)
output_v5  = module.matmul_v5(input1, input2)
output_v6  = module.matmul_v6(input1, input2)
output_v7  = module.matmul_v7(input1, input2)

torch.testing.assert_close(output_v1, output_ref)
torch.testing.assert_close(output_v2, output_ref)
torch.testing.assert_close(output_v3, output_ref)
torch.testing.assert_close(output_v4, output_ref)
torch.testing.assert_close(output_v5, output_ref)
torch.testing.assert_close(output_v6, output_ref)
torch.testing.assert_close(output_v7, output_ref)

print(f'torch.matmul: {benchmark(torch.matmul, input1, input2_trans)}')
print(f'v1: {benchmark(module.matmul_v1, input1, input2)}')
print(f'v2: {benchmark(module.matmul_v2, input1, input2)}')
print(f'v3: {benchmark(module.matmul_v3, input1, input2)}')
print(f'v4: {benchmark(module.matmul_v4, input1, input2)}')
print(f'v5: {benchmark(module.matmul_v5, input1, input2)}')
print(f'v6: {benchmark(module.matmul_v6, input1, input2)}')
print(f'v7: {benchmark(module.matmul_v7, input1, input2)}')