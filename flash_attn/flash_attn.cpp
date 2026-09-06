#include <torch/extension.h>
#include <cstdint>


using FlashAttnFn = void(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O
);

template <uint32_t kHeadDim>
requires (kHeadDim == 128)
void launch_flash_attn_split_q(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);

template <FlashAttnFn flash_attn_fn>
torch::Tensor flash_attn(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    const int batch_size = Q.size(0);
    const int q_seq_len = Q.size(1);
    const int q_head_num = Q.size(2);
    const int head_dim = Q.size(3);
    auto O = torch::empty({batch_size, q_seq_len, q_head_num, head_dim}, Q.options());
    flash_attn_fn(Q, K, V, O);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_split_q", &flash_attn<launch_flash_attn_split_q<128>>, "Flash attention split Q");
}
