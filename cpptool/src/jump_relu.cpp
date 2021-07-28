
// JumpReLU based on https://arxiv.org/abs/1904.03750
// Code based on https://pytorch.org/tutorials/advanced/cpp_extension.html
// and  https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/_cpools/src/right_pool.cpp

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> jump_relu_forward(
    at::Tensor input
) {
    at::Tensor output = at::zeros_like(input);
    output.copy_(input);
    output.add_(1);
    
    auto pos_ind = (input > 0);
    // Maybe no need to conver to float?
    // auto pos_msk = pos_ind.to(torch::kFloat32);

    output = output * pos_ind;
 
    return {
        output
    };
}

std::vector<at::Tensor> jump_relu_backward(
    at::Tensor input,
    at::Tensor output_grad
) {
    at::Tensor output = at::zeros_like(input);
    output.copy_(output_grad);

    auto neg_ind = (input > 0);
    output = output * neg_ind;
    
    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &jump_relu_forward, "JumpReLU Forward");
    m.def("backward", &jump_relu_backward, "JumpReLU Backward");
}
