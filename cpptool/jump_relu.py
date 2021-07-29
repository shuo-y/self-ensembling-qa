# based on https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/_cpools/__init__.py
# and https://pytorch.org/tutorials/advanced/cpp_extension.html

import torch
import jump_relu


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        outputs = jump_relu.forward(input)
        return outputs

    @staticmethod
    def backward(ctx, output_grad):
        input = ctx.saved_variables[0]
        output = jump_relu.backward(input, output_grad)[0]
        return output

class JumpReLU(torch.nn.Module):
    def forward(self, input):
        output = JumpReLUFunction.apply(input)
        return output


if __name__ == "__main__":
    func = JumpReLU()
    x = torch.randn(3, 5)
    print(x)
    y = func(x) 
    print(y)
