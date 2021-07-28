from setuptools import setup, Extension
from torch.utils import cpp_extension

# JumpReLU based on https://arxiv.org/abs/1904.03750
# Code based on https://pytorch.org/tutorials/advanced/cpp_extension.html
setup(
    name="jump_relu",
    ext_modules=[
        cpp_extension.CppExtension("jump_relu", ["src/jump_relu.cpp"])
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension
    }
)