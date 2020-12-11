import os
import platform
import setuptools

import numpy as np
import torch

from setuptools import Extension
from Cython.Build import cythonize

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def parse_requirements():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, 'requirements.txt')) as f:
        lines = f.readlines()
    lines = [l for l in map(lambda l: l.strip(), lines) if l != '' and l[0] != '#']
    return lines


def make_cuda_ext(name, module, sources):

    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError('CUDA is required to compile PyTorch NMS!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


requirements = parse_requirements()


setuptools.setup(
    name='nms_pytorch',
    version='1.0',
    packages=setuptools.find_packages(),
    package_data={'nms_pytorch': ['*/*.so']},
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    ext_modules=[
        make_cython_ext(
            name='soft_nms_cpu',
            module='nms_pytorch',
            sources=['src/soft_nms_cpu.pyx']
        ),
        make_cuda_ext(
            name='nms_cpu',
            module='nms_pytorch',
            sources=['src/nms_cpu.cpp']
        ),
        make_cuda_ext(
            name='nms_cuda',
            module='nms_pytorch',
            sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu']
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)