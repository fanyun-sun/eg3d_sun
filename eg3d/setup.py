# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

PACKAGE_NAME = 'eg3d'
DESCRIPTION = 'EG3D'
URL = 'https://gitlab-master.nvidia.com/jtremblay/eg3d_john'
AUTHOR = 'Eric Chan'
LICENSE = 'Custom'
version = '0.1.0'


def get_extensions():
    extensions = []
    return extensions


if __name__ == '__main__':
    packages = find_packages(".")
    print(f"Installing packages: {packages}")

    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        python_requires='~=3.8',
        # Package info
        packages=packages,
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)    
        }

    )
