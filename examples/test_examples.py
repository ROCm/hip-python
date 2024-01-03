# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import runpy
import os
import subprocess

import pytest
import shlex

from hip import ROCM_VERSION_TUPLE
device_printf_works = ROCM_VERSION_TUPLE[0:2] != (5,5)

try:
    from cuda import cuda
    del cuda
    have_hip_python_as_cuda = True
except ImportError:
    have_hip_python_as_cuda = False

python_examples = [
  "0_Basic_Usage/hip_deviceattributes.py",
  "0_Basic_Usage/hip_deviceproperties.py",
  "0_Basic_Usage/hip_python_device_array.py",
  "0_Basic_Usage/hip_stream.py",
  "0_Basic_Usage/hipblas_with_numpy.py",
  "0_Basic_Usage/hipfft.py",
  "0_Basic_Usage/hiprand_monte_carlo_pi.py",
  "0_Basic_Usage/rccl_comminitall_bcast.py",
]

if device_printf_works:
    python_examples += [
      "0_Basic_Usage/hiprtc_launch_kernel_args.py",
      "0_Basic_Usage/hiprtc_launch_kernel_no_args.py",
    ]

if have_hip_python_as_cuda:
    python_examples += [
      "1_CUDA_Interop/cuda_stream.py",
      "1_CUDA_Interop/cuda_error_hallucinate_enums.py",
    ]

python_examples += [
  "2_Advanced/hip_jacobi.py",
]

@pytest.mark.parametrize('example', python_examples)
def test_python_examples(example):
    abspath = os.path.join(os.path.dirname(__file__),example)
    runpy.run_path(abspath)

if have_hip_python_as_cuda:
    @pytest.mark.parametrize('example', ["1_CUDA_Interop/ccuda_stream.pyx"])
    def test_cython_examples(example):
        abspath = os.path.join(os.path.dirname(__file__),os.path.dirname(example))
        subprocess.check_call(shlex.split(f"make -C {abspath} run"))
