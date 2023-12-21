import runpy
import os
import subprocess

import pytest
import shlex

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
  "0_Basic_Usage/hiprtc_launch_kernel_args.py",
  "0_Basic_Usage/hiprtc_launch_kernel_no_args.py",
  "0_Basic_Usage/rccl_comminitall_bcast.py",
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