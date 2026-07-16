# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

import os
import runpy
import subprocess

import pytest
from rocm.version import ROCM_VERSION_TUPLE, ROCM_VERSION
from rocm.bindings import hip as hiprt

device_printf_works = ROCM_VERSION_TUPLE[0:2] != (5, 5)

_, props = hiprt.hipGetDeviceProperties(0)
gpugen = props.gcnArchName.decode("utf-8").split(":")[0]
have_compatible_gpu_target = gpugen == "gfx90a"
have_rccl_support = gpugen not in ("gfx1151",)

# Compiler-specific conditions (from compiler/test_examples.py)
have_matching_hip_python = False
hiprtc_cannot_produce_llvm_bitcode = False

try:
    import hip
    have_matching_hip_python = hip.ROCM_VERSION == ROCM_VERSION
except Exception:
    pass

# Check for ROCm 6.1.0 bitcode bug
hiprtc_cannot_produce_llvm_bitcode = ROCM_VERSION_TUPLE == (6, 1, 0)

try:
    from cuda.bindings import runtime

    del runtime
    have_hip_python_interop = True
except ImportError:
    have_hip_python_interop = False

python_examples = [
    "0_Basic_Usage/hip_deviceattributes.py",
    "0_Basic_Usage/hip_deviceproperties.py",
    "0_Basic_Usage/hip_python_device_array.py",
    "0_Basic_Usage/hip_stream.py",
    "0_Basic_Usage/hipblas_with_numpy.py",
    "0_Basic_Usage/hipblas_with_numpy_and_cu_mask.py",
    "0_Basic_Usage/hipfft.py",
    "0_Basic_Usage/hiprand_monte_carlo_pi.py",
]

if have_rccl_support:
    python_examples += [
        "0_Basic_Usage/rccl_comminitall_bcast.py"
    ]

def _have_runtime_library(module_name: str, probe_symbol: str) -> bool:
    """True only if the binding module imports AND its backing runtime-linked
    DLL exports ``probe_symbol``.
    """
    import importlib

    try:
        module = importlib.import_module(f"rocm.bindings.{module_name}")
        return module.has_symbol(probe_symbol)
    except (ImportError, AttributeError):
        return False


have_amdsmi = _have_runtime_library("amdsmi", "amdsmi_init")
have_roctx = _have_runtime_library("roctx", "roctxMarkA")
have_hipfile = _have_runtime_library("hipfile", "hipFileGetVersion")
have_hipblaslt = _have_runtime_library("hipblaslt", "hipblasLtCreate")
have_hipsparselt = _have_runtime_library("hipsparselt", "hipsparseLtInit")

_hipblaslt_skipif = pytest.mark.skipif(
    not have_hipblaslt,
    reason="requires the hipblaslt bindings with a loadable libhipblaslt.so",
)
_hipsparselt_skipif = pytest.mark.skipif(
    not have_hipsparselt,
    reason="requires the hipsparselt bindings with a loadable libhipsparselt.so",
)

if have_amdsmi:
    python_examples += [
        "0_Basic_Usage/amdsmi_enumerate_sockets.py",
    ]

# The hipfile_copy examples create their own scratch fixture at runtime, so they
# only need the hipFILE bindings with a loadable libhipfile.so (plus an
# O_DIRECT-capable temp dir, overridable via HIPFILE_TMPDIR).
_hipfile_skipif = pytest.mark.skipif(
    not have_hipfile,
    reason=(
        "requires the hipFILE bindings with a loadable libhipfile.so "
        "(the example creates its own scratch files under an "
        "O_DIRECT-capable HIPFILE_TMPDIR)"
    ),
)
python_examples += [
    pytest.param("0_Basic_Usage/hipfile_copy.py", marks=_hipfile_skipif),
    pytest.param("0_Basic_Usage/hipfile_copy_lowlevel.py", marks=_hipfile_skipif),
]

# hipBLASLt / hipSPARSELt GEMM examples. Both were recently re-enabled in the
# libraries wheel and are runtime-linked, so they are guarded on their backing
# shared library being loadable (see _have_runtime_library).
python_examples += [
    pytest.param("0_Basic_Usage/hipblaslt_gemm.py", marks=_hipblaslt_skipif),
    pytest.param("0_Basic_Usage/hipsparselt_spmm.py", marks=_hipsparselt_skipif),
]

if device_printf_works:
    python_examples += [
        "0_Basic_Usage/hiprtc_launch_kernel_args.py",
        "0_Basic_Usage/hiprtc_launch_kernel_no_args.py",
    ]

if have_hip_python_interop:
    python_examples += [
        "1_CUDA_Interop/cuda_stream.py",
        "1_CUDA_Interop/cuda_stream_with_cuda_bindings.py",
        "1_CUDA_Interop/cuda_error_hallucinate_enums.py",
    ]

# The pynvml shim ships with hip-python-interop but is backed by AMD SMI.
if have_hip_python_interop and have_amdsmi:
    python_examples += [
        "1_CUDA_Interop/pynvml_query_devices.py",
    ]

# The nvtx shim ships with hip-python-interop but is backed by ROCTX.
if have_hip_python_interop and have_roctx:
    python_examples += [
        "1_CUDA_Interop/nvtx_annotate_ranges.py",
    ]

# The cuFile shim ships with hip-python-interop but is backed by hipFILE. Like
# the hipfile_copy examples it creates its own O_DIRECT scratch fixture, so it
# only needs the hipFILE bindings with a loadable libhipfile.so (guarded by the
# same _hipfile_skipif marker as the low-level hipfile examples).
if have_hip_python_interop and have_hipfile:
    python_examples += [
        pytest.param(
            "1_CUDA_Interop/cufile_copy_with_cuda_bindings.py",
            marks=_hipfile_skipif,
        ),
    ]

python_examples += [
    "2_Advanced/hiprtc_linking_device_functions.py",
]

if have_compatible_gpu_target:
    python_examples += [
        "2_Advanced/hiprtc_jit_with_llvm_ir.py",
        "2_Advanced/hiprtc_linking_llvm_ir.py",
    ]

# Complex examples
python_examples += [
    "3_Complex/hip_jacobi.py",  # MOVED from 2_Advanced/
]

# Compiler examples (moved to 2_Advanced/)
python_examples += [
    "2_Advanced/list_targets.py",
    "2_Advanced/parse_llvm_bitcode.py",
    # "2_Advanced/execution_engine_sum.py",  # TODO: only direct running works
    "2_Advanced/amd_comgr_parse_amd_hsa_kernel_descriptor.py",
    "2_Advanced/amd_comgr_disassemble_amdgpu_program.py",
    "2_Advanced/amd_comgr_disassemble_amdgpu_code_obj.py",
    "2_Advanced/amd_comgr_hip_to_llvm_ir.py",
    "2_Advanced/hiprtc_amd_comgr_hip_to_hsa.py",
    "2_Advanced/amd_comgr_llvm_ir_to_hsa.py",
    "2_Advanced/hiprtc_amd_comgr_hsa_to_code_obj.py",
    pytest.param(
        "2_Advanced/hiprtc_amd_comgr_get_jit_kernel_metadata.py",
        marks=[
            pytest.mark.skipif(
                not have_matching_hip_python,
                reason="requires that 'hip-python' is installed",
            ),
        ],
    ),
    pytest.param(
        "2_Advanced/hiprtc_hip_to_llvm_ir.py",
        marks=pytest.mark.skipif(
            not have_matching_hip_python,
            reason="requires that 'hip-python' is installed",
        ),
    ),
    pytest.param(
        "2_Advanced/hiprtc_linking_with_llvm_ir.py",
        marks=pytest.mark.skipif(
            (
                not have_matching_hip_python
                or not have_compatible_gpu_target
                or hiprtc_cannot_produce_llvm_bitcode
            ),
            reason=(
                "requires that compatible GPU target (==gfx90a) is "
                "present, 'hip-python' is installed, and that hipRTC "
                + " can produce bitcode (ROCm != 6.1.0)"
            ),
        ),
    ),
]


@pytest.mark.parametrize("example", python_examples)
def test_python_examples(example):
    abspath = os.path.join(os.path.dirname(__file__), example)
    runpy.run_path(abspath)


if have_hip_python_interop:

    @pytest.mark.parametrize(
        "module_name",
        [
            "cyruntime_cuda_stream",
            "cyruntime_cuda_stream_with_cuda_bindings",
        ],
    )
    def test_cython_examples(module_name):
        """Build and test Cython examples using setup.py (cross-platform)."""
        import sys
        import tempfile

        example_dir = os.path.join(os.path.dirname(__file__), "1_CUDA_Interop")

        # Build the Cython module using setup.py
        # Use a temporary directory for build artifacts to avoid polluting source
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["PYTHONPATH"] = example_dir + os.pathsep + env.get("PYTHONPATH", "")

            # Build extension in-place
            subprocess.check_call(
                [sys.executable, "setup.py", "build_ext", "--inplace",
                 f"--build-temp={tmpdir}", f"--build-lib={example_dir}"],
                cwd=example_dir,
                env=env,
            )

            # Import and run the module
            subprocess.check_call(
                [sys.executable, "-c", f"import {module_name}"],
                cwd=example_dir,
                env=env,
            )
