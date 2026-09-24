# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Annotated application code against the `hip` compatibility namespace.

Type-checked, never run. See `../test_pyright_samples.py`.
"""

from hip import HIP_VERSION_NAME, hip, hiprtc


def device_arch(index: int = 0) -> str:
    """Architecture name of one device, ``gfx90a`` and the like."""
    err, props = hip.hipGetDeviceProperties(index)
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipGetDeviceProperties({index}): {err}")
    return str(props.gcnArchName)


def empty_properties() -> int:
    """Read a field off a record the annotation names, not an `Any`.

    `hipGetDeviceProperties` above returns `Any`, which type-checks
    whatever is asked of it; a constructed record is the case where the
    stub has to carry the field.
    """
    props = hip.hipDeviceProp_t()
    return int(props.multiProcessorCount)


def compile_empty_program() -> str:
    """The spellings a caller reaches for around HIPRTC.

    `_hiprtcProgram` is the name the module binds, underscore and all,
    and `createRef` comes from the util base class every wrapper
    inherits -- neither is visible unless the stub says so.
    """
    _, program = hiprtc.hiprtcCreateProgram(b"", b"empty", 0, [], [])
    handle: hiprtc._hiprtcProgram = program
    _, log_size = hiprtc.hiprtcGetProgramLogSize(handle)
    log = bytearray(int(log_size))
    hiprtc.hiprtcGetProgramLog(handle, log)
    hiprtc.hiprtcDestroyProgram(handle.createRef())
    return log.decode()


def null_stream() -> "hip.hipStream_t | None":
    """A handle typedef, the name callers annotate with."""
    if not hip.has_symbol("hipStreamCreate"):
        return None
    _, stream = hip.hipStreamCreate()
    return stream


def versions() -> tuple[int, str]:
    """The HIP runtime version, and the ROCm version the wheel was cut for."""
    _, runtime_version = hip.hipRuntimeGetVersion()
    return int(runtime_version), HIP_VERSION_NAME


def hiprtc_version() -> tuple[int, int]:
    """Major and minor version of the installed HIPRTC."""
    _, major, minor = hiprtc.hiprtcVersion()
    return int(major), int(minor)
