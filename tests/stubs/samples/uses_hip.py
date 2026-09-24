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
    # The generated stubs declare the enum classes but not their members.
    # Drop the suppression once they do: pyright then reports it as
    # unnecessary, which is an error under this suite's config. Keep the
    # access short enough that black leaves the comment on its line.
    status = hip.hipError_t
    if err != status.hipSuccess:  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(f"hipGetDeviceProperties({index}): {err}")
    return str(props.gcnArchName)


def versions() -> tuple[int, str]:
    """The HIP runtime version, and the ROCm version the wheel was cut for."""
    _, runtime_version = hip.hipRuntimeGetVersion()
    return int(runtime_version), HIP_VERSION_NAME


def hiprtc_version() -> tuple[int, int]:
    """Major and minor version of the installed HIPRTC."""
    _, major, minor = hiprtc.hiprtcVersion()
    return int(major), int(minor)
