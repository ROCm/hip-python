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

"""Annotated application code against `rocm.bindings`, the spelling new
code should use.

Type-checked, never run. See `../test_pyright_samples.py`.
"""

from rocm.bindings import hip, hiprtc
from rocm.version import HIP_VERSION_TUPLE, ROCM_VERSION_NAME


def allocate(nbytes: int) -> object:
    """A device allocation, as the caller would wrap it."""
    err, pointer = hip.hipMalloc(nbytes)
    if int(err) != 0:
        raise MemoryError(f"hipMalloc({nbytes}): {err}")
    return pointer


def release(pointer: object) -> None:
    hip.hipFree(pointer)


def compile_to_code_object(source: str, name: str) -> bytes:
    """Compile a kernel with HIPRTC and hand back its code object."""
    _, program = hiprtc.hiprtcCreateProgram(
        source.encode(), name.encode(), 0, [], []
    )
    try:
        hiprtc.hiprtcCompileProgram(program, 0, [])
        _, size = hiprtc.hiprtcGetCodeSize(program)
        code = bytearray(size)
        hiprtc.hiprtcGetCode(program, code)
    finally:
        hiprtc.hiprtcDestroyProgram(program)
    return bytes(code)


def target_rocm() -> tuple[str, tuple[int | str, ...]]:
    """What ROCm this build of the bindings belongs to.

    The HIP tuple carries the build's commit hash as its last element, so
    it is not a tuple of ints.
    """
    return ROCM_VERSION_NAME, HIP_VERSION_TUPLE
