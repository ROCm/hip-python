# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Minimal ``cuda.core`` compatibility shim, backed by HIP.

This is NOT a full port of NVIDIA's ``cuda.core`` / ``cuda.core.experimental``
package. It implements just enough of the high-level surface for HIP ports of
CUDA-Python consumers: a `~.Device` exposing ``uuid``.

Everything is implemented on top of the high-level ``rocm.bindings.hip`` HIP
runtime API. Additional high-level abstractions (Stream, Buffer, Program, ...)
are intentionally out of scope and can be added on demand.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

from rocm.bindings import hip

__all__ = ["Device"]


def _check(err):
    """Raise on a non-success HIP status returned by a ``rocm.bindings.hip`` call."""
    if int(err) != int(hip.hipError_t.hipSuccess):
        raise RuntimeError(f"HIP error in cuda.core: {err!r}")


class Device:
    """HIP-backed stand-in for ``cuda.core.Device``.

    Only the members required by current consumers are implemented. Passing no
    ``device_id`` selects the current device (via ``hipGetDevice``).
    """

    def __init__(self, device_id=None):
        if device_id is None:
            err, device_id = hip.hipGetDevice()
            _check(err)
        self._id = int(device_id)

    @property
    def device_id(self) -> int:
        """Ordinal of the underlying HIP device."""
        return self._id

    @property
    def uuid(self) -> str:
        """Device UUID as a ``GPU-<hex>`` string.

        Backed by ``hipDeviceGetUuid``. ROCm stores the UUID as ASCII text in
        the 16-byte field (e.g. ``b"50a437de77a657b8"``), so the result matches
        the ``GPU-<hex>`` string reported by ``rocminfo`` / ``amd-smi``. Falls
        back to a raw hex rendering if a runtime ever returns non-text bytes.
        """
        err, handle = hip.hipDeviceGetUuid(self._id)
        _check(err)
        raw = bytes(handle.get_bytes(0))[:16]
        stripped = raw.rstrip(b"\x00")
        try:
            text = stripped.decode("ascii")
            is_text = bool(text) and text.isprintable()
        except UnicodeDecodeError:
            is_text = False
        if not is_text:
            text = raw.hex()
        return text if text.startswith("GPU-") else f"GPU-{text}"

    def __repr__(self):
        return f"<cuda.core.Device id={self._id} (HIP)>"

    def set_current(self):
        err = hip.hipSetDevice(self._id)[0]
        _check(err)

    def sync(self):
        err = hip.hipDeviceSynchronize()[0]
        _check(err)
