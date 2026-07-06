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

"""Color helpers for the ``nvtx`` compatibility shim.

This mirrors the ``nvtx.colors`` submodule of the upstream (Apache-2.0)
``nvtx`` package so that code importing ``nvtx.colors`` keeps working. It is a
fresh, MIT-licensed re-implementation.

.. important::

   Colors have **no runtime effect** in this shim. The underlying
   ``rocm.bindings.roctx`` bindings are message-only and expose no color
   channel, so any color computed here is dropped before the event reaches
   roctx. These helpers exist purely for source compatibility.
"""

import functools

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

_NVTX_COLORS = {
    None: 0x0000FF,  # blue (default)
    "green": 0x008000,
    "blue": 0x0000FF,
    "yellow": 0xFFFF00,
    "purple": 0x800080,
    "rapids": 0x7400FF,
    "cyan": 0x00FFFF,
    "red": 0xFF0000,
    "white": 0xFFFFFF,
    "darkgreen": 0x006400,
    "orange": 0xFFA500,
}


@functools.lru_cache()
def color_to_hex(color=None):
    """Convert a color to its ARGB hex value.

    Accepts an integer (returned unchanged), one of the built-in color names,
    or - when ``matplotlib`` is installed - any matplotlib color spec.

    Note:
        The returned value is not forwarded to roctx (which has no color
        channel); this function is provided for API compatibility only.
    """
    if isinstance(color, int):
        return color
    if color in _NVTX_COLORS:
        return _NVTX_COLORS[color]
    try:
        import matplotlib.colors
    except ImportError as e:
        raise TypeError(
            f"Invalid color {color!r}. Please install matplotlib "
            "for additional color support."
        ) from e
    rgba = matplotlib.colors.to_rgba(color)
    argb = (rgba[-1], rgba[0], rgba[1], rgba[2])
    return int(matplotlib.colors.to_hex(argb, keep_alpha=True)[1:], 16)
