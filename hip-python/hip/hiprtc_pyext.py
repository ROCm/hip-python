#!/usr/bin/env python3
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

import ctypes

from hip import hiprtc as _hiprtc
from hip._util import types as _types

__all__ = [
    "HiprtcLinkCreateOpts",
    "hiprtcLinkCreate2",
]


class HiprtcLinkCreateOpts:
    r"""Converts a Python map to appropriate argument types for `~.hiprtcLinkCreate`.

    Implements ``__getitem__`` to allow writing:

    ```python
    link_state = hiprtcLinkCreate(*HiprtcLinkCreateOpts(
        HIPRTC_JIT_FAST_COMPILE=...,
        HIPRTC_JIT_TARGET=...,
        ...
    ))
    ```
    """

    def __init__(self, **kwargs):
        """Constructor.

        Construct hiprtcLinkCreate argument list via keyword arguments.

        Args:
            \*\*kwargs:
                The names of enum constants of type `_hiprtc.hiprtcJIT_option` and the
                corresponding value.

                The following key-value pairs may be used (state: ROCm 6.0.0):

                | Key                                    | Suitable value                                                    |
                |----------------------------------------|-------------------------------------------------------------------|
                | HIPRTC_JIT_NEW_SM3X_OPT                | All suitable arguments for `ctypes.c_bool`.                       |
                | HIPRTC_JIT_FAST_COMPILE                | ...                                                               |
                | HIPRTC_JIT_MAX_REGISTERS               | All suitable arguments for `ctypes.c_uint`.                       |
                | HIPRTC_JIT_THREADS_PER_BLOCK           | ...                                                               |
                | HIPRTC_JIT_OPTIMIZATION_LEVEL          | ...                                                               |
                | HIPRTC_JIT_TARGET_FROM_HIPCONTEXT      | ...                                                               |
                | HIPRTC_JIT_TARGET                      | ...                                                               |
                | HIPRTC_JIT_FALLBACK_STRATEGY           | ...                                                               |
                | HIPRTC_JIT_CACHE_MODE                  | ...                                                               |
                | HIPRTC_JIT_GLOBAL_SYMBOL_COUNT         | ...                                                               |
                | HIPRTC_JIT_GENERATE_DEBUG_INFO         | All suitable arguments for `ctypes.c_int`.                        |
                | HIPRTC_JIT_GENERATE_LINE_INFO          | ...                                                               |
                | HIPRTC_JIT_LTO                         | ...                                                               |
                | HIPRTC_JIT_FTZ                         | ...                                                               |
                | HIPRTC_JIT_PREC_DIV                    | ...                                                               |
                | HIPRTC_JIT_PREC_SQRT                   | ...                                                               |
                | HIPRTC_JIT_FMA                         | ...                                                               |
                | HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES  | Integer values. Must be suitable arguments for `ctypes.c_void_p`. |
                | HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES | ...                                                               |
                | HIPRTC_JIT_LOG_VERBOSE                 | ...                                                               |
                | HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT     | ...                                                               |
                | HIPRTC_JIT_WALL_TIME                   | ...                                                               |
                | HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS       | See `~._types.ListOfPointer`.                                     |
                | HIPRTC_JIT_GLOBAL_SYMBOL_NAMES         | See `~._types.ListOfBytes`.                                       |
                | HIPRTC_JIT_IR_TO_ISA_OPT_EXT           | ...                                                               |
                | HIPRTC_JIT_INFO_LOG_BUFFER             | See `~._types.Pointer`.                                           |
                | HIPRTC_JIT_ERROR_LOG_BUFFER            | ...                                                               |
        """
        if not kwargs:
            self.num_opts = 0
            self.keys = None
            self.values = None
            return
        self.num_opts = len(kwargs)
        self.keys = []
        self.values = []
        for key_str in kwargs.keys():
            value = kwargs[key_str]
            key = getattr(_hiprtc.hiprtcJIT_option, key_str, None)
            self.keys.append(key)
            if not key:
                raise KeyError(
                    f"key '{key_str}' is not the name of an enum constant of type '_hiprtc.hiprtcJIT_option'"
                )
            if key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_NEW_SM3X_OPT,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_FAST_COMPILE,
            ):
                self.values.append(ctypes.addressof(ctypes.c_bool(value)))
            if key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_MAX_REGISTERS,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_THREADS_PER_BLOCK,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_OPTIMIZATION_LEVEL,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_TARGET_FROM_HIPCONTEXT,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_TARGET,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_FALLBACK_STRATEGY,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_CACHE_MODE,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_GLOBAL_SYMBOL_COUNT,
            ):
                self.values.append(ctypes.addressof(ctypes.c_uint(value)))
            elif key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_GENERATE_DEBUG_INFO,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_GENERATE_LINE_INFO,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_LTO,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_FTZ,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_PREC_DIV,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_PREC_SQRT,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_FMA,
            ):
                self.values.append(ctypes.addressof(ctypes.c_int(value)))
            elif key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES,  # size_t
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,  # size_t
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_LOG_VERBOSE,  # size_t
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT,  # size_t
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_WALL_TIME,  # long
            ):
                self.values.append(
                    ctypes.c_void_p(
                        value
                    )  # actually size_t/long types but hiprtcLinkCreate `values`
                    # arg will be handled by `_types.ListOfPointer` adapter
                )
            elif key == _hiprtc.hiprtcJIT_option.HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS:
                self.values.append(_types.ListOfPointer(value))
            elif key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_GLOBAL_SYMBOL_NAMES,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
            ):
                self.values.append(_types.ListOfBytes(value))
            elif key in (
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_INFO_LOG_BUFFER,
                _hiprtc.hiprtcJIT_option.HIPRTC_JIT_ERROR_LOG_BUFFER,
            ):
                self.values.append(
                    _types.Pointer(
                        value
                    )  # obtains pointer to bytes' data via Python buffer protocol
                )
            else:
                raise NotImplementedError(f"could not handle key '{key_str}'")

    def __len__(self):
        return 3

    def __getitem__(self, item):
        """Allows to unpack the members via the * operator.
        The unpacking is done in the order expected by the argument
        list of `~.hiprtc.hiprtcLinkCreate`.
        """
        if isinstance(item, int):
            if item == 0:
                return self.num_opts
            elif item == 1:
                return self.keys
            elif item == 2:
                return self.values
            else:
                raise IndexError()
        raise ValueError("'item' type must be 'int'")


def hiprtcLinkCreate2(**kwargs):
    """Variant of `~.hiprtc.hiprtcLinkCreate` that takes link options via keyword args.

    Variant of `~.hiprtc.hiprtcLinkCreate` that takes link options via keyword args:

    ```python
    link_state = hiprtcLinkCreate2(
        HIPRTC_JIT_FAST_COMPILE=...,
        HIPRTC_JIT_TARGET=...,
        ...
    )
    ```

    Args:
        \*\*kwargs:
            The names of enum constants of type `_hiprtc.hiprtcJIT_option` and the
            corresponding value.

            The following key-value pairs may be used (state: ROCm 6.0.0):

            | Key                                    | Suitable value                                                    |
            |----------------------------------------|-------------------------------------------------------------------|
            | HIPRTC_JIT_NEW_SM3X_OPT                | All suitable arguments for `ctypes.c_bool`.                       |
            | HIPRTC_JIT_FAST_COMPILE                | ...                                                               |
            | HIPRTC_JIT_MAX_REGISTERS               | All suitable arguments for `ctypes.c_uint`.                       |
            | HIPRTC_JIT_THREADS_PER_BLOCK           | ...                                                               |
            | HIPRTC_JIT_OPTIMIZATION_LEVEL          | ...                                                               |
            | HIPRTC_JIT_TARGET_FROM_HIPCONTEXT      | ...                                                               |
            | HIPRTC_JIT_TARGET                      | ...                                                               |
            | HIPRTC_JIT_FALLBACK_STRATEGY           | ...                                                               |
            | HIPRTC_JIT_CACHE_MODE                  | ...                                                               |
            | HIPRTC_JIT_GLOBAL_SYMBOL_COUNT         | ...                                                               |
            | HIPRTC_JIT_GENERATE_DEBUG_INFO         | All suitable arguments for `ctypes.c_int`.                        |
            | HIPRTC_JIT_GENERATE_LINE_INFO          | ...                                                               |
            | HIPRTC_JIT_LTO                         | ...                                                               |
            | HIPRTC_JIT_FTZ                         | ...                                                               |
            | HIPRTC_JIT_PREC_DIV                    | ...                                                               |
            | HIPRTC_JIT_PREC_SQRT                   | ...                                                               |
            | HIPRTC_JIT_FMA                         | ...                                                               |
            | HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES  | Integer values. Must be suitable arguments for `ctypes.c_void_p`. |
            | HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES | ...                                                               |
            | HIPRTC_JIT_LOG_VERBOSE                 | ...                                                               |
            | HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT     | ...                                                               |
            | HIPRTC_JIT_WALL_TIME                   | ...                                                               |
            | HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS       | See `~._types.ListOfPointer`.                                     |
            | HIPRTC_JIT_GLOBAL_SYMBOL_NAMES         | See `~._types.ListOfBytes`.                                       |
            | HIPRTC_JIT_IR_TO_ISA_OPT_EXT           | ...                                                               |
            | HIPRTC_JIT_INFO_LOG_BUFFER             | See `~._types.Pointer`.                                           |
            | HIPRTC_JIT_ERROR_LOG_BUFFER            | ...                                                               |

    See:
        `~.HiprtcLinkCreateOpts`.
    """
    return _hiprtc.hiprtcLinkCreate(*HiprtcLinkCreateOpts(**kwargs))
