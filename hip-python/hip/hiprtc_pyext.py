#!/usr/bin/env python3
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
        hipJitOptionLto=...,
        ...
    ))
    ```
    """

    def __init__(self, **kwargs):
        r"""Constructor.

        Construct hiprtcLinkCreate argument list via keyword arguments.

        NOTE:

        Args:
            \*\*kwargs:
                The names of enum constants of type `~.hiprtcJIT_option` (exist before ROCm 6.4.0) /
                `~.hipJitOption` (exists after ROCm 6.4.0) plus a corresponding suitable value.

                The following key-value pairs may be used (state: ROCm 6.4.0):

                | ``hipJitOption`` key                    | ``hiprtcJIT_option`` key                   | Suitable value                                                      |
                |-----------------------------------------|--------------------------------------------|---------------------------------------------------------------------|
                | ``hipJitOptionSm3xOpt``                 | ``HIPRTC_JIT_NEW_SM3X_OPT``                | All suitable arguments for `ctypes.c_bool`.                         |
                | ``hipJitOptionFastCompile``             | ``HIPRTC_JIT_FAST_COMPILE``                | ...                                                                 |
                | ``hipJitOptionMaxRegisters``            | ``HIPRTC_JIT_MAX_REGISTERS``               | All suitable arguments for `ctypes.c_uint`.                         |
                | ``hipJitOptionThreadsPerBlock``         | ``HIPRTC_JIT_THREADS_PER_BLOCK``           | ...                                                                 |
                | ``hipJitOptionOptimizationLevel``       | ``HIPRTC_JIT_OPTIMIZATION_LEVEL``          | ...                                                                 |
                | ``hipJitOptionTargetFromContext``       | ``HIPRTC_JIT_TARGET_FROM_HIPCONTEXT``      | ...                                                                 |
                | ``hipJitOptionTarget``                  | ``HIPRTC_JIT_TARGET``                      | ...                                                                 |
                | ``hipJitOptionFallbackStrategy``        | ``HIPRTC_JIT_FALLBACK_STRATEGY``           | ...                                                                 |
                | ``hipJitOptionCacheMode``               | ``HIPRTC_JIT_CACHE_MODE``                  | ...                                                                 |
                | ``hipJitOptionGlobalSymbolCount``       | ``HIPRTC_JIT_GLOBAL_SYMBOL_COUNT``         | ...                                                                 |
                | ``hipJitOptionGenerateDebugInfo``       | ``HIPRTC_JIT_GENERATE_DEBUG_INFO``         | All suitable arguments for `ctypes.c_int`.                          |
                | ``hipJitOptionGenerateLineInfo``        | ``HIPRTC_JIT_GENERATE_LINE_INFO``          | ...                                                                 |
                | ``hipJitOptionLto``                     | ``HIPRTC_JIT_LTO``                         | ...                                                                 |
                | ``hipJitOptionFtz``                     | ``HIPRTC_JIT_FTZ``                         | ...                                                                 |
                | ``hipJitOptionPrecDiv``                 | ``HIPRTC_JIT_PREC_DIV``                    | ...                                                                 |
                | ``hipJitOptionPrecSqrt``                | ``HIPRTC_JIT_PREC_SQRT``                   | ...                                                                 |
                | ``hipJitOptionFma``                     | ``HIPRTC_JIT_FMA``                         | ...                                                                 |
                | ``hipJitOptionPositionIndependentCode`` | ``HIPRTC_JIT_POSITION_INDEPENDENT_CODE``   | ...                                                                 |
                | ``hipJitOptionMinCTAPerSM``             | ``HIPRTC_JIT_MIN_CTA_PER_SM``              | ...                                                                 |
                | ``hipJitOptionMaxThreadsPerBlock``      | ``HIPRTC_JIT_MAX_THREADS_PER_BLOCK``       | ...                                                                 |
                | ``hipJitOptionOverrideDirectiveValues`` | ``HIPRTC_JIT_OVERRIDE_DIRECT_VALUES``      | ...                                                                 |
                | ``hipJitOptionInfoLogBufferSizeBytes``  | ``HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES``  | Integer values. Must be suitable arguments for `ctypes.c_void_p`.   |
                | ``hipJitOptionErrorLogBufferSizeBytes`` | ``HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`` | ...                                                                 |
                | ``hipJitOptionLogVerbose``              | ``HIPRTC_JIT_LOG_VERBOSE``                 | ...                                                                 |
                | ``hipJitOptionIRtoISAOptCountExt``      | ``HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT``     | ...                                                                 |
                | ``hipJitOptionWallTime``                | ``HIPRTC_JIT_WALL_TIME``                   | ...                                                                 |
                | ``hipJitOptionGlobalSymbolAddresses``   | ``HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS``       | See `~._types.ListOfPointer`.                                       |
                | ``hipJitOptionGlobalSymbolNames``       | ``HIPRTC_JIT_GLOBAL_SYMBOL_NAMES``         | See `~._types.ListOfBytes`.                                         |
                | ``hipJitOptionIRtoISAOptExt``           | ``HIPRTC_JIT_IR_TO_ISA_OPT_EXT``           | ...                                                                 |
                | ``hipJitOptionInfoLogBuffer``           | ``HIPRTC_JIT_INFO_LOG_BUFFER``             | See `~._types.Pointer`.                                             |
                | ``hipJitOptionErrorLogBufferSizeBytes`` | ``HIPRTC_JIT_ERROR_LOG_BUFFER``            | ...                                                                 |
                | ``hipJitOptionNumOptions``              | ``HIPRTC_JIT_NUM_OPTIONS``                 | Could not be deduced, likely suitable arguments for `ctypes.c_int`. |

        Implementation details:

            This implementation is based on the following routine:

            https://github.com/ROCm/clr/blob/78d0ff2dbca16a4f7e5691204fe7e5966e25bb11/hipamd/src/hip_comgr_helper.cpp#L1388

        Intended use:

            This object can be unrolled, which yields three elements that can be passed
            as arguments to `~.hiprtc.hiprtcLinkCreate`:

            * Element #0 is the number of options that should be passed to ``hiprtcLinkCreate``.
            * Element #1 are the option keys, a list of enum constant values that specify what
              options the user should set. It is important that the type is accepted as valid
              constructor argument for `~.hip._hiprtc_helpers.HiprtcLinkCreate_option_ptr` as this
              type is used adapter for argument #1 of ``hiprtcLinkCreate``.
            * Element #2 are the option values. Here, ``hiprtcLinkCreate`` expects
              a type that is accepted by the `~.hip._util.types.ListOfPointer` adapter used for
              argument #2. This adapter takes a list of objects that must be in turn accepted
              by the `~.hip._util.types.Pointer` adapter type. Aside from the other `~.hip._util.types.*`
              adapter types and `ctypes.c_void_p`, the latter can be constructed
              from an `int` value, which is then interpreted as pointer address.
              Therefore, we see expressions alike ``self.values.append(int(ctypes.c_uint(value).value))``
              in the body of this function.

        Note:
            Many of the options may not be implemented by hipRTC, see `~.hiprtcJIT_option` (exist before ROCm 6.4.0) or
            `~.hiprtcJIT_option` (exists after ROCm 6.4.0) for more details.

        """
        if not kwargs:
            self.num_opts = 0
            self.keys = None
            self.values = None
            return
        self.num_opts = len(kwargs)
        self.keys = []
        self.values = []

        has_hip_jit_option = hasattr(_hiprtc, "hipJitOption")
        jit_option_type = (
            _hiprtc.hipJitOption
            if has_hip_jit_option
            else _hiprtc.hiprtcJIT_option
        )

        hiprtc_to_hip_jit_options = dict(
            HIPRTC_JIT_MAX_REGISTERS="hipJitOptionMaxRegisters",
            HIPRTC_JIT_THREADS_PER_BLOCK="hipJitOptionThreadsPerBlock",
            HIPRTC_JIT_WALL_TIME="hipJitOptionWallTime",
            HIPRTC_JIT_INFO_LOG_BUFFER="hipJitOptionInfoLogBuffer",
            HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES="hipJitOptionInfoLogBufferSizeBytes",
            HIPRTC_JIT_ERROR_LOG_BUFFER="hipJitOptionErrorLogBuffer",
            HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES="hipJitOptionErrorLogBufferSizeBytes",
            HIPRTC_JIT_OPTIMIZATION_LEVEL="hipJitOptionOptimizationLevel",
            HIPRTC_JIT_TARGET_FROM_HIPCONTEXT="hipJitOptionTargetFromContext",
            HIPRTC_JIT_TARGET="hipJitOptionTarget",
            HIPRTC_JIT_FALLBACK_STRATEGY="hipJitOptionFallbackStrategy",
            HIPRTC_JIT_GENERATE_DEBUG_INFO="hipJitOptionGenerateDebugInfo",
            HIPRTC_JIT_LOG_VERBOSE="hipJitOptionLogVerbose",
            HIPRTC_JIT_GENERATE_LINE_INFO="hipJitOptionGenerateLineInfo",
            HIPRTC_JIT_CACHE_MODE="hipJitOptionCacheMode",
            HIPRTC_JIT_NEW_SM3X_OPT="hipJitOptionSm3xOpt",
            HIPRTC_JIT_FAST_COMPILE="hipJitOptionFastCompile",
            HIPRTC_JIT_GLOBAL_SYMBOL_NAMES="hipJitOptionGlobalSymbolNames",
            HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS="hipJitOptionGlobalSymbolAddresses",
            HIPRTC_JIT_GLOBAL_SYMBOL_COUNT="hipJitOptionGlobalSymbolCount",
            HIPRTC_JIT_LTO="hipJitOptionLto",
            HIPRTC_JIT_FTZ="hipJitOptionFtz",
            HIPRTC_JIT_PREC_DIV="hipJitOptionPrecDiv",
            HIPRTC_JIT_PREC_SQRT="hipJitOptionPrecSqrt",
            HIPRTC_JIT_FMA="hipJitOptionFma",
            HIPRTC_JIT_POSITION_INDEPENDENT_CODE="hipJitOptionPositionIndependentCode",
            HIPRTC_JIT_MIN_CTA_PER_SM="hipJitOptionMinCTAPerSM",
            HIPRTC_JIT_MAX_THREADS_PER_BLOCK="hipJitOptionMaxThreadsPerBlock",
            HIPRTC_JIT_OVERRIDE_DIRECT_VALUES="hipJitOptionOverrideDirectiveValues",
            HIPRTC_JIT_NUM_OPTIONS="hipJitOptionNumOptions",
            HIPRTC_JIT_IR_TO_ISA_OPT_EXT="hipJitOptionIRtoISAOptExt",
            HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT="hipJitOptionIRtoISAOptCountExt",
        )

        for key_str in kwargs.keys():
            if has_hip_jit_option and key_str.startswith("HIPRTC_"):
                key = getattr(
                    jit_option_type, hiprtc_to_hip_jit_options[key_str]
                )
            else:
                key = getattr(jit_option_type, key_str)

            value = kwargs[key_str]

            self.keys.append(key)
            if not key:
                raise KeyError(
                    f"key '{key_str}' is not the name of an enum constant of type '_hiprtc.hiprtcJIT_option'"
                )
            if key_str in (
                "HIPRTC_JIT_NEW_SM3X_OPT",
                "HIPRTC_JIT_FAST_COMPILE",
                "hipJitOptionSm3xOpt",
                "hipJitOptionFastCompile",
            ):
                self.values.append(int(ctypes.c_bool(value).value))
            elif key_str in (
                "HIPRTC_JIT_MAX_REGISTERS",
                "HIPRTC_JIT_THREADS_PER_BLOCK",
                "HIPRTC_JIT_OPTIMIZATION_LEVEL",
                "HIPRTC_JIT_TARGET_FROM_HIPCONTEXT",
                "HIPRTC_JIT_TARGET",
                "HIPRTC_JIT_FALLBACK_STRATEGY",
                "HIPRTC_JIT_CACHE_MODE",
                "HIPRTC_JIT_GLOBAL_SYMBOL_COUNT",
                "hipJitOptionMaxRegisters",
                "hipJitOptionThreadsPerBlock",
                "hipJitOptionOptimizationLevel",
                "hipJitOptionTargetFromContext",
                "hipJitOptionTarget",
                "hipJitOptionFallbackStrategy",
                "hipJitOptionCacheMode",
                "hipJitOptionGlobalSymbolCount",
            ):
                self.values.append(int(ctypes.c_uint(value).value))
            elif key_str in (
                "HIPRTC_JIT_GENERATE_DEBUG_INFO",
                "HIPRTC_JIT_GENERATE_LINE_INFO",
                "HIPRTC_JIT_LTO",
                "HIPRTC_JIT_FTZ",
                "HIPRTC_JIT_PREC_DIV",
                "HIPRTC_JIT_PREC_SQRT",
                "HIPRTC_JIT_FMA",
                "hipJitOptionGenerateDebugInfo",
                "hipJitOptionGenerateLineInfo",
                "hipJitOptionLto",
                "hipJitOptionFtz",
                "hipJitOptionPrecDiv",
                "hipJitOptionPrecSqrt",
                "hipJitOptionFma",
                "HIPRTC_JIT_POSITION_INDEPENDENT_CODE",
                "HIPRTC_JIT_MIN_CTA_PER_SM",
                "HIPRTC_JIT_MAX_THREADS_PER_BLOCK",
                "HIPRTC_JIT_OVERRIDE_DIRECT_VALUES",
                "hipJitOptionPositionIndependentCode",
                "hipJitOptionMinCTAPerSM",
                "hipJitOptionMaxThreadsPerBlock",
                "hipJitOptionOverrideDirectiveValues",
            ):
                self.values.append(int(ctypes.c_int(value).value))
            elif key_str in (
                "HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES",  # size_t
                "HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES",  # size_t
                "HIPRTC_JIT_LOG_VERBOSE",  # size_t
                "HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT",  # size_t
                "HIPRTC_JIT_WALL_TIME",  # long
                "hipJitOptionInfoLogBufferSizeBytes",  # size_t
                "hipJitOptionErrorLogBufferSizeBytes",  # size_t
                "hipJitOptionLogVerbose",  # size_t
                "hipJitOptionIRtoISAOptCountExt",  # size_t
                "hipJitOptionWallTime",  # long
            ):
                self.values.append(
                    ctypes.c_void_p(
                        value
                    )  # actually size_t/long types but hiprtcLinkCreate `values`
                    # arg will be handled by `_types.ListOfPointer` adapter
                )
            elif key_str in (
                "HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS",
                "hipJitOptionGlobalSymbolAddresses",
            ):
                self.values.append(_types.ListOfPointer(value))
            elif key_str in (
                "HIPRTC_JIT_GLOBAL_SYMBOL_NAMES",
                "HIPRTC_JIT_IR_TO_ISA_OPT_EXT",
                "hipJitOptionGlobalSymbolNames",
                "hipJitOptionIRtoISAOptExt",
            ):
                self.values.append(_types.ListOfBytes(value))
            elif key_str in (
                "HIPRTC_JIT_INFO_LOG_BUFFER",
                "HIPRTC_JIT_ERROR_LOG_BUFFER",
                "hipJitOptionInfoLogBuffer",
                "hipJitOptionErrorLogBuffer",
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
    r"""Variant of `~.hiprtc.hiprtcLinkCreate` that takes link options via keyword args.

    Variant of `~.hiprtc.hiprtcLinkCreate` that takes link options via keyword args:

    ```python
    link_state = hiprtcLinkCreate2(
        HIPRTC_JIT_FAST_COMPILE=...,
        HIPRTC_JIT_TARGET=...,
        hipJitOptionLto=...,
        ...
    )
    ```

    Args:
        \*\*kwargs:
            The names of enum constants of type `~.hiprtcJIT_option` (exist before ROCm 6.4.0) or
            `~.hipJitOption` (exists after ROCm 6.4.0) plus a corresponding suitable value.

            The following key-value pairs may be used (state: ROCm 6.4.0):

            | ``hipJitOption`` key                    | ``hiprtcJIT_option`` key                   | Suitable value                                                      |
            |-----------------------------------------|--------------------------------------------|---------------------------------------------------------------------|
            | ``hipJitOptionSm3xOpt``                 | ``HIPRTC_JIT_NEW_SM3X_OPT``                | All suitable arguments for `ctypes.c_bool`.                         |
            | ``hipJitOptionFastCompile``             | ``HIPRTC_JIT_FAST_COMPILE``                | ...                                                                 |
            | ``hipJitOptionMaxRegisters``            | ``HIPRTC_JIT_MAX_REGISTERS``               | All suitable arguments for `ctypes.c_uint`.                         |
            | ``hipJitOptionThreadsPerBlock``         | ``HIPRTC_JIT_THREADS_PER_BLOCK``           | ...                                                                 |
            | ``hipJitOptionOptimizationLevel``       | ``HIPRTC_JIT_OPTIMIZATION_LEVEL``          | ...                                                                 |
            | ``hipJitOptionTargetFromContext``       | ``HIPRTC_JIT_TARGET_FROM_HIPCONTEXT``      | ...                                                                 |
            | ``hipJitOptionTarget``                  | ``HIPRTC_JIT_TARGET``                      | ...                                                                 |
            | ``hipJitOptionFallbackStrategy``        | ``HIPRTC_JIT_FALLBACK_STRATEGY``           | ...                                                                 |
            | ``hipJitOptionCacheMode``               | ``HIPRTC_JIT_CACHE_MODE``                  | ...                                                                 |
            | ``hipJitOptionGlobalSymbolCount``       | ``HIPRTC_JIT_GLOBAL_SYMBOL_COUNT``         | ...                                                                 |
            | ``hipJitOptionGenerateDebugInfo``       | ``HIPRTC_JIT_GENERATE_DEBUG_INFO``         | All suitable arguments for `ctypes.c_int`.                          |
            | ``hipJitOptionGenerateLineInfo``        | ``HIPRTC_JIT_GENERATE_LINE_INFO``          | ...                                                                 |
            | ``hipJitOptionLto``                     | ``HIPRTC_JIT_LTO``                         | ...                                                                 |
            | ``hipJitOptionFtz``                     | ``HIPRTC_JIT_FTZ``                         | ...                                                                 |
            | ``hipJitOptionPrecDiv``                 | ``HIPRTC_JIT_PREC_DIV``                    | ...                                                                 |
            | ``hipJitOptionPrecSqrt``                | ``HIPRTC_JIT_PREC_SQRT``                   | ...                                                                 |
            | ``hipJitOptionFma``                     | ``HIPRTC_JIT_FMA``                         | ...                                                                 |
            | ``hipJitOptionPositionIndependentCode`` | ``HIPRTC_JIT_POSITION_INDEPENDENT_CODE``   | ...                                                                 |
            | ``hipJitOptionMinCTAPerSM``             | ``HIPRTC_JIT_MIN_CTA_PER_SM``              | ...                                                                 |
            | ``hipJitOptionMaxThreadsPerBlock``      | ``HIPRTC_JIT_MAX_THREADS_PER_BLOCK``       | ...                                                                 |
            | ``hipJitOptionOverrideDirectiveValues`` | ``HIPRTC_JIT_OVERRIDE_DIRECT_VALUES``      | ...                                                                 |
            | ``hipJitOptionInfoLogBufferSizeBytes``  | ``HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES``  | Integer values. Must be suitable arguments for `ctypes.c_void_p`.   |
            | ``hipJitOptionErrorLogBufferSizeBytes`` | ``HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`` | ...                                                                 |
            | ``hipJitOptionLogVerbose``              | ``HIPRTC_JIT_LOG_VERBOSE``                 | ...                                                                 |
            | ``hipJitOptionIRtoISAOptCountExt``      | ``HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT``     | ...                                                                 |
            | ``hipJitOptionWallTime``                | ``HIPRTC_JIT_WALL_TIME``                   | ...                                                                 |
            | ``hipJitOptionGlobalSymbolAddresses``   | ``HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS``       | See `~._types.ListOfPointer`.                                       |
            | ``hipJitOptionGlobalSymbolNames``       | ``HIPRTC_JIT_GLOBAL_SYMBOL_NAMES``         | See `~._types.ListOfBytes`.                                         |
            | ``hipJitOptionIRtoISAOptExt``           | ``HIPRTC_JIT_IR_TO_ISA_OPT_EXT``           | ...                                                                 |
            | ``hipJitOptionInfoLogBuffer``           | ``HIPRTC_JIT_INFO_LOG_BUFFER``             | See `~._types.Pointer`.                                             |
            | ``hipJitOptionErrorLogBufferSizeBytes`` | ``HIPRTC_JIT_ERROR_LOG_BUFFER``            | ...                                                                 |
            | ``hipJitOptionNumOptions``              | ``HIPRTC_JIT_NUM_OPTIONS``                 | Could not be deduced, likely suitable arguments for `ctypes.c_int`. |

    Note:
        Many of the options may not be implemented by hipRTC, see `~.hiprtcJIT_option` (exist before ROCm 6.4.0) or
        `~.hiprtcJIT_option` (exists after ROCm 6.4.0) for more details.

    See:
        `~.HiprtcLinkCreateOpts`.
    """
    return _hiprtc.hiprtcLinkCreate(*HiprtcLinkCreateOpts(**kwargs))
