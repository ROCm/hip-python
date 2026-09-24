rocm.bindings.hiprtc_pyext
==========================

.. py:module:: rocm.bindings.hiprtc_pyext


Classes
-------

.. autoapisummary::

   rocm.bindings.hiprtc_pyext.HiprtcLinkCreateOpts


Functions
---------

.. autoapisummary::

   rocm.bindings.hiprtc_pyext.hiprtcLinkCreate2


Module Contents
---------------

.. py:class:: HiprtcLinkCreateOpts(**kwargs)

   Converts a Python map to appropriate argument types for `~.hiprtcLinkCreate`.

   .. deprecated::
       Use `~.rocm.bindings.hiprtc` directly. This helper predates the
       modern ``rocm.bindings`` bindings and is kept only for backward
       compatibility; it will be removed in a future release.

   Implements ``__getitem__`` to allow writing:

   ```python
   link_state = hiprtcLinkCreate(*HiprtcLinkCreateOpts(
       HIPRTC_JIT_FAST_COMPILE=...,
       HIPRTC_JIT_TARGET=...,
       hipJitOptionLto=...,
       ...
   ))
   ```


   .. py:attribute:: num_opts
      :value: 0



   .. py:attribute:: keys
      :value: []



   .. py:attribute:: values
      :value: []



.. py:function:: hiprtcLinkCreate2(**kwargs)

   Variant of `~.hiprtc.hiprtcLinkCreate` that takes link options via keyword args.

   .. deprecated::
       Use `~.rocm.bindings.hiprtc` directly. This helper predates the
       modern ``rocm.bindings`` bindings and is kept only for backward
       compatibility; it will be removed in a future release.

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


