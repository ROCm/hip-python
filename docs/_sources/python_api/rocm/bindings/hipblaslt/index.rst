rocm.bindings.hipblaslt
=======================

.. py:module:: rocm.bindings.hipblaslt


Attributes
----------

.. autoapisummary::

   rocm.bindings.hipblaslt.HIPBLASLT_VERSION_MAJOR
   rocm.bindings.hipblaslt.HIPBLASLT_VERSION_MINOR
   rocm.bindings.hipblaslt.HIPBLASLT_VERSION_PATCH
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDesc_t
   rocm.bindings.hipblaslt.hipblasLtMatmulDesc_t
   rocm.bindings.hipblaslt.hipblasLtMatrixLayout_t
   rocm.bindings.hipblaslt.hipblasLtMatmulPreference_t
   rocm.bindings.hipblaslt.hipblasLtMatmulAlgo_t
   rocm.bindings.hipblaslt.hipblasLtMatmulHeuristicResult_t


Classes
-------

.. autoapisummary::

   rocm.bindings.hipblaslt.hipblasLtEpilogue_t
   rocm.bindings.hipblaslt.hipblasLtBatchMode_t
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutAttribute_t
   rocm.bindings.hipblaslt.hipblasLtPointerMode_t
   rocm.bindings.hipblaslt.hipblasLtMatmulMatrixScale_t
   rocm.bindings.hipblaslt.hipblasLtStreamKTileSchedulingMode_t
   rocm.bindings.hipblaslt.hipblasLtMatmulDescAttributes_t
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceAttributes_t
   rocm.bindings.hipblaslt.hipblasLtOrder_t
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescAttributes_t
   rocm.bindings.hipblaslt.hipblasLtMatmulDescOpaque_t
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutOpaque_t
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceOpaque_t
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescOpaque_t


Functions
---------

.. autoapisummary::

   rocm.bindings.hipblaslt.has_symbol
   rocm.bindings.hipblaslt.hipblasLtGetVersion
   rocm.bindings.hipblaslt.hipblasLtGetGitRevision
   rocm.bindings.hipblaslt.hipblasLtGetArchName
   rocm.bindings.hipblaslt.hipblasLtCreate
   rocm.bindings.hipblaslt.hipblasLtDestroy
   rocm.bindings.hipblaslt.hipblasLtSetSmCountTarget
   rocm.bindings.hipblaslt.hipblasLtGetSmCountTarget
   rocm.bindings.hipblaslt.hipblasLtCheckNumericsDrain
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutCreate
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutDestroy
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutSetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatrixLayoutGetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatmulDescCreate
   rocm.bindings.hipblaslt.hipblasLtMatmulDescDestroy
   rocm.bindings.hipblaslt.hipblasLtMatmulDescSetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatmulDescGetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceCreate
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceDestroy
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceSetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatmulPreferenceGetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatmulAlgoGetHeuristic
   rocm.bindings.hipblaslt.hipblasLtMatmul
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescCreate
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescDestroy
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescSetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatrixTransformDescGetAttribute
   rocm.bindings.hipblaslt.hipblasLtMatrixTransform


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: HIPBLASLT_VERSION_MAJOR
   :type:  Any

.. py:data:: HIPBLASLT_VERSION_MINOR
   :type:  Any

.. py:data:: HIPBLASLT_VERSION_PATCH
   :type:  Any

.. py:class:: hipblasLtEpilogue_t

   Bases: :py:obj:`enum.IntEnum`


   Specifies the enumeration type to set the postprocessing options for the epilogue.
       


   .. py:attribute:: HIPBLASLT_EPILOGUE_DEFAULT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_RELU
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_BIAS
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_RELU_BIAS
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_GELU
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_GELU_BIAS
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_RELU_AUX
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_DRELU
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_DRELU_BGRAD
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_GELU_AUX
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_DGELU
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_DGELU_BGRAD
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_BGRADA
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_BGRADB
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_SIGMOID
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_SWISH_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_CLAMP_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT
      :type:  int


.. py:class:: hipblasLtBatchMode_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the batch mode of the matrices.
       


   .. py:attribute:: HIPBLASLT_BATCH_MODE_STRIDED
      :type:  int


   .. py:attribute:: HIPBLASLT_BATCH_MODE_POINTER_ARRAY
      :type:  int


.. py:class:: hipblasLtMatrixLayoutAttribute_t

   Bases: :py:obj:`enum.IntEnum`


   Specifies the attributes that define the details of the matrix.
       


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_TYPE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_ORDER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_ROWS
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_COLS
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_LD
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE
      :type:  int


.. py:class:: hipblasLtPointerMode_t

   Bases: :py:obj:`enum.IntEnum`


   Pointer mode to use for alpha.
       


   .. py:attribute:: HIPBLASLT_POINTER_MODE_HOST
      :type:  int


   .. py:attribute:: HIPBLASLT_POINTER_MODE_DEVICE
      :type:  int


   .. py:attribute:: HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST
      :type:  int


.. py:class:: hipblasLtMatmulMatrixScale_t

   Bases: :py:obj:`enum.IntEnum`


   Block scale mode for A and B.
       


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_MATRIX_SCALE_END
      :type:  int


.. py:class:: hipblasLtStreamKTileSchedulingMode_t

   Bases: :py:obj:`enum.IntEnum`


   Mode values for the ``HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT``
   attribute and the C++ ext ``GemmPreference::setStreamKTileSchedulingMode``.

   The attribute storage stays ``int32_t``; values outside ``{0, 1, 2}`` are
   rejected by the setter with ``HIPBLAS_STATUS_INVALID_VALUE``.


   .. py:attribute:: HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF
      :type:  int


   .. py:attribute:: HIPBLASLT_STREAMK_TILE_SCHEDULING_ON
      :type:  int


   .. py:attribute:: HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO
      :type:  int


.. py:class:: hipblasLtMatmulDescAttributes_t

   Bases: :py:obj:`enum.IntEnum`


   Specifies the attributes that define the specifics of the matrix multiply operation.
       


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_TRANSA
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_TRANSB
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_BIAS_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_POINTER_MODE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_A_SCALE_MODE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_B_SCALE_MODE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_DESC_MAX
      :type:  int


.. py:class:: hipblasLtMatmulPreferenceAttributes_t

   Bases: :py:obj:`enum.IntEnum`


   This is an enumerated type used to apply algorithm search preferences while fine-tuning the heuristic function.
       


   .. py:attribute:: HIPBLASLT_MATMUL_PREF_SEARCH_MODE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET
      :type:  int


   .. py:attribute:: HIPBLASLT_MATMUL_PREF_MAX
      :type:  int


.. py:class:: hipblasLtOrder_t

   Bases: :py:obj:`enum.IntEnum`


   Enumeration for data ordering.
       


   .. py:attribute:: HIPBLASLT_ORDER_COL
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_ROW
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_COL16_4R32
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_COL16_4R16
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_COL16_4R8
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_COL16_4R4
      :type:  int


   .. py:attribute:: HIPBLASLT_ORDER_COL16_4R2
      :type:  int


.. py:class:: hipblasLtMatrixTransformDescAttributes_t

   Bases: :py:obj:`enum.IntEnum`


   Matrix transform descriptor attributes to define details of the operation.
       


   .. py:attribute:: HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA
      :type:  int


   .. py:attribute:: HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB
      :type:  int


.. py:class:: hipblasLtMatmulDescOpaque_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipblasLtMatrixLayoutOpaque_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipblasLtMatmulPreferenceOpaque_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipblasLtMatrixTransformDescOpaque_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Semi-opaque descriptor for hipblasLtMatrixTransform() operation details
       


   .. py:attribute:: data
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipblasLtMatrixTransformDesc_t

.. py:data:: hipblasLtMatmulDesc_t

.. py:data:: hipblasLtMatrixLayout_t

.. py:data:: hipblasLtMatmulPreference_t

.. py:data:: hipblasLtMatmulAlgo_t

.. py:data:: hipblasLtMatmulHeuristicResult_t

.. py:function:: hipblasLtGetVersion(handle)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: (undocumented)
       * version (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle, int * version)


.. py:function:: hipblasLtGetGitRevision(handle, rev)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       rev (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle, char * rev)


.. py:function:: hipblasLtGetArchName()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: (undocumented)
       * archName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtGetArchName(char ** archName)


.. py:function:: hipblasLtCreate()

   Create a hipBLASLt handle

   This function initializes the hipBLASLt library and creates a handle to an
   opaque structure holding the hipBLASLt library context. It allocates light
   hardware resources on the host and device and must be called prior to making
   any other hipBLASLt library calls. The hipBLASLt library context is tied to
   the current ROCm device. To use the library on multiple devices, one
   hipBLASLt handle should be created for each device.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: The allocation completed successfully.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: ``handle`` == NULL.
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               Pointer to the allocated hipBLASLt handle for the created hipBLASLt
               context.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t * handle)


.. py:function:: hipblasLtDestroy(handle)

   Destroy a hipBLASLt handle.

   This function releases hardware resources used by the hipBLASLt library.
    It is usually the last call with a particular handle to the
   hipBLASLt library. Because hipblasLtCreate() allocates some internal
   resources and the release of those resources by calling hipblasLtDestroy()
   implicitly calls ``hipDeviceSynchronize``, it is recommended to minimize
   the number of hipblasLtCreate() / hipblasLtDestroy() occurrences.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the hipBLASLt handle to be destroyed.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: The hipBLASLt context was successfully
                 destroyed.
               - :py:obj:`~.HIPBLAS_STATUS_NOT_INITIALIZED`: The hipBLASLt library was
                 not initialized.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: ``handle`` == NULL.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle)


.. py:function:: hipblasLtSetSmCountTarget(handle, smCountTarget)

   Set the handle-level target compute-unit (CU / SM) count.

   The hipBLASLt analogue of cuBLAS's ``cublasSetSmCountTarget``. The value
   hints how many compute units hipBLASLt should target for kernel selection
   and persistent-grid sizing on subsequent matmul calls that use this handle.

   ``0`` (the default) means "no override; use all CUs the device exposes".
   Negative values are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. A
   per-matmul-descriptor (``HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET``) or
   per-preference (``HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET``) attribute, when
   set to a non-zero value, takes precedence over this handle-level value.

   The user must ensure thread safety when modifying handle state from
   multiple threads, the same as for any other handle-mutating helper.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           hipBLASLt library context.

       smCountTarget (:py:obj:`~.int`) -- *IN*:
           target CU/SM count; ``0`` for "use all CUs".

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: value stored.
               - :py:obj:`~.HIPBLAS_STATUS_NOT_INITIALIZED`: ``handle`` is null / uninitialized.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: ``smCountTarget`` is negative.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtSetSmCountTarget(hipblasLtHandle_t handle, int32_t smCountTarget)


.. py:function:: hipblasLtGetSmCountTarget(handle)

   Return the handle-level target compute-unit (CU / SM) count.

   Returns the value previously programmed via ``hipblasLtSetSmCountTarget``.
   Equivalent to cuBLAS's ``cublasGetSmCountTarget``.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           hipBLASLt library context.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: value returned.
               - :py:obj:`~.HIPBLAS_STATUS_NOT_INITIALIZED`: ``handle`` is null / uninitialized.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: ``smCountTarget`` is null.
       * :py:obj:`~.int`:
               receives the previously stored value (``0`` if never set).

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtGetSmCountTarget(hipblasLtHandle_t handle, int32_t * smCountTarget)


.. py:function:: hipblasLtCheckNumericsDrain(handle)

   Drain the post-GEMM check-numerics flag without destroying the handle.

   When ``HIPBLASLT_CHECK_NUMERICS`` is set, this function performs a
   device-wide synchronize, reads the persistent NaN flag, and resets it.
   The matmul ``call_id`` of the FIRST scanned NaN observed since the
   previous drain (or handle creation) is written to ``first_nan_call_id``
   if non-null. Zero means no NaN was observed in that window. Frameworks
   (e.g. PyTorch) call this to obtain a result without relying on the
   handle destructor (which may not run if the process is killed).

   When the env var is not set, the function is a no-op and returns
   ``HIPBLAS_STATUS_SUCCESS`` with ``*first_nan_call_id`` set to 0.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the allocated hipBLASLt handle.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: Drain completed (or scanning disabled).
               - :py:obj:`~.HIPBLAS_STATUS_NOT_INITIALIZED`: ``handle`` is null.
       * :py:obj:`~.int`:
               Optional. If non-null, receives the call_id of the
               first NaN seen in this drain window (0 = none).

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtCheckNumericsDrain(hipblasLtHandle_t handle, uint32_t * first_nan_call_id)


.. py:function:: hipblasLtMatrixLayoutCreate(type, rows, cols, ld)

   Create a matrix layout descriptor.

   This function creates a matrix layout descriptor by allocating the memory
   needed to hold its opaque structure.

   Args:
       type (:py:obj:`~.hipDataType`) -- *IN*:
           Enumerant that specifies the data precision for the matrix layout
           descriptor created by this function. See hipDataType.

       rows (:py:obj:`~.int`) -- *IN*:
           Number of rows of the matrix.

       cols (:py:obj:`~.int`) -- *IN*:
           Number of columns of the matrix.

       ld (:py:obj:`~.int`) -- *IN*:
           The leading dimension of the matrix. In column major layout, this is the
           number of elements to jump to reach the next column. Therefore, ld >= m (number of
           rows).

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the descriptor was created successfully.
               - :py:obj:`~.HIPBLAS_STATUS_ALLOC_FAILED`: If the memory could not be allocated.
       * :py:obj:`~.hipblasLtMatrixLayoutOpaque_t`:
               Pointer to the structure holding the matrix layout descriptor
               created by this function. See ``hipblasLtMatrixLayout_t`` .

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t * matLayout, hipDataType type, uint64_t rows, uint64_t cols, int64_t ld)


.. py:function:: hipblasLtMatrixLayoutDestroy(matLayout)

   Destroy a matrix layout descriptor

   This function destroys a previously created matrix layout descriptor object.

   Args:
       matLayout (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the structure holding the matrix layout descriptor to
           be destroyed by this function. see ``hipblasLtMatrixLayout_t`` .

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the operation was successful.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t matLayout)


.. py:function:: hipblasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes)

   Set an attribute for a matrix descriptor.

   This function sets the value of the specified attribute belonging to a
   previously created matrix descriptor.

   Args:
       matLayout (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           descriptor queried by this function. See ``hipblasLtMatrixLayout_t`` .

       attr (:py:obj:`~.hipblasLtMatrixLayoutAttribute_t`) -- *IN*:
           The attribute that will be set by this function. See \ref
           hipblasLtMatrixLayoutAttribute_t.

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The value to which the specified attribute should be set.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the buf buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute was set successfully.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If ``buf`` is NULL or ``sizeInBytes``
                 doesn't match the size of the internal storage for the selected attribute.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, const void * buf, size_t sizeInBytes)


.. py:function:: hipblasLtMatrixLayoutGetAttribute(matLayout, attr, buf, sizeInBytes)

   Query an attribute from a matrix descriptor.

   This function returns the value of the queried attribute belonging to a
   previously created matrix descriptor.

   Args:
       matLayout (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           descriptor queried by this function. See ``hipblasLtMatrixLayout_t`` .

       attr (:py:obj:`~.hipblasLtMatrixLayoutAttribute_t`) -- *IN*:
           The attribute that will be retrieved by this function. See
           ``hipblasLtMatrixLayoutAttribute_t`` .

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Memory address containing the attribute value retrieved by this
           function.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the ``buf`` buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute's value was successfully
                 written to user memory.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If \p
                 sizeInBytes is 0 and ``sizeWritten`` is NULL, or if ``sizeInBytes`` is non-zero
                 and ``buf`` is NULL, or ``sizeInBytes`` doesn't match the size of the internal storage
                 for the selected attribute.
       * :py:obj:`~.int`:
               Valid only when the return value is HIPBLAS_STATUS_SUCCESS. If
               sizeInBytes is non-zero, then sizeWritten is the number of bytes actually
               written. If sizeInBytes is 0, then sizeWritten is the number of bytes needed
               to write the full contents.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, void * buf, size_t sizeInBytes, size_t * sizeWritten)


.. py:function:: hipblasLtMatmulDescCreate(computeType, scaleType)

   Create a matrix multiply descriptor.

   This function creates a matrix multiply descriptor by allocating the memory
   needed to hold its opaque structure.

   Args:
       computeType (:py:obj:`~.hipblasComputeType_t`) -- *IN*:
           Enumerant that specifies the data precision for the matrix
           multiply descriptor this function creates. See hipblasComputeType_t.

       scaleType (:py:obj:`~.hipDataType`) -- *IN*:
           Enumerant that specifies the data precision for the matrix
           transform descriptor this function creates. See hipDataType.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the descriptor was created successfully.
               - :py:obj:`~.HIPBLAS_STATUS_ALLOC_FAILED`: If the memory could not be allocated.
       * :py:obj:`~.hipblasLtMatmulDescOpaque_t`:
               Pointer to the structure holding the matrix multiply descriptor
               created by this function. See ``hipblasLtMatmulDesc_t`` .

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t * matmulDesc, hipblasComputeType_t computeType, hipDataType scaleType)


.. py:function:: hipblasLtMatmulDescDestroy(matmulDesc)

   Destroy a matrix multiply descriptor.

   This function destroys a previously created matrix multiply descriptor
   object.

   Args:
       matmulDesc (:py:obj:`~.hipblasLtMatmulDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the structure holding the matrix multiply descriptor
           to be destroyed by this function. See ``hipblasLtMatmulDesc_t`` .

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If operation was successful.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t matmulDesc)


.. py:function:: hipblasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes)

   Set attribute to a matrix multiply descriptor.

   This function sets the value of the specified attribute belonging to a
   previously created matrix multiply descriptor.

   Args:
       matmulDesc (:py:obj:`~.hipblasLtMatmulDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           multiply descriptor queried by this function. See ``hipblasLtMatmulDesc_t`` .

       attr (:py:obj:`~.hipblasLtMatmulDescAttributes_t`) -- *IN*:
           The attribute that will be set by this function. See \ref
           hipblasLtMatmulDescAttributes_t.

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The value to which the specified attribute should be set.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the buf buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute was set successfully.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If ``buf`` is NULL or ``sizeInBytes``
                 doesn't match the size of the internal storage for the selected attribute.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, const void * buf, size_t sizeInBytes)


.. py:function:: hipblasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes)

   Query attribute from a matrix multiply descriptor.

   This function returns the value of the queried attribute belonging to a
   previously created matrix multiply descriptor.

   Args:
       matmulDesc (:py:obj:`~.hipblasLtMatmulDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           multiply descriptor queried by this function. See ``hipblasLtMatmulDesc_t`` .

       attr (:py:obj:`~.hipblasLtMatmulDescAttributes_t`) -- *IN*:
           The attribute that will be retrieved by this function. See
           ``hipblasLtMatmulDescAttributes_t`` .

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Memory address containing the attribute value retrieved by this
           function.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the ``buf`` buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute's value was successfully
                 written to user memory.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If \p
                 sizeInBytes is 0 and ``sizeWritten`` is NULL, or if ``sizeInBytes`` is non-zero
                 and ``buf`` is NULL, or ``sizeInBytes`` doesn't match the size of the internal storage
                 for the selected attribute.
       * :py:obj:`~.int`:
               Valid only when the return value is HIPBLAS_STATUS_SUCCESS. If
               sizeInBytes is non-zero, then sizeWritten is the number of bytes actually
               written. If sizeInBytes is 0, then sizeWritten is the number of bytes needed
               to write the full contents.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, void * buf, size_t sizeInBytes, size_t * sizeWritten)


.. py:function:: hipblasLtMatmulPreferenceCreate()

   Create a preference descriptor.

   This function creates a matrix multiply heuristic search preferences
   descriptor by allocating the memory needed to hold its opaque structure.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the descriptor was created
                 successfully.
               - :py:obj:`~.HIPBLAS_STATUS_ALLOC_FAILED`: If memory could not be
                 allocated.
       * :py:obj:`~.hipblasLtMatmulPreferenceOpaque_t`:
               Pointer to the structure holding the matrix multiply preferences
               descriptor created by this function. see ``hipblasLtMatmulPreference_t`` .

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t * pref)


.. py:function:: hipblasLtMatmulPreferenceDestroy(pref)

   Destroy a preference descriptor.

   This function destroys a previously created matrix multiply preferences
   descriptor object.

   Args:
       pref (:py:obj:`~.hipblasLtMatmulPreferenceOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the structure holding the matrix multiply preferences
           descriptor to be destroyed by this function. See \ref
           hipblasLtMatmulPreference_t.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If operation was successful.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref)


.. py:function:: hipblasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes)

   Set attribute in a preference descriptor.

   This function sets the value of the specified attribute belonging to a
   previously created matrix multiply preferences descriptor.

   Args:
       pref (:py:obj:`~.hipblasLtMatmulPreferenceOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           multiply preferences descriptor queried by this function. See \ref
           hipblasLtMatmulPreference_t.

       attr (:py:obj:`~.hipblasLtMatmulPreferenceAttributes_t`) -- *IN*:
           The attribute that will be set by this function. See \ref
           hipblasLtMatmulPreferenceAttributes_t.

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The value to which the specified attribute should be set.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the ``buf`` buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute was set successfully.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If ``buf`` is NULL or ``sizeInBytes``
                 doesn't match the size of the internal storage for the selected attribute.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, const void * buf, size_t sizeInBytes)


.. py:function:: hipblasLtMatmulPreferenceGetAttribute(pref, attr, buf, sizeInBytes)

   Query attribute from a preference descriptor.

   This function returns the value of the queried attribute belonging to a
   previously created matrix multiply heuristic search preferences descriptor.

   Args:
       pref (:py:obj:`~.hipblasLtMatmulPreferenceOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the previously created structure holding the matrix
           multiply heuristic search preferences descriptor queried by this function.
           See ``hipblasLtMatmulPreference_t`` .

       attr (:py:obj:`~.hipblasLtMatmulPreferenceAttributes_t`) -- *IN*:
           The attribute that will be retrieved by this function. See
           ``hipblasLtMatmulPreferenceAttributes_t`` .

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Memory address containing the attribute value retrieved by this
           function.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the ``buf`` buffer (in bytes) for verification.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute's value was successfully
                 written to user memory.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If \p
                 sizeInBytes is 0 and ``sizeWritten`` is NULL, or if ``sizeInBytes`` is non-zero
                 and ``buf`` is NULL, or ``sizeInBytes`` doesn't match the size of the internal storage
                 for the selected attribute.
       * :py:obj:`~.int`:
               Valid only when the return value is HIPBLAS_STATUS_SUCCESS. If
               sizeInBytes is non-zero, then sizeWritten is the number of bytes actually
               written. If sizeInBytes is 0, then sizeWritten is the number of bytes needed
               to write the full contents.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, void * buf, size_t sizeInBytes, size_t * sizeWritten)


.. py:function:: hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, requestedAlgoCount, heuristicResultsArray)

   Retrieve the possible algorithms.

   This function retrieves the possible algorithms for the matrix multiply
   operation hipblasLtMatmul() with the given input matrices A, B, and
   C, and the output matrix D. The output is placed in ``heuristicResultsArray``
   in order of increasing estimated compute time. Note that the wall duration
   increases if the ``requestedAlgoCount`` increases.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the allocated hipBLASLt handle for the
           hipBLASLt context. See ``hipblasLtHandle_t`` .

       matmulDesc (:py:obj:`~.hipblasLtMatmulDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handle to a previously created matrix multiplication
           descriptor of type ``hipblasLtMatmulDesc_t`` .

       Adesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       Bdesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       Cdesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       Ddesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       pref (:py:obj:`~.hipblasLtMatmulPreferenceOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the structure holding the heuristic
           search preferences descriptor. See ``hipblasLtMatmulPreference_t`` .

       requestedAlgoCount (:py:obj:`~.int`) -- *IN*:
           Size of the ``heuristicResultsArray`` (in elements).
           This is the requested maximum number of algorithms to return.

       heuristicResultsArray (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           [] Array containing the algorithm heuristics and
           associated runtime characteristics returned by this function, in order
           of increasing estimated compute time.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If query was successful. Inspect
                 ``heuristicResultsArray[0 to (returnAlgoCount -1)].state`` for the status of the
                 results.
               - :py:obj:`~.HIPBLAS_STATUS_NOT_SUPPORTED`: If no heuristic function is
                 available for current configuration.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If
                 ``requestedAlgoCount`` is less than or equal to zero.
       * :py:obj:`~.int`:
               Number of algorithms returned by this function. This
               is the number of ``heuristicResultsArray`` elements written.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc, hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc, hipblasLtMatmulPreference_t pref, int requestedAlgoCount, hipblasLtMatmulHeuristicResult_t[] heuristicResultsArray, int * returnAlgoCount)


.. py:function:: hipblasLtMatmul(handle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream)

   Compute a matrix multiplication on the described inputs.

   This function computes the matrix multiplication of matrices A and B to
   produce the output matrix D, according to the following operation: ``D`` = \p
   alpha*( ``A`` *``B)`` + ``beta*(`` ``C`` ), where ``A,`` ``B,`` and ``C`` are input
   matrices, and ``alpha`` and ``beta`` are input scalars. Note: This function
   supports both in-place matrix multiplication (``C == D`` and ``Cdesc == Ddesc``) and
   out-of-place matrix multiplication (``C != D``, both matrices must have the same
   data type, number of rows, number of columns, batch size, and memory order).
   In the out-of-place case, the leading dimension of ``C`` can be different from
   the leading dimension of ``D``. Specifically, the leading dimension of ``C`` can be 0
   to achieve row or column broadcast. If ``Cdesc`` is omitted, this function
   assumes it to be equal to ``Ddesc``.

   Args:
       handle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the allocated hipBLASLt handle for the
           hipBLASLt context. See ``hipblasLtHandle_t`` .

       matmulDesc (:py:obj:`~.hipblasLtMatmulDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handle to a previously created matrix multiplication
           descriptor of type ``hipblasLtMatmulDesc_t`` .

       alpha (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointers to the scalars used in the multiplication.

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointers to the GPU memory associated with the
           corresponding descriptors ``Adesc,`` ``Bdesc,`` and ``Cdesc.``

       Adesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointers to the GPU memory associated with the
           corresponding descriptors ``Adesc,`` ``Bdesc,`` and ``Cdesc.``

       Bdesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       beta (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointers to the scalars used in the multiplication.

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointers to the GPU memory associated with the
           corresponding descriptors ``Adesc,`` ``Bdesc,`` and ``Cdesc.``

       Cdesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to the GPU memory associated with the
           descriptor ``Ddesc.``

       Ddesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Handles to the previously created matrix layout
           descriptors of the type ``hipblasLtMatrixLayout_t`` .

       algo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Handle for matrix multiplication algorithm to be
           used. See ``hipblasLtMatmulAlgo_t`` . When NULL, an implicit heuristics query
           with default search preferences will be performed to determine the actual
           algorithm to use.

       workspace (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the workspace buffer allocated in the GPU
           memory. Pointer must be 16B aligned (that is, the lowest 4 bits of the address must
           be 0).

       workspaceSizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the workspace.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The HIP stream where all GPU work is
           submitted.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the operation completed
                 successfully.
               - :py:obj:`~.HIPBLAS_STATUS_EXECUTION_FAILED`: If HIP reported an
                 execution error from the device.
               - :py:obj:`~.HIPBLAS_STATUS_ARCH_MISMATCH`: If
                 the configured operation cannot be run using the selected device. \retval
                 HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
                 selected device doesn't support the configured operation. \retval
                 HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
                 conflict, or in an impossible configuration. For example, when
                 workspaceSizeInBytes is less than the workspace required by the configured algorithm.
               - :py:obj:`~.HIBLAS_STATUS_NOT_INITIALIZED`: If the hipBLASLt handle has not been
                 initialized.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, const void * alpha, const void * A, hipblasLtMatrixLayout_t Adesc, const void * B, hipblasLtMatrixLayout_t Bdesc, const void * beta, const void * C, hipblasLtMatrixLayout_t Cdesc, void * D, hipblasLtMatrixLayout_t Ddesc, const hipblasLtMatmulAlgo_t * algo, void * workspace, size_t workspaceSizeInBytes, hipStream_t stream)


.. py:function:: hipblasLtMatrixTransformDescCreate(scaleType)

   Create a new matrix transform operation descriptor.

   Args:
       scaleType (:py:obj:`~.hipDataType`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_ALLOC_FAILED`: If memory could not be allocated.
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the descriptor was created successfully.
       * transformDesc (:py:obj:`~.hipblasLtMatrixTransformDescOpaque_t`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixTransformDescCreate(hipblasLtMatrixTransformDesc_t * transformDesc, hipDataType scaleType)


.. py:function:: hipblasLtMatrixTransformDescDestroy(transformDesc)

   Destroy a matrix transform operation descriptor.

   Args:
       transformDesc (:py:obj:`~.hipblasLtMatrixTransformDescOpaque_t`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the operation was successful.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc)


.. py:function:: hipblasLtMatrixTransformDescSetAttribute(transformDesc, attr, buf, sizeInBytes)

   Set a matrix transform operation descriptor attribute.

   Args:
       transformDesc (:py:obj:`~.hipblasLtMatrixTransformDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           The descriptor.

       attr (:py:obj:`~.hipblasLtMatrixTransformDescAttributes_t`) -- *IN*:
           The attribute.

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Memory address containing the new value.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the buf buffer for verification (in bytes).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If buf is NULL or sizeInBytes doesn't match the size of the internal storage for
                 the selected attribute.
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute was set successfully.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixTransformDescSetAttribute(hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, const void * buf, size_t sizeInBytes)


.. py:function:: hipblasLtMatrixTransformDescGetAttribute(transformDesc, attr, buf, sizeInBytes)

   Gets the matrix transform attribute.

   Gets the attribute from the matrix transform operation descriptor.

   Args:
       transformDesc (:py:obj:`~.hipblasLtMatrixTransformDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           The descriptor.

       attr (:py:obj:`~.hipblasLtMatrixTransformDescAttributes_t`) -- *IN*:
           The attribute.

       buf (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Memory address containing the new value.

       sizeInBytes (:py:obj:`~.int`) -- *IN*:
           Size of the buf buffer for verification (in bytes).

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If sizeInBytes is 0 and sizeWritten is NULL, or sizeInBytes is non-zero
                 and buf is NULL, or sizeInBytes doesn't match the size of the internal storage for
                 the selected attribute.
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the attribute's value was successfully written to user memory.
       * :py:obj:`~.int`:
               Only valid when return value is HIPBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero, the number
               of bytes actually written. If sizeInBytes is 0, the number of bytes needed to write the full contents.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, void * buf, size_t sizeInBytes, size_t * sizeWritten)


.. py:function:: hipblasLtMatrixTransform(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, stream)

   Matrix layout conversion helper.

   The matrix layout conversion helper (``C = alpha * op(A) + beta * op(B)``),
   can be used to change the memory order of the data or to scale and shift the values.

   Args:
       lightHandle (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the allocated hipBLASLt handle for the
           hipBLASLt context. See ``hipblasLtHandle_t`` .

       transformDesc (:py:obj:`~.hipblasLtMatrixTransformDescOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the allocated matrix transform descriptor.

       alpha (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to scalar alpha. Pointer to either the host or device address.

       A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to matrix A. Must be a pointer to the device address.

       Adesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the layout for input matrix A.

       beta (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to scalar beta. Pointer to either the host or device address.

       B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the layout for matrix B. Must be a pointer to the device address.

       Bdesc (:py:obj:`~.hipblasLtMatrixLayoutOpaque_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the layout for input matrix B.

       C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to matrix C. Must be a pointer to the device address.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The HIP stream where all the GPU work will be submitted.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipblasStatus_t`: One of:
               - :py:obj:`~.HIPBLAS_STATUS_NOT_INITIALIZED`: If the hipBLASLt handle has not been initialized.
               - :py:obj:`~.HIPBLAS_STATUS_INVALID_VALUE`: If the parameters are in conflict or in an impossible configuration, for example,
                 when A is not NULL but Adesc is NULL.
               - :py:obj:`~.HIPBLAS_STATUS_NOT_SUPPORTED`: If the current implementation on the selected device doesn't support the configured
                 operation.
               - :py:obj:`~.HIPBLAS_STATUS_ARCH_MISMATCH`: If the configured operation cannot be run using the selected device.
               - :py:obj:`~.HIPBLAS_STATUS_EXECUTION_FAILED`: If HIP reported an execution error from the device.
               - :py:obj:`~.HIPBLAS_STATUS_SUCCESS`: If the operation completed successfully.
       * :py:obj:`~.hipblasLtMatrixLayoutOpaque_t`:
               Pointer to the layout for output matrix C.

   .. rubric:: C signature

   .. code-block:: c

       hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t lightHandle, hipblasLtMatrixTransformDesc_t transformDesc, const void * alpha, const void * A, hipblasLtMatrixLayout_t Adesc, const void * beta, const void * B, hipblasLtMatrixLayout_t Bdesc, void * C, hipblasLtMatrixLayout_t Cdesc, hipStream_t stream)


