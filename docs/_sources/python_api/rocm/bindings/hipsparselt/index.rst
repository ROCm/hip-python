rocm.bindings.hipsparselt
=========================

.. py:module:: rocm.bindings.hipsparselt


Classes
-------

.. autoapisummary::

   rocm.bindings.hipsparselt.hipsparseLtHandle_t
   rocm.bindings.hipsparselt.hipsparseLtMatDescriptor_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulDescriptor_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgSelection_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulPlan_t
   rocm.bindings.hipsparselt.hipsparseLtSparsity_t
   rocm.bindings.hipsparselt.hipsparseLtMatDescAttribute_t
   rocm.bindings.hipsparselt.hipsparseLtComputetype_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulDescAttribute_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlg_t
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgAttribute_t
   rocm.bindings.hipsparselt.hipsparseLtPruneAlg_t
   rocm.bindings.hipsparselt.hipsparseLtSplitKMode_t


Functions
---------

.. autoapisummary::

   rocm.bindings.hipsparselt.has_symbol
   rocm.bindings.hipsparselt.hipsparseLtInitialize
   rocm.bindings.hipsparselt.hipsparseLtGetVersion
   rocm.bindings.hipsparselt.hipsparseLtGetProperty
   rocm.bindings.hipsparselt.hipsparseLtGetGitRevision
   rocm.bindings.hipsparselt.hipsparseLtGetArchName
   rocm.bindings.hipsparselt.hipsparseLtInit
   rocm.bindings.hipsparselt.hipsparseLtDestroy
   rocm.bindings.hipsparselt.hipsparseLtDenseDescriptorInit
   rocm.bindings.hipsparselt.hipsparseLtStructuredDescriptorInit
   rocm.bindings.hipsparselt.hipsparseLtMatDescriptorDestroy
   rocm.bindings.hipsparselt.hipsparseLtMatDescSetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatDescGetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatmulDescriptorInit
   rocm.bindings.hipsparselt.hipsparseLtMatmulDescSetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatmulDescGetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgSelectionInit
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgSelectionDestroy
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgSetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatmulAlgGetAttribute
   rocm.bindings.hipsparselt.hipsparseLtMatmulGetWorkspace
   rocm.bindings.hipsparselt.hipsparseLtMatmulPlanInit
   rocm.bindings.hipsparselt.hipsparseLtMatmulPlanDestroy
   rocm.bindings.hipsparselt.hipsparseLtMatmul
   rocm.bindings.hipsparselt.hipsparseLtMatmulSearch
   rocm.bindings.hipsparselt.hipsparseLtSpMMAPrune
   rocm.bindings.hipsparselt.hipsparseLtSpMMAPruneCheck
   rocm.bindings.hipsparselt.hipsparseLtSpMMAPrune2
   rocm.bindings.hipsparselt.hipsparseLtSpMMAPruneCheck2
   rocm.bindings.hipsparselt.hipsparseLtSpMMACompressedSize
   rocm.bindings.hipsparselt.hipsparseLtSpMMACompress
   rocm.bindings.hipsparselt.hipsparseLtSpMMACompressedSize2
   rocm.bindings.hipsparselt.hipsparseLtSpMMACompress2


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: hipsparseLtHandle_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Handle to the hipSPARSELt library context queue.

   The hipSPARSELt handle is a structure holding the hipSPARSELt library context. It must
   be initialized using ``hipsparseLtInit`` and the returned handle must be
   passed to all subsequent library function calls. It should be destroyed at the end
   using ``hipsparseLtDestroy`` .


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


.. py:class:: hipsparseLtMatDescriptor_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Descriptor of the matrix.

   The hipSPARSELt matrix descriptor is a structure holding all properties of a matrix.
   It must be initialized using ``hipsparseLtDenseDescriptorInit`` and the returned
   descriptor must be passed to all subsequent library calls that involve the matrix.
   It should be destroyed at the end using ``hipsparseLtMatDescriptorDestroy`` .


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


.. py:class:: hipsparseLtMatmulDescriptor_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Descriptor of the matrix multiplication operation.

   The hipSPARSELt matrix multiplication descriptor is a structure holding
   the description of the matrix multiplication operation.
   It is initialized with the ``hipsparseLtMatmulDescriptorInit`` function.


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


.. py:class:: hipsparseLtMatmulAlgSelection_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Descriptor of the matrix multiplication algorithm.

   It is initialized with the ``hipsparseLtMatmulAlgSelectionInit`` function.


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


.. py:class:: hipsparseLtMatmulPlan_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Descriptor of the matrix multiplication execution plan

   The hipSPARSELt matrix multiplication execution plan descriptor is a structure holding
   all the information necessary to execute the ``hipsparseLtMatmul`` operation.
   It is initialized and destroyed using the ``hipsparseLtMatmulPlanInit`` 
   and ``hipsparseLtMatmulPlanDestroy`` functions, respectively.


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


.. py:class:: hipsparseLtSparsity_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the sparsity of the structured matrix.

   The enumerator specifies the sparsity ratio of the structured matrix as
   sparsity = nnz / total elements.
   The sparsity property is used in the ``hipsparseLtStructuredDescriptorInit`` function.


   .. py:attribute:: HIPSPARSELT_SPARSITY_50_PERCENT
      :type:  int


.. py:class:: hipsparseLtMatDescAttribute_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the additional attributes of a matrix descriptor.

   The  ``hipsparseLtMatDescAttribute_t`` enumeration is used in the
   ``hipsparseLtMatDescSetAttribute`` and ``hipsparseLtMatDescGetAttribute`` functions.


   .. py:attribute:: HIPSPARSELT_MAT_NUM_BATCHES
      :type:  int


   .. py:attribute:: HIPSPARSELT_MAT_BATCH_STRIDE
      :type:  int


.. py:class:: hipsparseLtComputetype_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the compute precision modes of the matrix.
       


   .. py:attribute:: HIPSPARSELT_COMPUTE_16F
      :type:  int


   .. py:attribute:: HIPSPARSELT_COMPUTE_32I
      :type:  int


   .. py:attribute:: HIPSPARSELT_COMPUTE_32F
      :type:  int


   .. py:attribute:: HIPSPARSELT_COMPUTE_TF32
      :type:  int


   .. py:attribute:: HIPSPARSELT_COMPUTE_TF32_FAST
      :type:  int


.. py:class:: hipsparseLtMatmulDescAttribute_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the additional attributes of a matrix multiplication descriptor.

   The  ``hipsparseLtMatmulDescAttribute_t`` enumeration is used in the
   ``hipsparseLtMatmulDescSetAttribute`` and ``hipsparseLtMatmulDescGetAttribute`` functions.


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_RELU
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_GELU
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_BIAS_STRIDE
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_BIAS_POINTER
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_ABS
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_TANH
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_BIAS_TYPE
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_SPARSE_MAT_POINTER
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_GATE_RESIDUAL_MAT_POINTER
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_GATE_RESIDUAL_DESC
      :type:  int


.. py:class:: hipsparseLtMatmulAlg_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the algorithm for matrix-matrix multiplication.

   The ``hipsparseLtMatmulAlg_t`` enumeration is used in the ``hipsparseLtMatmulAlgSelectionInit`` function.


   .. py:attribute:: HIPSPARSELT_MATMUL_ALG_DEFAULT
      :type:  int


.. py:class:: hipsparseLtMatmulAlgAttribute_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the matrix multiplication algorithm attributes.

   The ``hipsparseLtMatmulAlgAttribute_t`` enumeration is used in the
   ``hipsparseLtMatmulAlgGetAttribute`` and ``hipsparseLtMatmulAlgSetAttribute`` functions.


   .. py:attribute:: HIPSPARSELT_MATMUL_ALG_CONFIG_ID
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_SEARCH_ITERATIONS
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_SPLIT_K
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_SPLIT_K_MODE
      :type:  int


   .. py:attribute:: HIPSPARSELT_MATMUL_SPLIT_K_BUFFERS
      :type:  int


.. py:class:: hipsparseLtPruneAlg_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the pruning algorithm to apply to the structured matrix before the compression.

   The ``hipsparseLtPruneAlg_t`` enumeration is used in the ``hipsparseLtSpMMAPrune`` and ``hipsparseLtSpMMAPrune2`` functions.


   .. py:attribute:: HIPSPARSELT_PRUNE_SPMMA_TILE
      :type:  int


   .. py:attribute:: HIPSPARSELT_PRUNE_SPMMA_STRIP
      :type:  int


.. py:class:: hipsparseLtSplitKMode_t

   Bases: :py:obj:`enum.IntEnum`


   Specify the Split-K mode value.

   The ``hipsparseLtSplitKMode_t`` enumeration is used by the `HIPSPARSELT_MATMUL_SPLIT_K_MODE` attribute in ``hipsparseLtMatmulAlgAttribute_t`` .


   .. py:attribute:: HIPSPARSELT_SPLIT_K_MODE_ONE_KERNEL
      :type:  int


   .. py:attribute:: HIPSPARSELT_SPLIT_K_MODE_TWO_KERNELS
      :type:  int


.. py:function:: hipsparseLtInitialize()

   Initialize hipSPARSELt for the current HIP device.

   ``hipsparseLtInitialize`` Initialize hipSPARSELt for the current HIP device to avoid costly startup time at the first call on that device.
   This function is only supported by the HIP backend.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`:
               Always returns `~.hipsparseStatus_t.HIPSPARSE_STATUS_SUCCESS`.

   .. rubric:: C signature

   .. code-block:: c

       void hipsparseLtInitialize()


.. py:function:: hipsparseLtGetVersion(handle)

   Retrieve the version number of the hipSPARSELt library.

   ``hipsparseLtGetVersion`` returns the version number of the hipSPARSELt library.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: the ``handle`` is invalid.
       * :py:obj:`~.int`:
               the version number of the library.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtGetVersion(const hipsparseLtHandle_t * handle, int * version)


.. py:function:: hipsparseLtGetProperty(propertyType)

   Retrieve the value of the requested property.

   ``hipsparseLtGetProperty`` returns the value of the requested property.

   Args:
       propertyType (:py:obj:`~.hipLibraryPropertyType`) -- *IN*:
           hipLibraryPropertyType property type (as defined in library_types.h).

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`
       * :py:obj:`~.int`:
               value of the requested property.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtGetProperty(hipLibraryPropertyType propertyType, int * value)


.. py:function:: hipsparseLtGetGitRevision(handle, rev)

   (No short description, might be part of a group.)

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`):
           (undocumented)

       rev (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t handle, char * rev)


.. py:function:: hipsparseLtGetArchName()

   (No short description, might be part of a group.)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: (undocumented)
       * archName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtGetArchName(char ** archName)


.. py:function:: hipsparseLtInit(handle)

   Create a hipSPARSELt handle

   ``hipsparseLtInit`` creates the hipSPARSELt library context. It must be
   initialized before any other hipSPARSELt API function is invoked and must be passed to
   all subsequent library function calls. The handle should be destroyed at the end
   using ``hipsparseLtDestroy_handle()``.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *OUT*:
           hipsparselt library handle.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the initialization succeeded.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: the ``handle`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t * handle)


.. py:function:: hipsparseLtDestroy(handle)

   Destroy a hipSPARSELt handle.

   ``hipsparseLtDestroy`` destroys the hipSPARSELt library context and releases all
   resources used by the hipSPARSELt library.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_INITIALIZED`: the ``handle`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t * handle)


.. py:function:: hipsparseLtDenseDescriptorInit(handle, matDescr, rows, cols, ld, alignment, valueType, order)

   Create a descriptor for a dense matrix

   ``hipsparseLtDenseDescriptorInit`` creates and initializes a matrix descriptor.
   It should be destroyed at the end using ``hipsparseLtMatDescriptorDestroy`` ().

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *OUT*:
           the pointer to the dense matrix descriptor.

       rows (:py:obj:`~.int`) -- *IN*:
           number of rows.

       cols (:py:obj:`~.int`) -- *IN*:
           number of columns.

       ld (:py:obj:`~.int`) -- *IN*:
           leading dimension.

       alignment (:py:obj:`~.int`) -- *IN*:
           memory alignment in bytes (not used by the HIP backend).

       valueType (:py:obj:`~.hipDataType`) -- *IN*:
           data type of the matrix. Data type: hipDataType.

       order (:py:obj:`~.hipsparseOrder_t`) -- *IN*:
           memory layout: ``HIPSPARSE_ORDER_COL`` or ``HIPSPARSE_ORDER_ROW.``

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``descr,`` ``rows,`` ``cols,`` or ``ld`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``rows,`` ``cols,`` ``ld,`` ``alignment,`` ``valueType,`` or ``order`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t * handle, hipsparseLtMatDescriptor_t * matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, hipDataType valueType, hipsparseOrder_t order)


.. py:function:: hipsparseLtStructuredDescriptorInit(handle, matDescr, rows, cols, ld, alignment, valueType, order, sparsity)

   Create a descriptor for a structured matrix.

   ``hipsparseLtStructuredDescriptorInit`` creates and initializes a matrix descriptor.
   It should be destroyed at the end using ``hipsparseLtMatDescriptorDestroy`` ().

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *OUT*:
           the pointer to the dense matrix descriptor.

       rows (:py:obj:`~.int`) -- *IN*:
           number of rows.

       cols (:py:obj:`~.int`) -- *IN*:
           number of columns.

       ld (:py:obj:`~.int`) -- *IN*:
           leading dimension.

       alignment (:py:obj:`~.int`) -- *IN*:
           memory alignment in bytes (not used by the HIP backend).

       valueType (:py:obj:`~.hipDataType`) -- *IN*:
           data type of the matrix. Data type: hipDataType.

       order (:py:obj:`~.hipsparseOrder_t`) -- *IN*:
           memory layout: ``HIPSPARSE_ORDER_COL`` or ``HIPSPARSE_ORDER_ROW.``

       sparsity (:py:obj:`~.hipsparseLtSparsity_t`) -- *IN*:
           matrix sparsity ratio. See ``hipsparseLtSparsity_t`` .

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``descr,`` ``rows,`` ``cols,``  or ``ld``  is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``rows,`` ``cols,`` ``ld,`` ``alignment,`` ``valueType,`` or ``order`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t * handle, hipsparseLtMatDescriptor_t * matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, hipDataType valueType, hipsparseOrder_t order, hipsparseLtSparsity_t sparsity)


.. py:function:: hipsparseLtMatDescriptorDestroy(matDescr)

   Destroy a matrix descriptor.

   ``hipsparseLtMatDescriptorDestroy`` destroys a matrix descriptor and releases all
   resources used by the descriptor.

   Args:
       matDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``descr`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t * matDescr)


.. py:function:: hipsparseLtMatDescSetAttribute(handle, matDescr, matAttribute, data, dataSize)

   Specify the matrix attribute of a matrix descriptor.

   ``hipsparseLtMatDescSetAttribute`` sets the value of the specified attribute belonging
   to a matrix descriptor, such as number of batches and their stride.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *INOUT*:
           the matrix descriptor.

       matAttribute (:py:obj:`~.hipsparseLtMatDescAttribute_t`) -- *IN*:
           ``HIPSPARSELT_MAT_NUM_BATCHES`` or ``HIPSPARSELT_MAT_BATCH_STRIDE`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the value to which the specified attribute will be set.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matmulDescr,`` ``data,`` or ``dataSize`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t * handle, hipsparseLtMatDescriptor_t * matDescr, hipsparseLtMatDescAttribute_t matAttribute, const void * data, size_t dataSize)


.. py:function:: hipsparseLtMatDescGetAttribute(handle, matDescr, matAttribute, data, dataSize)

   Get the matrix type of a matrix descriptor.

   ``hipsparseLtMatDescGetAttribute`` returns the matrix attribute of a matrix descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor.

       matAttribute (:py:obj:`~.hipsparseLtMatDescAttribute_t`) -- *IN*:
           ``HIPSPARSELT_MAT_NUM_BATCHES`` or ``HIPSPARSELT_MAT_BATCH_STRIDE`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           the memory address containing the attribute value retrieved by this function.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``descr,`` ``data,`` or ``dataSize`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t * handle, const hipsparseLtMatDescriptor_t * matDescr, hipsparseLtMatDescAttribute_t matAttribute, void * data, size_t dataSize)


.. py:function:: hipsparseLtMatmulDescriptorInit(handle, matmulDescr, opA, opB, matA, matB, matC, matD, computeType)

   Initializes the matrix multiplication descriptor.

   ``hipsparseLtMatmulDescriptorInit`` creates a matrix multiplication descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *INOUT*:
           the matrix multiplication descriptor.

       opA (:py:obj:`~.hipsparseOperation_t`) -- *IN*:
           hipsparse operation for Matrix A: ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` or ``HIPSPARSE_OPERATION_TRANSPOSE.``

       opB (:py:obj:`~.hipsparseOperation_t`) -- *IN*:
           hipsparse operation for Matrix B: ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` or ``HIPSPARSE_OPERATION_TRANSPOSE.``

       matA (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor (one and only one of matA or matB is a structured sparsity matrix).

       matB (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor (one and only one of matA or matB is a structured sparsity matrix).

       matC (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor (dense matrix).

       matD (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix descriptor (dense matrix).

       computeType (:py:obj:`~.hipsparseLtComputetype_t`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matmulDescr,`` ``opA,`` ``opB,`` ``matA,`` ``matB,`` ``matC,`` ``matD,`` or ``computeType`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``opA,`` ``opB,`` or ``computeType`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t * handle, hipsparseLtMatmulDescriptor_t * matmulDescr, hipsparseOperation_t opA, hipsparseOperation_t opB, const hipsparseLtMatDescriptor_t * matA, const hipsparseLtMatDescriptor_t * matB, const hipsparseLtMatDescriptor_t * matC, const hipsparseLtMatDescriptor_t * matD, hipsparseLtComputetype_t computeType)


.. py:function:: hipsparseLtMatmulDescSetAttribute(handle, matmulDescr, matmulAttribute, data, dataSize)

   Specify the matrix attribute of a matrix descriptor.

   ``hipsparseLtMatmulDescSetAttribute`` sets the value of the specified attribute belonging
   to a matrix descriptor, such as number of batches and their stride.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *INOUT*:
           the matrix multiplication descriptor.

       matmulAttribute (:py:obj:`~.hipsparseLtMatmulDescAttribute_t`) -- *IN*:
           see ``hipsparseLtMatmulDescAttribute_t`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the value to which the specified attribute will be set.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matDescr,`` ``data,`` or ``dataSize`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``hipsparseLtMatmulDescAttribute_t`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t * handle, hipsparseLtMatmulDescriptor_t * matmulDescr, hipsparseLtMatmulDescAttribute_t matmulAttribute, const void * data, size_t dataSize)


.. py:function:: hipsparseLtMatmulDescGetAttribute(handle, matmulDescr, matmulAttribute, data, dataSize)

   Get the matrix type of a matrix descriptor.

   ``hipsparseLtMatmulDescGetAttribute`` returns the matrix attribute of a matrix descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix multiplication descriptor.

       matmulAttribute (:py:obj:`~.hipsparseLtMatmulDescAttribute_t`) -- *IN*:
           see ``hipsparseLtMatmulDescAttribute_t`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           the memory address containing the attribute value retrieved by this function.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matDescr,`` ``data,`` or ``dataSize`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``hipsparseLtMatmulDescAttribute_t`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulDescriptor_t * matmulDescr, hipsparseLtMatmulDescAttribute_t matmulAttribute, void * data, size_t dataSize)


.. py:function:: hipsparseLtMatmulAlgSelectionInit(handle, algSelection, matmulDescr, alg)

   Initializes the algorithm selection descriptor.

   ``hipsparseLtMatmulAlgSelectionInit`` creates a algorithm selection descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       algSelection (:py:obj:`~.hipsparseLtMatmulAlgSelection_t`/:py:obj:`~.object`) -- *OUT*:
           the pointer to the algorithm selection descriptor.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix multiplication descriptor.

       alg (:py:obj:`~.hipsparseLtMatmulAlg_t`) -- *IN*:
           the algorithm used to perform the matrix multiplication.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matmulDescr,`` or ``algSelection`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t * handle, hipsparseLtMatmulAlgSelection_t * algSelection, const hipsparseLtMatmulDescriptor_t * matmulDescr, hipsparseLtMatmulAlg_t alg)


.. py:function:: hipsparseLtMatmulAlgSelectionDestroy(algSelection)

   Destroy the algorithm selection descriptor.

   ``hipsparseLtMatmulAlgSelectionDestroy`` releases the resources used by an instance
   of the algorithm selection. This function is the last call with a specific algorithm selection
   instance.

   Args:
       algSelection (:py:obj:`~.hipsparseLtMatmulAlgSelection_t`/:py:obj:`~.object`) -- *IN*:
           the algorithm selection descriptor

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``algSelection`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulAlgSelectionDestroy(const hipsparseLtMatmulAlgSelection_t * algSelection)


.. py:function:: hipsparseLtMatmulAlgSetAttribute(handle, algSelection, attribute, data, dataSize)

   Specify the algorithm attribute of a algorithm selection descriptor.

   ``hipsparseLtMatmulAlgSetAttribute`` sets the value of the specified attribute
   belonging to a algorithm selection descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       algSelection (:py:obj:`~.hipsparseLtMatmulAlgSelection_t`/:py:obj:`~.object`) -- *INOUT*:
           the algorithm selection descriptor.

       attribute (:py:obj:`~.hipsparseLtMatmulAlgAttribute_t`) -- *IN*:
           attributes are specified in ``hipsparseLtMatmulAlgAttribute_t`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the value to which the specified attribute will be set.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``algSelection,`` ``attribute,`` ``data,`` or ``dataSize`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``attribute`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t * handle, hipsparseLtMatmulAlgSelection_t * algSelection, hipsparseLtMatmulAlgAttribute_t attribute, const void * data, size_t dataSize)


.. py:function:: hipsparseLtMatmulAlgGetAttribute(handle, algSelection, attribute, data, dataSize)

   Get the specific algorithm attribute from the algorithm selection descriptor.

   ``hipsparseLtMatmulAlgGetAttribute`` returns the value of the queried attribute belonging
   to the algorithm selection descriptor.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           the hipsparselt handle.

       algSelection (:py:obj:`~.hipsparseLtMatmulAlgSelection_t`/:py:obj:`~.object`) -- *IN*:
           the algorithm selection descriptor.

       attribute (:py:obj:`~.hipsparseLtMatmulAlgAttribute_t`) -- *IN*:
           attributes are specified in ``hipsparseLtMatmulAlgAttribute_t`` .

       data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *INOUT*:
           the memory address containing the attribute value retrieved by this function.

       dataSize (:py:obj:`~.int`) -- *IN*:
           size in bytes of the attribute value used for verification.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``algSelection,`` ``attribute,`` ``data,`` or ``dataSize`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: ``attribute`` is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulAlgSelection_t * algSelection, hipsparseLtMatmulAlgAttribute_t attribute, void * data, size_t dataSize)


.. py:function:: hipsparseLtMatmulGetWorkspace(handle, plan, workspaceSize)

   Determines the required workspace size.

   ``hipsparseLtMatmulGetWorkspace`` determines the required workspace size
   associated with the selected algorithm.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           the matrix multiplication plan descriptor.

       workspaceSize (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           workspace size in bytes.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``algSelection,`` or ``workspaceSize`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulPlan_t * plan, size_t * workspaceSize)


.. py:function:: hipsparseLtMatmulPlanInit(handle, plan, matmulDescr, algSelection)

   Initializes the matrix multiplication plan descriptor.

   ``hipsparseLtMatmulPlanInit`` creates a matrix multiplication plan descriptor.
   It should be destroyed at the end using ``hipsparseLtMatmulPlanDestroy`` .

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *OUT*:
           the matrix multiplication plan descriptor.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           the matrix multiplication descriptor.

       algSelection (:py:obj:`~.hipsparseLtMatmulAlgSelection_t`/:py:obj:`~.object`) -- *IN*:
           the algorithm selection descriptor.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``plan,`` ``matmulDescr,`` ``algSelection,`` or ``workspaceSize`` is invalid. ``HIPSPARSELT_MAT_NUM_BATCHES`` from matrix A to D are inconsistent.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t * handle, hipsparseLtMatmulPlan_t * plan, const hipsparseLtMatmulDescriptor_t * matmulDescr, const hipsparseLtMatmulAlgSelection_t * algSelection)


.. py:function:: hipsparseLtMatmulPlanDestroy(plan)

   Destroy a matrix multiplication plan descriptor.

   ``hipsparseLtMatmulPlanDestroy`` releases the resources used by an instance
   of the matrix multiplication plan. This function is the last call with a specific plan
   instance.

   Args:
       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           the matrix multiplication plan descriptor.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``plan`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t * plan)


.. py:function:: hipsparseLtMatmul(handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, numStreams)

   Sparse matrix and dense matrix multiplication.

   ``hipsparseLtMatmul`` computes the matrix multiplication of matrices ``A`` and ``B`` to
   produce the output matrix ``D``, according to the following operation:

   .. math::

      D := Activation(\alpha \cdot op(A) \cdot op(B) + \beta \cdot C + bias) * scale

   Note:
       This function is non-blocking and executed asynchronously with respect to the host.
       It can return before the actual computation has finished.

   Note:
       This function only supports the case where ``D`` has the same shape of ``C``.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication plan.

       alpha (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           scalar :math:`\alpha` (float).

       d_A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the structured matrix A.

       d_B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the dense matrix B.

       beta (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           scalar :math:`\beta` (float).

       d_C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the dense matrix C.

       d_D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Pointer to the dense matrix D.

       workspace (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the workspace.

       numStreams (:py:obj:`~.int`) -- *IN*:
           Number of HIP streams in ``streams.``

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_INITIALIZED`: ``handle`` or ``plan`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``plan,`` ``alpha,`` ``d_A,`` ``d_B,`` ``beta,`` ``d_C`` , ``d_D`` , ``workspace,`` ``streams,`` or ``numStreams`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.
       * :py:obj:`~.ihipStream_t`:
               Pointer to HIP stream array for the computation.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulPlan_t * plan, const void * alpha, const void * d_A, const void * d_B, const void * beta, const void * d_C, void * d_D, void * workspace, hipStream_t * streams, int32_t numStreams)


.. py:function:: hipsparseLtMatmulSearch(handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, numStreams)

   Sparse matrix and dense matrix multiplication

   ``hipsparseLtMatmulSearch`` evaluates all available algorithms for the matrix multiplication
   and automatically updates the plan by selecting the fastest one.
   The functionality is intended to be used for auto-tuning purposes when the same operation
   is repeated multiple times over different inputs.

   Note:
       The behavior of this function is the same as ``hipsparseLtMatmul`` .

   Note:
       ``d_C`` and ``d_D`` must be two different memory buffers, otherwise the output will be incorrect.

   Note:
       This function is NOT asynchronous with respect to ``streams[0]`` (blocking call).

   Note:
       The number of iterations for the evaluation can be set by using
       ``hipsparseLtMatmulAlgSetAttribute()`` with ``HIPSPARSELT_MATMUL_SEARCH_ITERATIONS``.

   Note:
       The selected algorithm id can be retrieved by using

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication plan.

       alpha (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           scalar :math:`\alpha` (float).

       d_A (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the structured matrix A.

       d_B (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix B.

       beta (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           scalar :math:`\beta` (float).

       d_C (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix C.

       d_D (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           pointer to the dense matrix D.

       workspace (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the worksapce.

       numStreams (:py:obj:`~.int`) -- *IN*:
           number of HIP streams in ``streams.``

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_INITIALIZED`: ``handle`` or ``plan`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``plan,`` ``alpha,`` ``d_A,`` ``d_B,`` ``beta,`` ``d_C,`` ``d_D,`` ``workspace,`` ``streams,`` or ``numStreams`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.
       * :py:obj:`~.ihipStream_t`:
               pointer to HIP stream array for the computation.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t * handle, hipsparseLtMatmulPlan_t * plan, const void * alpha, const void * d_A, const void * d_B, const void * beta, const void * d_C, void * d_D, void * workspace, hipStream_t * streams, int32_t numStreams)


.. py:function:: hipsparseLtSpMMAPrune(handle, matmulDescr, d_in, d_out, pruneAlg, stream)

   Prune a dense matrix.

   ``hipsparseLtSpMMAPrune`` prunes the dense matrix ``d_in`` according to the specified
   algorithm ``pruneAlg``, which can be ``HIPSPARSELT_PRUNE_SPMMA_TILE`` or ``HIPSPARSELT_PRUNE_SPMMA_STRIP``.

   Note:
       The function requires no extra storage. It supports asynchronous execution with respect to ``stream``.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication descriptor.

       d_in (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix.

       d_out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           pointer to the pruned matrix.

       pruneAlg (:py:obj:`~.hipsparseLtPruneAlg_t`) -- *IN*:
           pruning algorithm.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matmulDescr,`` ``d_in,`` or ``d_out`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulDescriptor_t * matmulDescr, const void * d_in, void * d_out, hipsparseLtPruneAlg_t pruneAlg, hipStream_t stream)


.. py:function:: hipsparseLtSpMMAPruneCheck(handle, matmulDescr, d_in, d_valid, stream)

   Check the correctness of the pruning structure for a given matrix.

   ``hipsparseLtSpMMAPruneCheck`` checks the correctness of the pruning structure for a given matrix.
   Contents in the provided matrix must have a sparsity of 2:4.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       matmulDescr (:py:obj:`~.hipsparseLtMatmulDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication descriptor.

       d_in (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the matrix to check.

       d_valid (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *OUT*:
           validation results (0 is correct, and 1 is incorrect).

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``matmulDescr,`` ``d_in,`` or ``d_valid`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulDescriptor_t * matmulDescr, const void * d_in, int * d_valid, hipStream_t stream)


.. py:function:: hipsparseLtSpMMAPrune2(handle, sparseMatDescr, isSparseA, op, d_in, d_out, pruneAlg, stream)

   Prune a dense matrix.

   ``hipsparseLtSpMMAPrune2`` prunes the dense matrix ``d_in`` according to the specified
   algorithm ``pruneAlg``, which can be ``HIPSPARSELT_PRUNE_SPMMA_TILE`` or ``HIPSPARSELT_PRUNE_SPMMA_STRIP``.

   Note:
       The function requires no extra storage. It supports asynchronous execution with respect to ``stream``.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       sparseMatDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           structured (sparse) matrix descriptor.

       isSparseA (:py:obj:`~.int`) -- *IN*:
           specify if the structured (or sparse) matrix is in the first position (matA or matB). (It currently only supports matA.)

       op (:py:obj:`~.hipsparseOperation_t`) -- *IN*:
           operation that will be applied to the structured (or sparse) matrix in the multiplication.

       d_in (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix.

       d_out (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           pointer to the pruned matrix.

       pruneAlg (:py:obj:`~.hipsparseLtPruneAlg_t`) -- *IN*:
           pruning algorithm.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``sparseMatDescr,`` ``op,`` ``d_in,`` or ``d_out`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t * handle, const hipsparseLtMatDescriptor_t * sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void * d_in, void * d_out, hipsparseLtPruneAlg_t pruneAlg, hipStream_t stream)


.. py:function:: hipsparseLtSpMMAPruneCheck2(handle, sparseMatDescr, isSparseA, op, d_in, d_valid, stream)

   Check the correctness of the pruning structure for a given matrix.

   ``hipsparseLtSpMMAPruneCheck2`` checks the correctness of the pruning structure for a given matrix.
   Contents in the provided matrix must have a sparsity of 2:4.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       sparseMatDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           structured (sparse) matrix descriptor.

       isSparseA (:py:obj:`~.int`) -- *IN*:
           specify if the structured (or sparse) matrix is in the first position (matA or matB). (The HIP backend only supports matA.)

       op (:py:obj:`~.hipsparseOperation_t`) -- *IN*:
           operation that will be applied to the structured (or sparse) matrix in the multiplication.

       d_in (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the matrix to check.

       d_valid (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *OUT*:
           validation results (0 is correct, and 1 is incorrect).

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``sparseMatDescr,`` ``op,`` ``d_in,`` or ``d_valid`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t * handle, const hipsparseLtMatDescriptor_t * sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void * d_in, int * d_valid, hipStream_t stream)


.. py:function:: hipsparseLtSpMMACompressedSize(handle, plan, compressedSize, compressBufferSize)

   Provide the size of the compressed matrix.

   ``hipsparseLtSpMMACompressedSize`` provides the size of the compressed matrix
   to be allocated before calling ``hipsparseLtSpMMACompress`` () or ``hipsparseLtSpMMACompress2`` ().

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication plan descriptor.

       compressedSize (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           size in bytes of the compressed matrix.

       compressBufferSize (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           size in bytes for the buffer needed for the matrix compression.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``plan,`` ``compressedSize,`` or ``compressBufferSize`` is invalid.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulPlan_t * plan, size_t * compressedSize, size_t * compressBufferSize)


.. py:function:: hipsparseLtSpMMACompress(handle, plan, d_dense, d_compressed, d_compressBuffer, stream)

   Compress a dense matrix to structured matrix.

   ``hipsparseLtSpMMACompress`` compresses the dense matrix ``d_dense``.
   The compressed matrix is intended to be used as the first/second operand A/B
   in the ``hipsparseLtMatmul`` () function.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           handle to the hipsparselt library context queue.

       plan (:py:obj:`~.hipsparseLtMatmulPlan_t`/:py:obj:`~.object`) -- *IN*:
           matrix multiplication plan descriptor.

       d_dense (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix.

       d_compressed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           compressed matrix and metadata.

       d_compressBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           temporary buffer for the compression.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``plan,`` ``d_dense,`` or ``d_compressed`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t * handle, const hipsparseLtMatmulPlan_t * plan, const void * d_dense, void * d_compressed, void * d_compressBuffer, hipStream_t stream)


.. py:function:: hipsparseLtSpMMACompressedSize2(handle, sparseMatDescr, compressedSize, compressBufferSize)

   Provide the size of the compressed matrix.

   ``hipsparseLtSpMMACompressedSize2`` provides the size of the compressed matrix
   to be allocated before calling ``hipsparseLtSpMMACompress`` or ``hipsparseLtSpMMACompress2`` .

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           hipsparselt library handle.

       sparseMatDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           structured (sparse) matrix descriptor.

       compressedSize (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           size in bytes of the compressed matrix.

       compressBufferSize (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *OUT*:
           size in bytes for the buffer needed for the matrix compression.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_INITIALIZED`: ``handle,`` ``sparseMatDescr,`` ``compressedSize,`` or ``compressBufferSize`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t * handle, const hipsparseLtMatDescriptor_t * sparseMatDescr, size_t * compressedSize, size_t * compressBufferSize)


.. py:function:: hipsparseLtSpMMACompress2(handle, sparseMatDescr, isSparseA, op, d_dense, d_compressed, d_compressBuffer, stream)

   Compress a dense matrix to structured matrix.

   ``hipsparseLtSpMMACompress2`` compresses the dense matrix ``d_dense``.
   The compressed matrix is intended to be used as the first/second operand A/B
   in the ``hipsparseLtMatmul`` () function.

   Args:
       handle (:py:obj:`~.hipsparseLtHandle_t`/:py:obj:`~.object`) -- *IN*:
           handle to the hipsparselt library context queue.

       sparseMatDescr (:py:obj:`~.hipsparseLtMatDescriptor_t`/:py:obj:`~.object`) -- *IN*:
           structured (sparse) matrix descriptor.

       isSparseA (:py:obj:`~.int`) -- *IN*:
           specify whether the structured (or sparse) matrix is in the first position (matA or matB).

       op (:py:obj:`~.hipsparseOperation_t`) -- *IN*:
           operation that will be applied to the structured (or sparse) matrix in the multiplication.

       d_dense (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           pointer to the dense matrix.

       d_compressed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           compressed matrix and metadata.

       d_compressBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           temporary buffer for the compression.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           HIP stream for the computation.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipsparseStatus_t`: One of:
               - :py:obj:`~.HIPSPARSE_STATUS_SUCCESS`: the operation completed successfully.
               - :py:obj:`~.HIPSPARSE_STATUS_INVALID_VALUE`: ``handle,`` ``sparseMatDescr,`` ``op,`` ``d_dense,`` or ``d_compressed`` is invalid.
               - :py:obj:`~.HIPSPARSE_STATUS_NOT_SUPPORTED`: the problem is not supported.

   .. rubric:: C signature

   .. code-block:: c

       hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t * handle, const hipsparseLtMatDescriptor_t * sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void * d_dense, void * d_compressed, void * d_compressBuffer, hipStream_t stream)


