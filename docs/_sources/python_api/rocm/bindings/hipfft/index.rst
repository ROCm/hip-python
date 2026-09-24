rocm.bindings.hipfft
====================

.. py:module:: rocm.bindings.hipfft


Attributes
----------

.. autoapisummary::

   rocm.bindings.hipfft.HIPFFT_FORWARD
   rocm.bindings.hipfft.HIPFFT_BACKWARD
   rocm.bindings.hipfft.hipfftResult
   rocm.bindings.hipfft.hipfftType
   rocm.bindings.hipfft.hipfftLibraryPropertyType
   rocm.bindings.hipfft.hipfftHandle
   rocm.bindings.hipfft.hipfftComplex
   rocm.bindings.hipfft.hipfftDoubleComplex


Classes
-------

.. autoapisummary::

   rocm.bindings.hipfft.hipfftResult_t
   rocm.bindings.hipfft.hipfftType_t
   rocm.bindings.hipfft.hipfftLibraryPropertyType_t
   rocm.bindings.hipfft.hipfftHandle_t


Functions
---------

.. autoapisummary::

   rocm.bindings.hipfft.has_symbol
   rocm.bindings.hipfft.hipfftPlan1d
   rocm.bindings.hipfft.hipfftPlan2d
   rocm.bindings.hipfft.hipfftPlan3d
   rocm.bindings.hipfft.hipfftPlanMany
   rocm.bindings.hipfft.hipfftCreate
   rocm.bindings.hipfft.hipfftExtPlanScaleFactor
   rocm.bindings.hipfft.hipfftMakePlan1d
   rocm.bindings.hipfft.hipfftMakePlan2d
   rocm.bindings.hipfft.hipfftMakePlan3d
   rocm.bindings.hipfft.hipfftMakePlanMany
   rocm.bindings.hipfft.hipfftMakePlanMany64
   rocm.bindings.hipfft.hipfftEstimate1d
   rocm.bindings.hipfft.hipfftEstimate2d
   rocm.bindings.hipfft.hipfftEstimate3d
   rocm.bindings.hipfft.hipfftEstimateMany
   rocm.bindings.hipfft.hipfftGetSize1d
   rocm.bindings.hipfft.hipfftGetSize2d
   rocm.bindings.hipfft.hipfftGetSize3d
   rocm.bindings.hipfft.hipfftGetSizeMany
   rocm.bindings.hipfft.hipfftGetSizeMany64
   rocm.bindings.hipfft.hipfftGetSize
   rocm.bindings.hipfft.hipfftSetAutoAllocation
   rocm.bindings.hipfft.hipfftSetWorkArea
   rocm.bindings.hipfft.hipfftExecC2C
   rocm.bindings.hipfft.hipfftExecR2C
   rocm.bindings.hipfft.hipfftExecC2R
   rocm.bindings.hipfft.hipfftExecZ2Z
   rocm.bindings.hipfft.hipfftExecD2Z
   rocm.bindings.hipfft.hipfftExecZ2D
   rocm.bindings.hipfft.hipfftSetStream
   rocm.bindings.hipfft.hipfftDestroy
   rocm.bindings.hipfft.hipfftGetVersion
   rocm.bindings.hipfft.hipfftGetProperty


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: HIPFFT_FORWARD
   :type:  Any

.. py:data:: HIPFFT_BACKWARD
   :type:  Any

.. py:class:: hipfftResult_t

   Bases: :py:obj:`enum.IntEnum`


   Result/status/error codes
       


   .. py:attribute:: HIPFFT_SUCCESS
      :type:  int


   .. py:attribute:: HIPFFT_INVALID_PLAN
      :type:  int


   .. py:attribute:: HIPFFT_ALLOC_FAILED
      :type:  int


   .. py:attribute:: HIPFFT_INVALID_TYPE
      :type:  int


   .. py:attribute:: HIPFFT_INVALID_VALUE
      :type:  int


   .. py:attribute:: HIPFFT_INTERNAL_ERROR
      :type:  int


   .. py:attribute:: HIPFFT_EXEC_FAILED
      :type:  int


   .. py:attribute:: HIPFFT_SETUP_FAILED
      :type:  int


   .. py:attribute:: HIPFFT_INVALID_SIZE
      :type:  int


   .. py:attribute:: HIPFFT_UNALIGNED_DATA
      :type:  int


   .. py:attribute:: HIPFFT_INCOMPLETE_PARAMETER_LIST
      :type:  int


   .. py:attribute:: HIPFFT_INVALID_DEVICE
      :type:  int


   .. py:attribute:: HIPFFT_PARSE_ERROR
      :type:  int


   .. py:attribute:: HIPFFT_NO_WORKSPACE
      :type:  int


   .. py:attribute:: HIPFFT_NOT_IMPLEMENTED
      :type:  int


   .. py:attribute:: HIPFFT_NOT_SUPPORTED
      :type:  int


.. py:data:: hipfftResult

.. py:class:: hipfftType_t

   Bases: :py:obj:`enum.IntEnum`


   Transform type

   This type is used to declare the Fourier transform type that will be executed.


   .. py:attribute:: HIPFFT_R2C
      :type:  int


   .. py:attribute:: HIPFFT_C2R
      :type:  int


   .. py:attribute:: HIPFFT_C2C
      :type:  int


   .. py:attribute:: HIPFFT_D2Z
      :type:  int


   .. py:attribute:: HIPFFT_Z2D
      :type:  int


   .. py:attribute:: HIPFFT_Z2Z
      :type:  int


.. py:data:: hipfftType

.. py:class:: hipfftLibraryPropertyType_t

   Bases: :py:obj:`enum.IntEnum`


   (No short description)
       


   .. py:attribute:: HIPFFT_MAJOR_VERSION
      :type:  int


   .. py:attribute:: HIPFFT_MINOR_VERSION
      :type:  int


   .. py:attribute:: HIPFFT_PATCH_LEVEL
      :type:  int


.. py:data:: hipfftLibraryPropertyType

.. py:class:: hipfftHandle_t(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: hipfftHandle

.. py:data:: hipfftComplex

.. py:data:: hipfftDoubleComplex

.. py:function:: hipfftPlan1d(nx, type, batch)

   Create a new one-dimensional FFT plan.

   Allocate and initialize a new one-dimensional FFT plan.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           FFT length.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to compute.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftPlan1d(hipfftHandle * plan, int nx, hipfftType type, int batch)


.. py:function:: hipfftPlan2d(nx, ny, type)

   Create a new two-dimensional FFT plan.

   Allocate and initialize a new two-dimensional FFT plan.
   Two-dimensional data should be stored in C ordering (row-major
   format), so that indexes in y-direction (j index) vary the
   fastest.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction (slow index).

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction (fast index).

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftPlan2d(hipfftHandle * plan, int nx, int ny, hipfftType type)


.. py:function:: hipfftPlan3d(nx, ny, nz, type)

   Create a new three-dimensional FFT plan.

   Allocate and initialize a new three-dimensional FFT plan.
   Three-dimensional data should be stored in C ordering (row-major
   format), so that indexes in z-direction (k index) vary the
   fastest.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction (slowest index).

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       nz (:py:obj:`~.int`) -- *IN*:
           Number of elements in the z-direction (fastest index).

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftPlan3d(hipfftHandle * plan, int nx, int ny, int nz, hipfftType type)


.. py:function:: hipfftPlanMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   Create a new batched rank-dimensional FFT plan with advanced data layout.

   Allocate and initialize a new batched rank-dimensional
   FFT plan. The number of elements to transform in each direction of
   the input data is specified in n.

   The batch parameter tells hipFFT how many transforms to perform.
   The distance between the first elements of two consecutive batches
   of the input and output data are specified with the idist and odist
   parameters.

   The inembed and onembed parameters define the input and output data
   layouts. The number of elements in the data is assumed to be larger
   than the number of elements in the transform. Strided data layouts
   are also supported. Strides along the fastest direction in the input
   and output data are specified via the istride and ostride parameters.

   If both inembed and onembed parameters are set to NULL, all the
   advanced data layout parameters are ignored and reverted to default
   values, i.e., the batched transform is performed with non-strided data
   access and the number of data/transform elements are assumed to be
   equivalent.

   Args:
       rank (:py:obj:`~.int`) -- *IN*:
           Dimension of transform (1, 2, or 3).

       n (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements to transform in the x/y/z directions.

       inembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the input data in the x/y/z directions.

       istride (:py:obj:`~.int`) -- *IN*:
           Distance between two successive elements in the input data.

       idist (:py:obj:`~.int`) -- *IN*:
           Distance between input batches.

       onembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the output data in the x/y/z directions.

       ostride (:py:obj:`~.int`) -- *IN*:
           Distance between two successive elements in the output data.

       odist (:py:obj:`~.int`) -- *IN*:
           Distance between output batches.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftPlanMany(hipfftHandle * plan, int rank, int * n, int * inembed, int istride, int idist, int * onembed, int ostride, int odist, hipfftType type, int batch)


.. py:function:: hipfftCreate()

   Allocate a new plan.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle to be allocated.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftCreate(hipfftHandle * plan)


.. py:function:: hipfftExtPlanScaleFactor(plan, scalefactor)

   Set scaling factor.

   hipFFT multiplies each element of the result by the given factor at the end of the transform.

   The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.

   This function must be called after the plan is allocated using
   ::hipfftCreate, but before the plan is initialized by any of the
   "MakePlan" functions.  Therefore, API functions that combine
   creation and initialization (::hipfftPlan1d, ::hipfftPlan2d,
   ::hipfftPlan3d, and ::hipfftPlanMany) cannot set a scale factor.

   Note that the scale factor applies to both forward and
   backward transforms executed with the specified plan handle.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`):
           (undocumented)

       scalefactor (:py:obj:`~.float`/:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExtPlanScaleFactor(hipfftHandle plan, double scalefactor)


.. py:function:: hipfftMakePlan1d(plan, nx, type, batch)

   Initialize a new one-dimensional FFT plan.

   Assumes that the plan has been created already, and
   modifies the plan associated with the plan handle.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Handle of the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           FFT length.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to compute.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftMakePlan2d(plan, nx, ny, type)

   Initialize a new two-dimensional FFT plan.

   Assumes that the plan has been created already, and
   modifies the plan associated with the plan handle.
   Two-dimensional data should be stored in C ordering (row-major
   format), so that indexes in y-direction (j index) vary the
   fastest.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Handle of the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction (slow index).

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction (fast index).

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t * workSize)


.. py:function:: hipfftMakePlan3d(plan, nx, ny, nz, type)

   Initialize a new two-dimensional FFT plan.

   Assumes that the plan has been created already, and
   modifies the plan associated with the plan handle.
   Three-dimensional data should be stored in C ordering (row-major
   format), so that indexes in z-direction (k index) vary the
   fastest.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Handle of the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction (slowest index).

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       nz (:py:obj:`~.int`) -- *IN*:
           Number of elements in the z-direction (fastest index).

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t * workSize)


.. py:function:: hipfftMakePlanMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   Initialize a new batched rank-dimensional FFT plan with advanced data layout.

   Assumes that the plan has been created already, and
   modifies the plan associated with the plan handle. The number
   of elements to transform in each direction of the input data
   in the FFT plan is specified in n.

   The batch parameter tells hipFFT how many transforms to perform.
   The distance between the first elements of two consecutive batches
   of the input and output data are specified with the idist and odist
   parameters.

   The inembed and onembed parameters define the input and output data
   layouts. The number of elements in the data is assumed to be larger
   than the number of elements in the transform. Strided data layouts
   are also supported. Strides along the fastest direction in the input
   and output data are specified via the istride and ostride parameters.

   If both inembed and onembed parameters are set to NULL, all the
   advanced data layout parameters are ignored and reverted to default
   values, i.e., the batched transform is performed with non-strided data
   access and the number of data/transform elements are assumed to be
   equivalent.

   Args:
       rank (:py:obj:`~.int`) -- *IN*:
           Dimension of transform (1, 2, or 3).

       n (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements to transform in the x/y/z directions.

       inembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the input data in the x/y/z directions.

       istride (:py:obj:`~.int`) -- *IN*:
           Distance between two successive elements in the input data.

       idist (:py:obj:`~.int`) -- *IN*:
           Distance between input batches.

       onembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the output data in the x/y/z directions.

       ostride (:py:obj:`~.int`) -- *IN*:
           Distance between two successive elements in the output data.

       odist (:py:obj:`~.int`) -- *IN*:
           Distance between output batches.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.hipfftHandle_t`:
               Pointer to the FFT plan handle.
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftMakePlanMany(hipfftHandle plan, int rank, int * n, int * inembed, int istride, int idist, int * onembed, int ostride, int odist, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   (No short description, might be part of a group.)

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       inembed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       istride (:py:obj:`~.int`):
           (undocumented)

       idist (:py:obj:`~.int`):
           (undocumented)

       onembed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ostride (:py:obj:`~.int`):
           (undocumented)

       odist (:py:obj:`~.int`):
           (undocumented)

       type (:py:obj:`~.hipfftType_t`):
           (undocumented)

       batch (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * workSize (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftMakePlanMany64(hipfftHandle plan, int rank, long long * n, long long * inembed, long long istride, long long idist, long long * onembed, long long ostride, long long odist, hipfftType type, long long batch, size_t * workSize)


.. py:function:: hipfftEstimate1d(nx, type, batch)

   Return an estimate of the work area size required for a 1D plan.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftEstimate2d(nx, ny, type)

   Return an estimate of the work area size required for a 2D plan.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t * workSize)


.. py:function:: hipfftEstimate3d(nx, ny, nz, type)

   Return an estimate of the work area size required for a 3D plan.

   Args:
       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       nz (:py:obj:`~.int`) -- *IN*:
           Number of elements in the z-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t * workSize)


.. py:function:: hipfftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   Return an estimate of the work area size required for a rank-dimensional plan.

   Args:
       rank (:py:obj:`~.int`) -- *IN*:
           Dimension of FFT transform (1, 2, or 3).

       n (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the x/y/z directions.

       inembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:

       istride (:py:obj:`~.int`) -- *IN*:

       idist (:py:obj:`~.int`) -- *IN*:
           Distance between input batches.

       onembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:

       ostride (:py:obj:`~.int`) -- *IN*:

       odist (:py:obj:`~.int`) -- *IN*:
           Distance between output batches.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftEstimateMany(int rank, int * n, int * inembed, int istride, int idist, int * onembed, int ostride, int odist, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftGetSize1d(plan, nx, type, batch)

   Return size of the work area size required for a 1D plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftGetSize2d(plan, nx, ny, type)

   Return size of the work area size required for a 2D plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t * workSize)


.. py:function:: hipfftGetSize3d(plan, nx, ny, nz, type)

   Return size of the work area size required for a 3D plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       nx (:py:obj:`~.int`) -- *IN*:
           Number of elements in the x-direction.

       ny (:py:obj:`~.int`) -- *IN*:
           Number of elements in the y-direction.

       nz (:py:obj:`~.int`) -- *IN*:
           Number of elements in the z-direction.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t * workSize)


.. py:function:: hipfftGetSizeMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   Return size of the work area size required for a rank-dimensional plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       rank (:py:obj:`~.int`) -- *IN*:
           Dimension of FFT transform (1, 2, or 3).

       n (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:
           Number of elements in the x/y/z directions.

       inembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:

       istride (:py:obj:`~.int`) -- *IN*:

       idist (:py:obj:`~.int`) -- *IN*:
           Distance between input batches.

       onembed (:py:obj:`~.rocm.bindings.util.types.PointerToInt`/:py:obj:`~.object`) -- *IN*:

       ostride (:py:obj:`~.int`) -- *IN*:

       odist (:py:obj:`~.int`) -- *IN*:
           Distance between output batches.

       type (:py:obj:`~.hipfftType_t`) -- *IN*:
           FFT type.

       batch (:py:obj:`~.int`) -- *IN*:
           Number of batched transforms to perform.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSizeMany(hipfftHandle plan, int rank, int * n, int * inembed, int istride, int idist, int * onembed, int ostride, int odist, hipfftType type, int batch, size_t * workSize)


.. py:function:: hipfftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)

   (No short description, might be part of a group.)

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`):
           (undocumented)

       rank (:py:obj:`~.int`):
           (undocumented)

       n (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       inembed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       istride (:py:obj:`~.int`):
           (undocumented)

       idist (:py:obj:`~.int`):
           (undocumented)

       onembed (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ostride (:py:obj:`~.int`):
           (undocumented)

       odist (:py:obj:`~.int`):
           (undocumented)

       type (:py:obj:`~.hipfftType_t`):
           (undocumented)

       batch (:py:obj:`~.int`):
           (undocumented)

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * workSize (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSizeMany64(hipfftHandle plan, int rank, long long * n, long long * inembed, long long istride, long long idist, long long * onembed, long long ostride, long long odist, hipfftType type, long long batch, size_t * workSize)


.. py:function:: hipfftGetSize(plan)

   Return size of the work area size required for a rank-dimensional plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Pointer to work area size (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetSize(hipfftHandle plan, size_t * workSize)


.. py:function:: hipfftSetAutoAllocation(plan, autoAllocate)

   Set the plan's auto-allocation flag.  The plan will allocate its own workarea.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       autoAllocate (:py:obj:`~.int`) -- *IN*:
           0 to disable auto-allocation, non-zero to enable.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)


.. py:function:: hipfftSetWorkArea(plan, workArea)

   Set the plan's work area.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Pointer to the FFT plan.

       workArea (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Pointer to the work area (on device).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftSetWorkArea(hipfftHandle plan, void * workArea)


.. py:function:: hipfftExecC2C(plan, idata, odata, direction)

   Execute a (float) complex-to-complex FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

       direction (:py:obj:`~.int`) -- *IN*:
           Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex * idata, hipfftComplex * odata, int direction)


.. py:function:: hipfftExecR2C(plan, idata, odata)

   Execute a (float) real-to-complex FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal * idata, hipfftComplex * odata)


.. py:function:: hipfftExecC2R(plan, idata, odata)

   Execute a (float) complex-to-real FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex * idata, hipfftReal * odata)


.. py:function:: hipfftExecZ2Z(plan, idata, odata, direction)

   Execute a (double) complex-to-complex FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

       direction (:py:obj:`~.int`) -- *IN*:
           Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex * idata, hipfftDoubleComplex * odata, int direction)


.. py:function:: hipfftExecD2Z(plan, idata, odata)

   Execute a (double) real-to-complex FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal * idata, hipfftDoubleComplex * odata)


.. py:function:: hipfftExecZ2D(plan, idata, odata)

   Execute a (double) complex-to-real FFT.

   If the input and output buffers are equal, an in-place
   transform is performed.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       idata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Input data (on device).

       odata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *OUT*:
           Output data (on device).

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex * idata, hipfftDoubleReal * odata)


.. py:function:: hipfftSetStream(plan, stream)

   Set HIP stream to execute plan on.

   Associates a HIP stream with a hipFFT plan.  All kernels
   launched by this plan are associated with the provided stream.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           The FFT plan.

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           The HIP stream.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)


.. py:function:: hipfftDestroy(plan)

   Destroy and deallocate an existing plan.

   Args:
       plan (:py:obj:`~.hipfftHandle_t`/:py:obj:`~.object`) -- *IN*:
           Handle of the FFT plan to be destroyed.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftDestroy(hipfftHandle plan)


.. py:function:: hipfftGetVersion()

   Get rocFFT/cuFFT version.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               cuFFT/rocFFT version (returned value).

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetVersion(int * version)


.. py:function:: hipfftGetProperty(type)

   Get library property.

   Args:
       type (:py:obj:`~.hipfftLibraryPropertyType_t`) -- *IN*:
           Property type.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipfftResult_t`: (undocumented)
       * :py:obj:`~.int`:
               Returned value.

   .. rubric:: C signature

   .. code-block:: c

       hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int * value)


