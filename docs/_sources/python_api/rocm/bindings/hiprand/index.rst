rocm.bindings.hiprand
=====================

.. py:module:: rocm.bindings.hiprand


Attributes
----------

.. autoapisummary::

   rocm.bindings.hiprand.HIPRAND_VERSION
   rocm.bindings.hiprand.HIPRAND_DEFAULT_MAX_BLOCK_SIZE
   rocm.bindings.hiprand.HIPRAND_DEFAULT_MIN_WARPS_PER_EU
   rocm.bindings.hiprand.rocrand_discrete_distribution
   rocm.bindings.hiprand.rocrand_generator
   rocm.bindings.hiprand.hiprandGenerator_st
   rocm.bindings.hiprand.hiprandDiscreteDistribution_st
   rocm.bindings.hiprand.hiprandGenerator_t
   rocm.bindings.hiprand.hiprandDiscreteDistribution_t
   rocm.bindings.hiprand.hiprandStatus_t
   rocm.bindings.hiprand.hiprandRngType_t
   rocm.bindings.hiprand.hiprandOrdering_t
   rocm.bindings.hiprand.hiprandDirectionVectorSet_t


Classes
-------

.. autoapisummary::

   rocm.bindings.hiprand.uint4
   rocm.bindings.hiprand.rocrand_discrete_distribution_st
   rocm.bindings.hiprand.rocrand_generator_base_type
   rocm.bindings.hiprand.rocrand_status
   rocm.bindings.hiprand.rocrand_rng_type
   rocm.bindings.hiprand.rocrand_ordering
   rocm.bindings.hiprand.rocrand_direction_vector_set
   rocm.bindings.hiprand.hiprandStatus
   rocm.bindings.hiprand.hiprandRngType
   rocm.bindings.hiprand.hiprandOrdering
   rocm.bindings.hiprand.hiprandDirectionVectorSet


Functions
---------

.. autoapisummary::

   rocm.bindings.hiprand.has_symbol
   rocm.bindings.hiprand.hiprandCreateGenerator
   rocm.bindings.hiprand.hiprandCreateGeneratorHost
   rocm.bindings.hiprand.hiprandDestroyGenerator
   rocm.bindings.hiprand.hiprandGenerate
   rocm.bindings.hiprand.hiprandGenerateChar
   rocm.bindings.hiprand.hiprandGenerateShort
   rocm.bindings.hiprand.hiprandGenerateLongLong
   rocm.bindings.hiprand.hiprandGenerateUniform
   rocm.bindings.hiprand.hiprandGenerateUniformDouble
   rocm.bindings.hiprand.hiprandGenerateUniformHalf
   rocm.bindings.hiprand.hiprandGenerateNormal
   rocm.bindings.hiprand.hiprandGenerateNormalDouble
   rocm.bindings.hiprand.hiprandGenerateNormalHalf
   rocm.bindings.hiprand.hiprandGenerateLogNormal
   rocm.bindings.hiprand.hiprandGenerateLogNormalDouble
   rocm.bindings.hiprand.hiprandGenerateLogNormalHalf
   rocm.bindings.hiprand.hiprandGeneratePoisson
   rocm.bindings.hiprand.hiprandGenerateSeeds
   rocm.bindings.hiprand.hiprandSetStream
   rocm.bindings.hiprand.hiprandSetPseudoRandomGeneratorSeed
   rocm.bindings.hiprand.hiprandSetGeneratorOffset
   rocm.bindings.hiprand.hiprandSetGeneratorOrdering
   rocm.bindings.hiprand.hiprandSetQuasiRandomGeneratorDimensions
   rocm.bindings.hiprand.hiprandGetVersion
   rocm.bindings.hiprand.hiprandCreatePoissonDistribution
   rocm.bindings.hiprand.hiprandDestroyDistribution
   rocm.bindings.hiprand.hiprandGetDirectionVectors32
   rocm.bindings.hiprand.hiprandGetDirectionVectors64
   rocm.bindings.hiprand.hiprandGetScrambleConstants32
   rocm.bindings.hiprand.hiprandGetScrambleConstants64


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: HIPRAND_VERSION
   :type:  Any

.. py:data:: HIPRAND_DEFAULT_MAX_BLOCK_SIZE
   :type:  Any

.. py:data:: HIPRAND_DEFAULT_MIN_WARPS_PER_EU
   :type:  Any

.. py:class:: uint4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: x
      :type:  Any


   .. py:attribute:: y
      :type:  Any


   .. py:attribute:: z
      :type:  Any


   .. py:attribute:: w
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: rocrand_discrete_distribution_st(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a discrete probability distribution
       


   .. py:attribute:: size
      :type:  Any


   .. py:attribute:: offset
      :type:  Any


   .. py:attribute:: alias
      :type:  Any


   .. py:attribute:: probability
      :type:  Any


   .. py:attribute:: cdf
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: rocrand_discrete_distribution

.. py:class:: rocrand_generator_base_type(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: rocrand_generator

.. py:class:: rocrand_status

   Bases: :py:obj:`enum.IntEnum`


   rocRAND function call status type
       


   .. py:attribute:: ROCRAND_STATUS_SUCCESS
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_VERSION_MISMATCH
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_NOT_CREATED
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_ALLOCATION_FAILED
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_TYPE_ERROR
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_OUT_OF_RANGE
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_LAUNCH_FAILURE
      :type:  int


   .. py:attribute:: ROCRAND_STATUS_INTERNAL_ERROR
      :type:  int


.. py:class:: rocrand_rng_type

   Bases: :py:obj:`enum.IntEnum`


   rocRAND generator type
       


   .. py:attribute:: ROCRAND_RNG_PSEUDO_DEFAULT
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_XORWOW
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_MRG32K3A
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_MTGP32
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_PHILOX4_32_10
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_MRG31K3P
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_LFSR113
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_MT19937
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_THREEFRY2_32_20
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_THREEFRY2_64_20
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_THREEFRY4_32_20
      :type:  int


   .. py:attribute:: ROCRAND_RNG_PSEUDO_THREEFRY4_64_20
      :type:  int


   .. py:attribute:: ROCRAND_RNG_QUASI_DEFAULT
      :type:  int


   .. py:attribute:: ROCRAND_RNG_QUASI_SOBOL32
      :type:  int


   .. py:attribute:: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
      :type:  int


   .. py:attribute:: ROCRAND_RNG_QUASI_SOBOL64
      :type:  int


   .. py:attribute:: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64
      :type:  int


.. py:class:: rocrand_ordering

   Bases: :py:obj:`enum.IntEnum`


   rocRAND generator ordering
       


   .. py:attribute:: ROCRAND_ORDERING_PSEUDO_BEST
      :type:  int


   .. py:attribute:: ROCRAND_ORDERING_PSEUDO_DEFAULT
      :type:  int


   .. py:attribute:: ROCRAND_ORDERING_PSEUDO_SEEDED
      :type:  int


   .. py:attribute:: ROCRAND_ORDERING_PSEUDO_LEGACY
      :type:  int


   .. py:attribute:: ROCRAND_ORDERING_PSEUDO_DYNAMIC
      :type:  int


   .. py:attribute:: ROCRAND_ORDERING_QUASI_DEFAULT
      :type:  int


.. py:class:: rocrand_direction_vector_set

   Bases: :py:obj:`enum.IntEnum`


   rocRAND vector set
       


   .. py:attribute:: ROCRAND_DIRECTION_VECTORS_32_JOEKUO6
      :type:  int


   .. py:attribute:: ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
      :type:  int


   .. py:attribute:: ROCRAND_DIRECTION_VECTORS_64_JOEKUO6
      :type:  int


   .. py:attribute:: ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
      :type:  int


.. py:data:: hiprandGenerator_st

.. py:data:: hiprandDiscreteDistribution_st

.. py:data:: hiprandGenerator_t

.. py:data:: hiprandDiscreteDistribution_t

.. py:class:: hiprandStatus

   Bases: :py:obj:`enum.IntEnum`


   hipRAND function call status type
       


   .. py:attribute:: HIPRAND_STATUS_SUCCESS
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_VERSION_MISMATCH
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_NOT_INITIALIZED
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_ALLOCATION_FAILED
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_TYPE_ERROR
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_OUT_OF_RANGE
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_LAUNCH_FAILURE
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_PREEXISTING_FAILURE
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_INITIALIZATION_FAILED
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_ARCH_MISMATCH
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_INTERNAL_ERROR
      :type:  int


   .. py:attribute:: HIPRAND_STATUS_NOT_IMPLEMENTED
      :type:  int


.. py:data:: hiprandStatus_t

.. py:class:: hiprandRngType

   Bases: :py:obj:`enum.IntEnum`


   hipRAND generator type
       


   .. py:attribute:: HIPRAND_RNG_TEST
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_DEFAULT
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_XORWOW
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_MRG32K3A
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_MTGP32
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_MT19937
      :type:  int


   .. py:attribute:: HIPRAND_RNG_PSEUDO_PHILOX4_32_10
      :type:  int


   .. py:attribute:: HIPRAND_RNG_QUASI_DEFAULT
      :type:  int


   .. py:attribute:: HIPRAND_RNG_QUASI_SOBOL32
      :type:  int


   .. py:attribute:: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
      :type:  int


   .. py:attribute:: HIPRAND_RNG_QUASI_SOBOL64
      :type:  int


   .. py:attribute:: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
      :type:  int


.. py:data:: hiprandRngType_t

.. py:class:: hiprandOrdering

   Bases: :py:obj:`enum.IntEnum`


   hipRAND generator ordering
       


   .. py:attribute:: HIPRAND_ORDERING_PSEUDO_BEST
      :type:  int


   .. py:attribute:: HIPRAND_ORDERING_PSEUDO_DEFAULT
      :type:  int


   .. py:attribute:: HIPRAND_ORDERING_PSEUDO_SEEDED
      :type:  int


   .. py:attribute:: HIPRAND_ORDERING_PSEUDO_LEGACY
      :type:  int


   .. py:attribute:: HIPRAND_ORDERING_PSEUDO_DYNAMIC
      :type:  int


   .. py:attribute:: HIPRAND_ORDERING_QUASI_DEFAULT
      :type:  int


.. py:data:: hiprandOrdering_t

.. py:class:: hiprandDirectionVectorSet

   Bases: :py:obj:`enum.IntEnum`


   hipRAND vector set for quasirandom generators.
       


   .. py:attribute:: HIPRAND_DIRECTION_VECTORS_32_JOEKUO6
      :type:  int


   .. py:attribute:: HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6
      :type:  int


   .. py:attribute:: HIPRAND_DIRECTION_VECTORS_64_JOEKUO6
      :type:  int


   .. py:attribute:: HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6
      :type:  int


.. py:data:: hiprandDirectionVectorSet_t

.. py:function:: hiprandCreateGenerator(rng_type)

   Creates a new random number generator.

   Creates a new random number generator of type ``rng_type,``
   and returns it in ``generator.`` That generator will use
   GPU to create random numbers.

   Values for ``rng_type`` are:
   - HIPRAND_RNG_PSEUDO_DEFAULT
   - HIPRAND_RNG_PSEUDO_XORWOW
   - HIPRAND_RNG_PSEUDO_MRG32K3A
   - HIPRAND_RNG_PSEUDO_MTGP32
   - HIPRAND_RNG_PSEUDO_MT19937
   - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
   - HIPRAND_RNG_QUASI_DEFAULT
   - HIPRAND_RNG_QUASI_SOBOL32
   - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
   - HIPRAND_RNG_QUASI_SOBOL64
   - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64

   Args:
       rng_type (:py:obj:`~.hiprandRngType`):
           Type of random number generator to create

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed 

           - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU 

           - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
             dynamically linked library version 

           - HIPRAND_STATUS_TYPE_ERROR if the value for ``rng_type`` is invalid 

           - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type ``rng_type`` is not implemented yet 

           - HIPRAND_STATUS_SUCCESS if generator was created successfully
       * :py:obj:`~.rocrand_generator_base_type`:
               Pointer to generator

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandCreateGenerator(hiprandGenerator_t * generator, hiprandRngType_t rng_type)


.. py:function:: hiprandCreateGeneratorHost(rng_type)

   Creates a new random number generator on host.

   Creates a new host random number generator of type ``rng_type``
   and returns it in ``generator.`` Created generator will use
   host CPU to generate random numbers.

   Values for ``rng_type`` are:
   - HIPRAND_RNG_PSEUDO_DEFAULT
   - HIPRAND_RNG_PSEUDO_XORWOW
   - HIPRAND_RNG_PSEUDO_MRG32K3A
   - HIPRAND_RNG_PSEUDO_MTGP32
   - HIPRAND_RNG_PSEUDO_MT19937
   - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
   - HIPRAND_RNG_QUASI_DEFAULT
   - HIPRAND_RNG_QUASI_SOBOL32
   - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
   - HIPRAND_RNG_QUASI_SOBOL64
   - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64

   Args:
       rng_type (:py:obj:`~.hiprandRngType`):
           Type of random number generator to create

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed 

           - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
             dynamically linked library version 

           - HIPRAND_STATUS_TYPE_ERROR if the value for ``rng_type`` is invalid 

           - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type ``rng_type`` is not implemented yet 

           - HIPRAND_STATUS_SUCCESS if generator was created successfully
       * :py:obj:`~.rocrand_generator_base_type`:
               Pointer to generator

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandCreateGeneratorHost(hiprandGenerator_t * generator, hiprandRngType_t rng_type)


.. py:function:: hiprandDestroyGenerator(generator)

   Destroys random number generator.

   Destroys random number generator and frees related memory.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to be destroyed

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandDestroyGenerator(hiprandGenerator_t generator)


.. py:function:: hiprandGenerate(generator, output_data, n)

   Generates uniformly distributed 32-bit unsigned integers.

   Generates ``n`` uniformly distributed 32-bit unsigned integers and
   saves them to ``output_data.``

   Generated numbers are between ``0`` and ``2^32,`` including ``0`` and
   excluding ``2^32.``

   Note: ``generator`` must be not be of type ``HIPRAND_RNG_QUASI_SOBOL64``
   or ``HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of 32-bit unsigned integers to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerate(hiprandGenerator_t generator, unsigned int * output_data, size_t n)


.. py:function:: hiprandGenerateChar(generator, output_data, n)

   Generates uniformly distributed 8-bit unsigned integers.

   Generates ``n`` uniformly distributed 8-bit unsigned integers and
   saves them to ``output_data.``

   Generated numbers are between ``0`` and ``2^8,`` including ``0`` and
   excluding ``2^8.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of 8-bit unsigned integers to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateChar(hiprandGenerator_t generator, unsigned char * output_data, size_t n)


.. py:function:: hiprandGenerateShort(generator, output_data, n)

   Generates uniformly distributed 16-bit unsigned integers.

   Generates ``n`` uniformly distributed 16-bit unsigned integers and
   saves them to ``output_data.``

   Generated numbers are between ``0`` and ``2^16,`` including ``0`` and
   excluding ``2^16.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of 16-bit unsigned integers to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateShort(hiprandGenerator_t generator, unsigned short * output_data, size_t n)


.. py:function:: hiprandGenerateLongLong(generator, output_data, n)

   Generates uniformly distributed 64-bit unsigned integers.

   Generates ``n`` uniformly distributed 64-bit unsigned integers and
   saves them to ``output_data.``

   Generated numbers are between ``0`` and ``2^64,`` including ``0`` and
   excluding ``2^64.``

   Note: ``generator`` must be of type ``HIPRAND_RNG_QUASI_SOBOL64``
   or ``HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of 64-bit unsigned integers to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateLongLong(hiprandGenerator_t generator, unsigned long long * output_data, size_t n)


.. py:function:: hiprandGenerateUniform(generator, output_data, n)

   Generates uniformly distributed floats.

   Generates ``n`` uniformly distributed 32-bit floating-point values
   and saves them to ``output_data.``

   Generated numbers are between ``0.0f`` and ``1.0f,`` excluding ``0.0f`` and
   including ``1.0f.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of floats to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateUniform(hiprandGenerator_t generator, float * output_data, size_t n)


.. py:function:: hiprandGenerateUniformDouble(generator, output_data, n)

   Generates uniformly distributed double-precision floating-point values.

   Generates ``n`` uniformly distributed 64-bit double-precision floating-point
   values and saves them to ``output_data.``

   Generated numbers are between ``0.0`` and ``1.0,`` excluding ``0.0`` and
   including ``1.0.``

   Note: When ``generator`` is of type: ``HIPRAND_RNG_PSEUDO_MRG32K3A,``
   ``HIPRAND_RNG_PSEUDO_MTGP32,`` ``HIPRAND_RNG_QUASI_SOBOL32,`` or
   ``HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32`` then the returned ``double``
   values are generated from only 32 random bits
   each (one ``unsigned int`` value per one generated ``double).``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of floats to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateUniformDouble(hiprandGenerator_t generator, double * output_data, size_t n)


.. py:function:: hiprandGenerateUniformHalf(generator, output_data, n)

   Generates uniformly distributed half-precision floating-point values.

   Generates ``n`` uniformly distributed 16-bit half-precision floating-point
   values and saves them to ``output_data.``

   Generated numbers are between ``0.0`` and ``1.0,`` excluding ``0.0`` and
   including ``1.0.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of halfs to generate

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateUniformHalf(hiprandGenerator_t generator, half * output_data, size_t n)


.. py:function:: hiprandGenerateNormal(generator, output_data, n, mean, stddev)

   Generates normally distributed floats.

   Generates ``n`` normally distributed 32-bit floating-point
   values and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of floats to generate

       mean (:py:obj:`~.float`/:py:obj:`~.int`):
           Mean value of normal distribution

       stddev (:py:obj:`~.float`/:py:obj:`~.int`):
           Standard deviation value of normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(float2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t generator, float * output_data, size_t n, float mean, float stddev)


.. py:function:: hiprandGenerateNormalDouble(generator, output_data, n, mean, stddev)

   Generates normally distributed doubles.

   Generates ``n`` normally distributed 64-bit double-precision floating-point
   numbers and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of doubles to generate

       mean (:py:obj:`~.float`/:py:obj:`~.int`):
           Mean value of normal distribution

       stddev (:py:obj:`~.float`/:py:obj:`~.int`):
           Standard deviation value of normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(double2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateNormalDouble(hiprandGenerator_t generator, double * output_data, size_t n, double mean, double stddev)


.. py:function:: hiprandGenerateNormalHalf(generator, output_data, n, mean, stddev)

   Generates normally distributed halfs.

   Generates ``n`` normally distributed 16-bit half-precision floating-point
   numbers and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of halfs to generate

       mean (:py:obj:`~.int`):
           Mean value of normal distribution

       stddev (:py:obj:`~.int`):
           Standard deviation value of normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(half2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateNormalHalf(hiprandGenerator_t generator, half * output_data, size_t n, half mean, half stddev)


.. py:function:: hiprandGenerateLogNormal(generator, output_data, n, mean, stddev)

   Generates log-normally distributed floats.

   Generates ``n`` log-normally distributed 32-bit floating-point values
   and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of floats to generate

       mean (:py:obj:`~.float`/:py:obj:`~.int`):
           Mean value of log normal distribution

       stddev (:py:obj:`~.float`/:py:obj:`~.int`):
           Standard deviation value of log normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(float2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateLogNormal(hiprandGenerator_t generator, float * output_data, size_t n, float mean, float stddev)


.. py:function:: hiprandGenerateLogNormalDouble(generator, output_data, n, mean, stddev)

   Generates log-normally distributed doubles.

   Generates ``n`` log-normally distributed 64-bit double-precision floating-point
   values and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of doubles to generate

       mean (:py:obj:`~.float`/:py:obj:`~.int`):
           Mean value of log normal distribution

       stddev (:py:obj:`~.float`/:py:obj:`~.int`):
           Standard deviation value of log normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(double2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateLogNormalDouble(hiprandGenerator_t generator, double * output_data, size_t n, double mean, double stddev)


.. py:function:: hiprandGenerateLogNormalHalf(generator, output_data, n, mean, stddev)

   Generates log-normally distributed halfs.

   Generates ``n`` log-normally distributed 16-bit half-precision floating-point
   values and saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of halfs to generate

       mean (:py:obj:`~.int`):
           Mean value of log normal distribution

       stddev (:py:obj:`~.int`):
           Standard deviation value of log normal distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
           aligned to ``sizeof(half2)`` bytes, or ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateLogNormalHalf(hiprandGenerator_t generator, half * output_data, size_t n, half mean, half stddev)


.. py:function:: hiprandGeneratePoisson(generator, output_data, n, lambda_)

   Generates Poisson-distributed 32-bit unsigned integers.

   Generates ``n`` Poisson-distributed 32-bit unsigned integers and
   saves them to ``output_data.``

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to use

       output_data (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           Pointer to memory to store generated numbers

       n (:py:obj:`~.int`):
           Number of 32-bit unsigned integers to generate

       lambda (:py:obj:`~.float`/:py:obj:`~.int`):
           lambda for the Poisson distribution

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

           - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive 

           - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
           of used quasi-random generator 

           - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGeneratePoisson(hiprandGenerator_t generator, unsigned int * output_data, size_t n, double lambda)


.. py:function:: hiprandGenerateSeeds(generator)

   Initializes the generator's state on GPU or host.

   Initializes the generator's state on GPU or host.

   If hiprandGenerateSeeds() was not called for a generator, it will be
   automatically called by functions which generates random numbers like
   hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to initialize

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created 

           - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
             a previous kernel launch 

           - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason 

           - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGenerateSeeds(hiprandGenerator_t generator)


.. py:function:: hiprandSetStream(generator, stream)

   Sets the current stream for kernel launches.

   Sets the current stream for all kernel launches of the generator.
   All functions will use this stream.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Generator to modify

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Stream to use or NULL for default stream

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_SUCCESS if stream was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandSetStream(hiprandGenerator_t generator, hipStream_t stream)


.. py:function:: hiprandSetPseudoRandomGeneratorSeed(generator, seed)

   Sets the seed of a pseudo-random number generator.

   Sets the seed of the pseudo-random number generator.

   - This operation resets the generator's internal state.
   - This operation does not change the generator's offset.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Pseudo-random number generator

       seed (:py:obj:`~.int`):
           New seed value

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator 

           - HIPRAND_STATUS_SUCCESS if seed was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator, unsigned long long seed)


.. py:function:: hiprandSetGeneratorOffset(generator, offset)

   Sets the offset of a random number generator.

   Sets the absolute offset of the random number generator.

   - This operation resets the generator's internal state.
   - This operation does not change the generator's seed.

   Absolute offset cannot be set if generator's type is
   HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Random number generator

       offset (:py:obj:`~.int`):
           New absolute offset

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_SUCCESS if offset was successfully set 

           - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
           or HIPRAND_RNG_PSEUDO_MT19937

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandSetGeneratorOffset(hiprandGenerator_t generator, unsigned long long offset)


.. py:function:: hiprandSetGeneratorOrdering(generator, order)

   Sets the ordering of a random number generator.

   Sets the ordering of the results of a random number generator.

   - This operation resets the generator's internal state.
   - This operation does not change the generator's seed.

   The ordering choices for pseudorandom sequences are
   HIPRAND_ORDERING_PSEUDO_DEFAULT and
   HIPRAND_ORDERING_PSEUDO_LEGACY.
   The default ordering is HIPRAND_ORDERING_PSEUDO_DEFAULT, which is equal to
   HIPRAND_ORDERING_PSEUDO_LEGACY for now.

   For quasirandom sequences there is only one ordering, HIPRAND_ORDERING_QUASI_DEFAULT.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Random number generator

       order (:py:obj:`~.hiprandOrdering`):
           New ordering of results

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

           - HIPRAND_STATUS_OUT_OF_RANGE if the ordering is not valid 

           - HIPRAND_STATUS_SUCCESS if the ordering was successfully set 

           - HIPRAND_STATUS_TYPE_ERROR if generator's type is not valid

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandSetGeneratorOrdering(hiprandGenerator_t generator, hiprandOrdering_t order)


.. py:function:: hiprandSetQuasiRandomGeneratorDimensions(generator, dimensions)

   Set the number of dimensions of a quasi-random number generator.

   Set the number of dimensions of a quasi-random number generator.
   Supported values of ``dimensions`` are 1 to 20000.

   - This operation resets the generator's internal state.
   - This operation does not change the generator's offset.

   Args:
       generator (:py:obj:`~.rocrand_generator_base_type`/:py:obj:`~.object`):
           Quasi-random number generator

       dimensions (:py:obj:`~.int`):
           Number of dimensions

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_NOT_CREATED if the generator wasn't created 

           - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator 

           - HIPRAND_STATUS_OUT_OF_RANGE if ``dimensions`` is out of range 

           - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator, unsigned int dimensions)


.. py:function:: hiprandGetVersion()

   Returns the version number of the cuRAND or rocRAND library.

   Returns in ``version`` the version number of the underlying cuRAND or
   rocRAND library.

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``version`` is NULL 

           - HIPRAND_STATUS_SUCCESS if the version number was successfully returned
       * :py:obj:`~.int`:
               Version of the library

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGetVersion(int * version)


.. py:function:: hiprandCreatePoissonDistribution(lambda_)

   Construct the histogram for a Poisson distribution.

   Construct the histogram for the Poisson distribution with lambda ``lambda.``

   Args:
       lambda (:py:obj:`~.float`/:py:obj:`~.int`):
           lambda for the Poisson distribution

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated 

           - HIPRAND_STATUS_OUT_OF_RANGE if ``discrete_distribution`` pointer was null 

           - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive 

           - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully
       * :py:obj:`~.rocrand_discrete_distribution_st`:
               pointer to the histogram in device memory

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandCreatePoissonDistribution(double lambda, hiprandDiscreteDistribution_t * discrete_distribution)


.. py:function:: hiprandDestroyDistribution(discrete_distribution)

   Destroy the histogram array for a discrete distribution.

   Destroy the histogram array for a discrete distribution created by
   hiprandCreatePoissonDistribution.

   Args:
       discrete_distribution (:py:obj:`~.rocrand_discrete_distribution_st`/:py:obj:`~.object`):
           pointer to the histogram in device memory

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``discrete_distribution`` was null 

           - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution)


.. py:function:: hiprandGetDirectionVectors32(vectors, set)

   Retrieves the Sobol 32 direction vector array specified by ``set.``

   Args:
       vectors (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to the Sobol 32 direction vector array.

       set (:py:obj:`~.hiprandDirectionVectorSet`):
           Specifies which hipRAND vector set for quasirandom generators to retrieve.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``set`` is invalid 

           - HIPRAND_STATUS_SUCCESS if ``vectors`` was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGetDirectionVectors32(hiprandDirectionVectors32_t ** vectors, hiprandDirectionVectorSet_t set)


.. py:function:: hiprandGetDirectionVectors64(vectors, set)

   Retrieves the Sobol 64 direction vector array specified by ``set.``

   Args:
       vectors (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to the Sobol 64 direction vector array.

       set (:py:obj:`~.hiprandDirectionVectorSet`):
           Specifies which hipRAND vector set for quasirandom generators to retrieve.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``set`` is invalid 

           - HIPRAND_STATUS_SUCCESS if ``vectors`` was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGetDirectionVectors64(hiprandDirectionVectors64_t ** vectors, hiprandDirectionVectorSet_t set)


.. py:function:: hiprandGetScrambleConstants32(constants)

   Retrieves the scramble constants for 32-bit scrambled Sobol generation.

   Args:
       constants (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to the constants pointer.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_SUCCESS if the pointer was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGetScrambleConstants32(const unsigned int ** constants)


.. py:function:: hiprandGetScrambleConstants64(constants)

   Retrieves the scramble constants for 64-bit scrambled Sobol generation.

   Args:
       constants (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Pointer to the constants pointer.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hiprandStatus`: HIPRAND_STATUS_SUCCESS if the pointer was set successfully

   .. rubric:: C signature

   .. code-block:: c

       hiprandStatus_t hiprandGetScrambleConstants64(const unsigned long long ** constants)


