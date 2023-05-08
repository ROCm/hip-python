# AMD_COPYRIGHT
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libhiprand.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _rocrand_create_generator__funptr = NULL
# \brief Creates a new random number generator.
# Creates a new pseudo random number generator of type \p rng_type
# and returns it in \p generator.
# Values for \p rng_type are:
# - ROCRAND_RNG_PSEUDO_XORWOW
# - ROCRAND_RNG_PSEUDO_MRG31K3P
# - ROCRAND_RNG_PSEUDO_MRG32K3A
# - ROCRAND_RNG_PSEUDO_MTGP32
# - ROCRAND_RNG_PSEUDO_PHILOX4_32_10
# - ROCRAND_RNG_PSEUDO_LFSR113
# - ROCRAND_RNG_QUASI_SOBOL32
# - ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
# - ROCRAND_RNG_QUASI_SOBOL64
# - ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64
# \param generator - Pointer to generator
# \param rng_type - Type of generator to create
# \return
# - ROCRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \n
# - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
# dynamically linked library version \n
# - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
# - ROCRAND_STATUS_SUCCESS if generator was created successfully \n
cdef rocrand_status rocrand_create_generator(rocrand_generator* generator,rocrand_rng_type rng_type) nogil:
    global _rocrand_create_generator__funptr
    __init_symbol(&_rocrand_create_generator__funptr,"rocrand_create_generator")
    return (<rocrand_status (*)(rocrand_generator*,rocrand_rng_type) nogil> _rocrand_create_generator__funptr)(generator,rng_type)


cdef void* _rocrand_destroy_generator__funptr = NULL
# \brief Destroys random number generator.
# Destroys random number generator and frees related memory.
# \param generator - Generator to be destroyed
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_SUCCESS if generator was destroyed successfully \n
cdef rocrand_status rocrand_destroy_generator(rocrand_generator generator) nogil:
    global _rocrand_destroy_generator__funptr
    __init_symbol(&_rocrand_destroy_generator__funptr,"rocrand_destroy_generator")
    return (<rocrand_status (*)(rocrand_generator) nogil> _rocrand_destroy_generator__funptr)(generator)


cdef void* _rocrand_generate__funptr = NULL
# \brief Generates uniformly distributed 32-bit unsigned integers.
# Generates \p n uniformly distributed 32-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^32, including \p 0 and
# excluding \p 2^32.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate(rocrand_generator generator,unsigned int * output_data,unsigned long n) nogil:
    global _rocrand_generate__funptr
    __init_symbol(&_rocrand_generate__funptr,"rocrand_generate")
    return (<rocrand_status (*)(rocrand_generator,unsigned int *,unsigned long) nogil> _rocrand_generate__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_long_long__funptr = NULL
# \brief Generates uniformly distributed 64-bit unsigned integers.
# Generates \p n uniformly distributed 64-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^64, including \p 0 and
# excluding \p 2^64.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 64-bit unsigned integers to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_TYPE_ERROR if the generator can't natively generate 64-bit random numbers \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_long_long(rocrand_generator generator,unsigned long long * output_data,unsigned long n) nogil:
    global _rocrand_generate_long_long__funptr
    __init_symbol(&_rocrand_generate_long_long__funptr,"rocrand_generate_long_long")
    return (<rocrand_status (*)(rocrand_generator,unsigned long long *,unsigned long) nogil> _rocrand_generate_long_long__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_char__funptr = NULL
# \brief Generates uniformly distributed 8-bit unsigned integers.
# Generates \p n uniformly distributed 8-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^8, including \p 0 and
# excluding \p 2^8.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 8-bit unsigned integers to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_char(rocrand_generator generator,unsigned char * output_data,unsigned long n) nogil:
    global _rocrand_generate_char__funptr
    __init_symbol(&_rocrand_generate_char__funptr,"rocrand_generate_char")
    return (<rocrand_status (*)(rocrand_generator,unsigned char *,unsigned long) nogil> _rocrand_generate_char__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_short__funptr = NULL
# \brief Generates uniformly distributed 16-bit unsigned integers.
# Generates \p n uniformly distributed 16-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^16, including \p 0 and
# excluding \p 2^16.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 16-bit unsigned integers to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_short(rocrand_generator generator,unsigned short * output_data,unsigned long n) nogil:
    global _rocrand_generate_short__funptr
    __init_symbol(&_rocrand_generate_short__funptr,"rocrand_generate_short")
    return (<rocrand_status (*)(rocrand_generator,unsigned short *,unsigned long) nogil> _rocrand_generate_short__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_uniform__funptr = NULL
# \brief Generates uniformly distributed \p float values.
# Generates \p n uniformly distributed 32-bit floating-point values
# and saves them to \p output_data.
# Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
# including \p 1.0f.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>float</tt>s to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_uniform(rocrand_generator generator,float * output_data,unsigned long n) nogil:
    global _rocrand_generate_uniform__funptr
    __init_symbol(&_rocrand_generate_uniform__funptr,"rocrand_generate_uniform")
    return (<rocrand_status (*)(rocrand_generator,float *,unsigned long) nogil> _rocrand_generate_uniform__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_uniform_double__funptr = NULL
# \brief Generates uniformly distributed double-precision floating-point values.
# Generates \p n uniformly distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>double</tt>s to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_uniform_double(rocrand_generator generator,double * output_data,unsigned long n) nogil:
    global _rocrand_generate_uniform_double__funptr
    __init_symbol(&_rocrand_generate_uniform_double__funptr,"rocrand_generate_uniform_double")
    return (<rocrand_status (*)(rocrand_generator,double *,unsigned long) nogil> _rocrand_generate_uniform_double__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_uniform_half__funptr = NULL
# \brief Generates uniformly distributed half-precision floating-point values.
# Generates \p n uniformly distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>half</tt>s to generate
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_uniform_half(rocrand_generator generator,int * output_data,unsigned long n) nogil:
    global _rocrand_generate_uniform_half__funptr
    __init_symbol(&_rocrand_generate_uniform_half__funptr,"rocrand_generate_uniform_half")
    return (<rocrand_status (*)(rocrand_generator,int *,unsigned long) nogil> _rocrand_generate_uniform_half__funptr)(generator,output_data,n)


cdef void* _rocrand_generate_normal__funptr = NULL
# \brief Generates normally distributed \p float values.
# Generates \p n normally distributed distributed 32-bit floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>float</tt>s to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_normal(rocrand_generator generator,float * output_data,unsigned long n,float mean,float stddev) nogil:
    global _rocrand_generate_normal__funptr
    __init_symbol(&_rocrand_generate_normal__funptr,"rocrand_generate_normal")
    return (<rocrand_status (*)(rocrand_generator,float *,unsigned long,float,float) nogil> _rocrand_generate_normal__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_normal_double__funptr = NULL
# \brief Generates normally distributed \p double values.
# Generates \p n normally distributed 64-bit double-precision floating-point
# numbers and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>double</tt>s to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_normal_double(rocrand_generator generator,double * output_data,unsigned long n,double mean,double stddev) nogil:
    global _rocrand_generate_normal_double__funptr
    __init_symbol(&_rocrand_generate_normal_double__funptr,"rocrand_generate_normal_double")
    return (<rocrand_status (*)(rocrand_generator,double *,unsigned long,double,double) nogil> _rocrand_generate_normal_double__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_normal_half__funptr = NULL
# \brief Generates normally distributed \p half values.
# Generates \p n normally distributed 16-bit half-precision floating-point
# numbers and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>half</tt>s to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_normal_half(rocrand_generator generator,int * output_data,unsigned long n,int mean,int stddev) nogil:
    global _rocrand_generate_normal_half__funptr
    __init_symbol(&_rocrand_generate_normal_half__funptr,"rocrand_generate_normal_half")
    return (<rocrand_status (*)(rocrand_generator,int *,unsigned long,int,int) nogil> _rocrand_generate_normal_half__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_log_normal__funptr = NULL
# \brief Generates log-normally distributed \p float values.
# Generates \p n log-normally distributed 32-bit floating-point values
# and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>float</tt>s to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_log_normal(rocrand_generator generator,float * output_data,unsigned long n,float mean,float stddev) nogil:
    global _rocrand_generate_log_normal__funptr
    __init_symbol(&_rocrand_generate_log_normal__funptr,"rocrand_generate_log_normal")
    return (<rocrand_status (*)(rocrand_generator,float *,unsigned long,float,float) nogil> _rocrand_generate_log_normal__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_log_normal_double__funptr = NULL
# \brief Generates log-normally distributed \p double values.
# Generates \p n log-normally distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>double</tt>s to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_log_normal_double(rocrand_generator generator,double * output_data,unsigned long n,double mean,double stddev) nogil:
    global _rocrand_generate_log_normal_double__funptr
    __init_symbol(&_rocrand_generate_log_normal_double__funptr,"rocrand_generate_log_normal_double")
    return (<rocrand_status (*)(rocrand_generator,double *,unsigned long,double,double) nogil> _rocrand_generate_log_normal_double__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_log_normal_half__funptr = NULL
# \brief Generates log-normally distributed \p half values.
# Generates \p n log-normally distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of <tt>half</tt>s to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_log_normal_half(rocrand_generator generator,int * output_data,unsigned long n,int mean,int stddev) nogil:
    global _rocrand_generate_log_normal_half__funptr
    __init_symbol(&_rocrand_generate_log_normal_half__funptr,"rocrand_generate_log_normal_half")
    return (<rocrand_status (*)(rocrand_generator,int *,unsigned long,int,int) nogil> _rocrand_generate_log_normal_half__funptr)(generator,output_data,n,mean,stddev)


cdef void* _rocrand_generate_poisson__funptr = NULL
# \brief Generates Poisson-distributed 32-bit unsigned integers.
# Generates \p n Poisson-distributed 32-bit unsigned integers and
# saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# \param lambda - lambda for the Poisson distribution
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef rocrand_status rocrand_generate_poisson(rocrand_generator generator,unsigned int * output_data,unsigned long n,double lambda_) nogil:
    global _rocrand_generate_poisson__funptr
    __init_symbol(&_rocrand_generate_poisson__funptr,"rocrand_generate_poisson")
    return (<rocrand_status (*)(rocrand_generator,unsigned int *,unsigned long,double) nogil> _rocrand_generate_poisson__funptr)(generator,output_data,n,lambda_)


cdef void* _rocrand_initialize_generator__funptr = NULL
# \brief Initializes the generator's state on GPU or host.
# Initializes the generator's state on GPU or host. User it not
# required to call this function before using a generator.
# If rocrand_initialize() was not called for a generator, it will be
# automatically called by functions which generates random numbers like
# rocrand_generate(), rocrand_generate_uniform() etc.
# \param generator - Generator to initialize
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
# - ROCRAND_STATUS_SUCCESS if the seeds were generated successfully \n
cdef rocrand_status rocrand_initialize_generator(rocrand_generator generator) nogil:
    global _rocrand_initialize_generator__funptr
    __init_symbol(&_rocrand_initialize_generator__funptr,"rocrand_initialize_generator")
    return (<rocrand_status (*)(rocrand_generator) nogil> _rocrand_initialize_generator__funptr)(generator)


cdef void* _rocrand_set_stream__funptr = NULL
# \brief Sets the current stream for kernel launches.
# Sets the current stream for all kernel launches of the generator.
# All functions will use this stream.
# \param generator - Generator to modify
# \param stream - Stream to use or NULL for default stream
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_SUCCESS if stream was set successfully \n
cdef rocrand_status rocrand_set_stream(rocrand_generator generator,hipStream_t stream) nogil:
    global _rocrand_set_stream__funptr
    __init_symbol(&_rocrand_set_stream__funptr,"rocrand_set_stream")
    return (<rocrand_status (*)(rocrand_generator,hipStream_t) nogil> _rocrand_set_stream__funptr)(generator,stream)


cdef void* _rocrand_set_seed__funptr = NULL
# \brief Sets the seed of a pseudo-random number generator.
# Sets the seed of the pseudo-random number generator.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# For an MRG32K3a or MRG31K3p generator the seed value can't be zero. If \p seed is
# equal to zero and generator's type is ROCRAND_RNG_PSEUDO_MRG32K3A or ROCRAND_RNG_PSEUDO_MRG31K3P,
# value \p 12345 is used as seed instead.
# For a LFSR113 generator seed values must be larger than 1, 7, 15,
# 127. The \p seed upper and lower 32 bits used as first and
# second seed value. If those values smaller than 2 and/or 8, those
# are increased with 1 and/or 7.
# \param generator - Pseudo-random number generator
# \param seed - New seed value
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator \n
# - ROCRAND_STATUS_SUCCESS if seed was set successfully \n
cdef rocrand_status rocrand_set_seed(rocrand_generator generator,unsigned long long seed) nogil:
    global _rocrand_set_seed__funptr
    __init_symbol(&_rocrand_set_seed__funptr,"rocrand_set_seed")
    return (<rocrand_status (*)(rocrand_generator,unsigned long long) nogil> _rocrand_set_seed__funptr)(generator,seed)


cdef void* _rocrand_set_seed_uint4__funptr = NULL
# \brief Sets the seeds of a pseudo-random number generator.
# Sets the seed of the pseudo-random number generator. Currently only for LFSR113
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# Only usable for LFSR113.
# For a LFSR113 generator seed values must be bigger than 1, 7, 15,
# 127. If those values smaller, than the requested minimum values [2, 8, 16, 128], then
# it will be increased with the minimum values minus 1 [1, 7, 15, 127].
# \param generator - Pseudo-random number generator
# \param seed - New seed value
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator \n
# - ROCRAND_STATUS_SUCCESS if seed was set successfully \n
cdef rocrand_status rocrand_set_seed_uint4(rocrand_generator generator,uint4 seed) nogil:
    global _rocrand_set_seed_uint4__funptr
    __init_symbol(&_rocrand_set_seed_uint4__funptr,"rocrand_set_seed_uint4")
    return (<rocrand_status (*)(rocrand_generator,uint4) nogil> _rocrand_set_seed_uint4__funptr)(generator,seed)


cdef void* _rocrand_set_offset__funptr = NULL
# \brief Sets the offset of a random number generator.
# Sets the absolute offset of the random number generator.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's seed.
# Absolute offset cannot be set if generator's type is ROCRAND_RNG_PSEUDO_MTGP32 or
# ROCRAND_RNG_PSEUDO_LFSR113.
# \param generator - Random number generator
# \param offset - New absolute offset
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_SUCCESS if offset was successfully set \n
# - ROCRAND_STATUS_TYPE_ERROR if generator's type is ROCRAND_RNG_PSEUDO_MTGP32 or
# ROCRAND_RNG_PSEUDO_LFSR113
cdef rocrand_status rocrand_set_offset(rocrand_generator generator,unsigned long long offset) nogil:
    global _rocrand_set_offset__funptr
    __init_symbol(&_rocrand_set_offset__funptr,"rocrand_set_offset")
    return (<rocrand_status (*)(rocrand_generator,unsigned long long) nogil> _rocrand_set_offset__funptr)(generator,offset)


cdef void* _rocrand_set_quasi_random_generator_dimensions__funptr = NULL
# \brief Set the number of dimensions of a quasi-random number generator.
# Set the number of dimensions of a quasi-random number generator.
# Supported values of \p dimensions are 1 to 20000.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# \param generator - Quasi-random number generator
# \param dimensions - Number of dimensions
# \return
# - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - ROCRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
# - ROCRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
# - ROCRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
cdef rocrand_status rocrand_set_quasi_random_generator_dimensions(rocrand_generator generator,unsigned int dimensions) nogil:
    global _rocrand_set_quasi_random_generator_dimensions__funptr
    __init_symbol(&_rocrand_set_quasi_random_generator_dimensions__funptr,"rocrand_set_quasi_random_generator_dimensions")
    return (<rocrand_status (*)(rocrand_generator,unsigned int) nogil> _rocrand_set_quasi_random_generator_dimensions__funptr)(generator,dimensions)


cdef void* _rocrand_get_version__funptr = NULL
# \brief Returns the version number of the library.
# Returns in \p version the version number of the dynamically linked
# rocRAND library.
# \param version - Version of the library
# \return
# - ROCRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
# - ROCRAND_STATUS_SUCCESS if the version number was successfully returned \n
cdef rocrand_status rocrand_get_version(int * version) nogil:
    global _rocrand_get_version__funptr
    __init_symbol(&_rocrand_get_version__funptr,"rocrand_get_version")
    return (<rocrand_status (*)(int *) nogil> _rocrand_get_version__funptr)(version)


cdef void* _rocrand_create_poisson_distribution__funptr = NULL
# \brief Construct the histogram for a Poisson distribution.
# Construct the histogram for the Poisson distribution with lambda \p lambda.
# \param lambda - lambda for the Poisson distribution
# \param discrete_distribution - pointer to the histogram in device memory
# \return
# - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
# - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
# - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
cdef rocrand_status rocrand_create_poisson_distribution(double lambda_,rocrand_discrete_distribution* discrete_distribution) nogil:
    global _rocrand_create_poisson_distribution__funptr
    __init_symbol(&_rocrand_create_poisson_distribution__funptr,"rocrand_create_poisson_distribution")
    return (<rocrand_status (*)(double,rocrand_discrete_distribution*) nogil> _rocrand_create_poisson_distribution__funptr)(lambda_,discrete_distribution)


cdef void* _rocrand_create_discrete_distribution__funptr = NULL
# \brief Construct the histogram for a custom discrete distribution.
# Construct the histogram for the discrete distribution of \p size
# 32-bit unsigned integers from the range [\p offset, \p offset + \p size)
# using \p probabilities as probabilities.
# \param probabilities - probabilities of the the distribution in host memory
# \param size - size of \p probabilities
# \param offset - offset of values
# \param discrete_distribution - pointer to the histogram in device memory
# \return
# - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
# - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
# - ROCRAND_STATUS_OUT_OF_RANGE if \p size was zero \n
# - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
cdef rocrand_status rocrand_create_discrete_distribution(const double * probabilities,unsigned int size,unsigned int offset,rocrand_discrete_distribution* discrete_distribution) nogil:
    global _rocrand_create_discrete_distribution__funptr
    __init_symbol(&_rocrand_create_discrete_distribution__funptr,"rocrand_create_discrete_distribution")
    return (<rocrand_status (*)(const double *,unsigned int,unsigned int,rocrand_discrete_distribution*) nogil> _rocrand_create_discrete_distribution__funptr)(probabilities,size,offset,discrete_distribution)


cdef void* _rocrand_destroy_discrete_distribution__funptr = NULL
# \brief Destroy the histogram array for a discrete distribution.
# Destroy the histogram array for a discrete distribution created by
# rocrand_create_poisson_distribution.
# \param discrete_distribution - pointer to the histogram in device memory
# \return
# - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
# - ROCRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
cdef rocrand_status rocrand_destroy_discrete_distribution(rocrand_discrete_distribution discrete_distribution) nogil:
    global _rocrand_destroy_discrete_distribution__funptr
    __init_symbol(&_rocrand_destroy_discrete_distribution__funptr,"rocrand_destroy_discrete_distribution")
    return (<rocrand_status (*)(rocrand_discrete_distribution) nogil> _rocrand_destroy_discrete_distribution__funptr)(discrete_distribution)


cdef void* _hiprandCreateGenerator__funptr = NULL
# \brief Creates a new random number generator.
# Creates a new random number generator of type \p rng_type,
# and returns it in \p generator. That generator will use
# GPU to create random numbers.
# Values for \p rng_type are:
# - HIPRAND_RNG_PSEUDO_DEFAULT
# - HIPRAND_RNG_PSEUDO_XORWOW
# - HIPRAND_RNG_PSEUDO_MRG32K3A
# - HIPRAND_RNG_PSEUDO_MTGP32
# - HIPRAND_RNG_PSEUDO_MT19937
# - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
# - HIPRAND_RNG_QUASI_DEFAULT
# - HIPRAND_RNG_QUASI_SOBOL32
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
# - HIPRAND_RNG_QUASI_SOBOL64
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
# \param generator - Pointer to generator
# \param rng_type - Type of random number generator to create
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
# - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
# - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
# dynamically linked library version \n
# - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
# - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type \p rng_type is not implemented yet \n
# - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
cdef hiprandStatus hiprandCreateGenerator(hiprandGenerator_t* generator,hiprandRngType rng_type) nogil:
    global _hiprandCreateGenerator__funptr
    __init_symbol(&_hiprandCreateGenerator__funptr,"hiprandCreateGenerator")
    return (<hiprandStatus (*)(hiprandGenerator_t*,hiprandRngType) nogil> _hiprandCreateGenerator__funptr)(generator,rng_type)


cdef void* _hiprandCreateGeneratorHost__funptr = NULL
# \brief Creates a new random number generator on host.
# Creates a new host random number generator of type \p rng_type
# and returns it in \p generator. Created generator will use
# host CPU to generate random numbers.
# Values for \p rng_type are:
# - HIPRAND_RNG_PSEUDO_DEFAULT
# - HIPRAND_RNG_PSEUDO_XORWOW
# - HIPRAND_RNG_PSEUDO_MRG32K3A
# - HIPRAND_RNG_PSEUDO_MTGP32
# - HIPRAND_RNG_PSEUDO_MT19937
# - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
# - HIPRAND_RNG_QUASI_DEFAULT
# - HIPRAND_RNG_QUASI_SOBOL32
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
# - HIPRAND_RNG_QUASI_SOBOL64
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
# \param generator - Pointer to generator
# \param rng_type - Type of random number generator to create
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
# - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
# dynamically linked library version \n
# - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
# - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type \p rng_type is not implemented yet \n
# - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
cdef hiprandStatus hiprandCreateGeneratorHost(hiprandGenerator_t* generator,hiprandRngType rng_type) nogil:
    global _hiprandCreateGeneratorHost__funptr
    __init_symbol(&_hiprandCreateGeneratorHost__funptr,"hiprandCreateGeneratorHost")
    return (<hiprandStatus (*)(hiprandGenerator_t*,hiprandRngType) nogil> _hiprandCreateGeneratorHost__funptr)(generator,rng_type)


cdef void* _hiprandDestroyGenerator__funptr = NULL
# \brief Destroys random number generator.
# Destroys random number generator and frees related memory.
# \param generator - Generator to be destroyed
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully \n
cdef hiprandStatus hiprandDestroyGenerator(hiprandGenerator_t generator) nogil:
    global _hiprandDestroyGenerator__funptr
    __init_symbol(&_hiprandDestroyGenerator__funptr,"hiprandDestroyGenerator")
    return (<hiprandStatus (*)(hiprandGenerator_t) nogil> _hiprandDestroyGenerator__funptr)(generator)


cdef void* _hiprandGenerate__funptr = NULL
# \brief Generates uniformly distributed 32-bit unsigned integers.
# Generates \p n uniformly distributed 32-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^32, including \p 0 and
# excluding \p 2^32.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerate(hiprandGenerator_t generator,unsigned int * output_data,unsigned long n) nogil:
    global _hiprandGenerate__funptr
    __init_symbol(&_hiprandGenerate__funptr,"hiprandGenerate")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned int *,unsigned long) nogil> _hiprandGenerate__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateChar__funptr = NULL
# \brief Generates uniformly distributed 8-bit unsigned integers.
# Generates \p n uniformly distributed 8-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^8, including \p 0 and
# excluding \p 2^8.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 8-bit unsigned integers to generate
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateChar(hiprandGenerator_t generator,unsigned char * output_data,unsigned long n) nogil:
    global _hiprandGenerateChar__funptr
    __init_symbol(&_hiprandGenerateChar__funptr,"hiprandGenerateChar")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned char *,unsigned long) nogil> _hiprandGenerateChar__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateShort__funptr = NULL
# \brief Generates uniformly distributed 16-bit unsigned integers.
# Generates \p n uniformly distributed 16-bit unsigned integers and
# saves them to \p output_data.
# Generated numbers are between \p 0 and \p 2^16, including \p 0 and
# excluding \p 2^16.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 16-bit unsigned integers to generate
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateShort(hiprandGenerator_t generator,unsigned short * output_data,unsigned long n) nogil:
    global _hiprandGenerateShort__funptr
    __init_symbol(&_hiprandGenerateShort__funptr,"hiprandGenerateShort")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned short *,unsigned long) nogil> _hiprandGenerateShort__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateUniform__funptr = NULL
# \brief Generates uniformly distributed floats.
# Generates \p n uniformly distributed 32-bit floating-point values
# and saves them to \p output_data.
# Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
# including \p 1.0f.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniform(hiprandGenerator_t generator,float * output_data,unsigned long n) nogil:
    global _hiprandGenerateUniform__funptr
    __init_symbol(&_hiprandGenerateUniform__funptr,"hiprandGenerateUniform")
    return (<hiprandStatus (*)(hiprandGenerator_t,float *,unsigned long) nogil> _hiprandGenerateUniform__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateUniformDouble__funptr = NULL
# \brief Generates uniformly distributed double-precision floating-point values.
# Generates \p n uniformly distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# Note: When \p generator is of type: \p HIPRAND_RNG_PSEUDO_MRG32K3A,
# \p HIPRAND_RNG_PSEUDO_MTGP32, or \p HIPRAND_RNG_QUASI_SOBOL32,
# then the returned \p double values are generated from only 32 random bits
# each (one <tt>unsigned int</tt> value per one generated \p double).
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniformDouble(hiprandGenerator_t generator,double * output_data,unsigned long n) nogil:
    global _hiprandGenerateUniformDouble__funptr
    __init_symbol(&_hiprandGenerateUniformDouble__funptr,"hiprandGenerateUniformDouble")
    return (<hiprandStatus (*)(hiprandGenerator_t,double *,unsigned long) nogil> _hiprandGenerateUniformDouble__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateUniformHalf__funptr = NULL
# \brief Generates uniformly distributed half-precision floating-point values.
# Generates \p n uniformly distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniformHalf(hiprandGenerator_t generator,int * output_data,unsigned long n) nogil:
    global _hiprandGenerateUniformHalf__funptr
    __init_symbol(&_hiprandGenerateUniformHalf__funptr,"hiprandGenerateUniformHalf")
    return (<hiprandStatus (*)(hiprandGenerator_t,int *,unsigned long) nogil> _hiprandGenerateUniformHalf__funptr)(generator,output_data,n)


cdef void* _hiprandGenerateNormal__funptr = NULL
# \brief Generates normally distributed floats.
# Generates \p n normally distributed 32-bit floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormal(hiprandGenerator_t generator,float * output_data,unsigned long n,float mean,float stddev) nogil:
    global _hiprandGenerateNormal__funptr
    __init_symbol(&_hiprandGenerateNormal__funptr,"hiprandGenerateNormal")
    return (<hiprandStatus (*)(hiprandGenerator_t,float *,unsigned long,float,float) nogil> _hiprandGenerateNormal__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGenerateNormalDouble__funptr = NULL
# \brief Generates normally distributed doubles.
# Generates \p n normally distributed 64-bit double-precision floating-point
# numbers and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of doubles to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormalDouble(hiprandGenerator_t generator,double * output_data,unsigned long n,double mean,double stddev) nogil:
    global _hiprandGenerateNormalDouble__funptr
    __init_symbol(&_hiprandGenerateNormalDouble__funptr,"hiprandGenerateNormalDouble")
    return (<hiprandStatus (*)(hiprandGenerator_t,double *,unsigned long,double,double) nogil> _hiprandGenerateNormalDouble__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGenerateNormalHalf__funptr = NULL
# \brief Generates normally distributed halfs.
# Generates \p n normally distributed 16-bit half-precision floating-point
# numbers and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormalHalf(hiprandGenerator_t generator,int * output_data,unsigned long n,int mean,int stddev) nogil:
    global _hiprandGenerateNormalHalf__funptr
    __init_symbol(&_hiprandGenerateNormalHalf__funptr,"hiprandGenerateNormalHalf")
    return (<hiprandStatus (*)(hiprandGenerator_t,int *,unsigned long,int,int) nogil> _hiprandGenerateNormalHalf__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGenerateLogNormal__funptr = NULL
# \brief Generates log-normally distributed floats.
# Generates \p n log-normally distributed 32-bit floating-point values
# and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormal(hiprandGenerator_t generator,float * output_data,unsigned long n,float mean,float stddev) nogil:
    global _hiprandGenerateLogNormal__funptr
    __init_symbol(&_hiprandGenerateLogNormal__funptr,"hiprandGenerateLogNormal")
    return (<hiprandStatus (*)(hiprandGenerator_t,float *,unsigned long,float,float) nogil> _hiprandGenerateLogNormal__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGenerateLogNormalDouble__funptr = NULL
# \brief Generates log-normally distributed doubles.
# Generates \p n log-normally distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of doubles to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormalDouble(hiprandGenerator_t generator,double * output_data,unsigned long n,double mean,double stddev) nogil:
    global _hiprandGenerateLogNormalDouble__funptr
    __init_symbol(&_hiprandGenerateLogNormalDouble__funptr,"hiprandGenerateLogNormalDouble")
    return (<hiprandStatus (*)(hiprandGenerator_t,double *,unsigned long,double,double) nogil> _hiprandGenerateLogNormalDouble__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGenerateLogNormalHalf__funptr = NULL
# \brief Generates log-normally distributed halfs.
# Generates \p n log-normally distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormalHalf(hiprandGenerator_t generator,int * output_data,unsigned long n,int mean,int stddev) nogil:
    global _hiprandGenerateLogNormalHalf__funptr
    __init_symbol(&_hiprandGenerateLogNormalHalf__funptr,"hiprandGenerateLogNormalHalf")
    return (<hiprandStatus (*)(hiprandGenerator_t,int *,unsigned long,int,int) nogil> _hiprandGenerateLogNormalHalf__funptr)(generator,output_data,n,mean,stddev)


cdef void* _hiprandGeneratePoisson__funptr = NULL
# \brief Generates Poisson-distributed 32-bit unsigned integers.
# Generates \p n Poisson-distributed 32-bit unsigned integers and
# saves them to \p output_data.
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# \param lambda - lambda for the Poisson distribution
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGeneratePoisson(hiprandGenerator_t generator,unsigned int * output_data,unsigned long n,double lambda_) nogil:
    global _hiprandGeneratePoisson__funptr
    __init_symbol(&_hiprandGeneratePoisson__funptr,"hiprandGeneratePoisson")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned int *,unsigned long,double) nogil> _hiprandGeneratePoisson__funptr)(generator,output_data,n,lambda_)


cdef void* _hiprandGenerateSeeds__funptr = NULL
# \brief Initializes the generator's state on GPU or host.
# Initializes the generator's state on GPU or host.
# If hiprandGenerateSeeds() was not called for a generator, it will be
# automatically called by functions which generates random numbers like
# hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.
# \param generator - Generator to initialize
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
# - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
# a previous kernel launch \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
# - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully \n
cdef hiprandStatus hiprandGenerateSeeds(hiprandGenerator_t generator) nogil:
    global _hiprandGenerateSeeds__funptr
    __init_symbol(&_hiprandGenerateSeeds__funptr,"hiprandGenerateSeeds")
    return (<hiprandStatus (*)(hiprandGenerator_t) nogil> _hiprandGenerateSeeds__funptr)(generator)


cdef void* _hiprandSetStream__funptr = NULL
# \brief Sets the current stream for kernel launches.
# Sets the current stream for all kernel launches of the generator.
# All functions will use this stream.
# \param generator - Generator to modify
# \param stream - Stream to use or NULL for default stream
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if stream was set successfully \n
cdef hiprandStatus hiprandSetStream(hiprandGenerator_t generator,hipStream_t stream) nogil:
    global _hiprandSetStream__funptr
    __init_symbol(&_hiprandSetStream__funptr,"hiprandSetStream")
    return (<hiprandStatus (*)(hiprandGenerator_t,hipStream_t) nogil> _hiprandSetStream__funptr)(generator,stream)


cdef void* _hiprandSetPseudoRandomGeneratorSeed__funptr = NULL
# \brief Sets the seed of a pseudo-random number generator.
# Sets the seed of the pseudo-random number generator.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# \param generator - Pseudo-random number generator
# \param seed - New seed value
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator \n
# - HIPRAND_STATUS_SUCCESS if seed was set successfully \n
cdef hiprandStatus hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator,unsigned long long seed) nogil:
    global _hiprandSetPseudoRandomGeneratorSeed__funptr
    __init_symbol(&_hiprandSetPseudoRandomGeneratorSeed__funptr,"hiprandSetPseudoRandomGeneratorSeed")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned long long) nogil> _hiprandSetPseudoRandomGeneratorSeed__funptr)(generator,seed)


cdef void* _hiprandSetGeneratorOffset__funptr = NULL
# \brief Sets the offset of a random number generator.
# Sets the absolute offset of the random number generator.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's seed.
# Absolute offset cannot be set if generator's type is
# HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.
# \param generator - Random number generator
# \param offset - New absolute offset
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if offset was successfully set \n
# - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
# or HIPRAND_RNG_PSEUDO_MT19937 \n
cdef hiprandStatus hiprandSetGeneratorOffset(hiprandGenerator_t generator,unsigned long long offset) nogil:
    global _hiprandSetGeneratorOffset__funptr
    __init_symbol(&_hiprandSetGeneratorOffset__funptr,"hiprandSetGeneratorOffset")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned long long) nogil> _hiprandSetGeneratorOffset__funptr)(generator,offset)


cdef void* _hiprandSetQuasiRandomGeneratorDimensions__funptr = NULL
# \brief Set the number of dimensions of a quasi-random number generator.
# Set the number of dimensions of a quasi-random number generator.
# Supported values of \p dimensions are 1 to 20000.
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# \param generator - Quasi-random number generator
# \param dimensions - Number of dimensions
# \return
# - HIPRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
# - HIPRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
# - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
cdef hiprandStatus hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator,unsigned int dimensions) nogil:
    global _hiprandSetQuasiRandomGeneratorDimensions__funptr
    __init_symbol(&_hiprandSetQuasiRandomGeneratorDimensions__funptr,"hiprandSetQuasiRandomGeneratorDimensions")
    return (<hiprandStatus (*)(hiprandGenerator_t,unsigned int) nogil> _hiprandSetQuasiRandomGeneratorDimensions__funptr)(generator,dimensions)


cdef void* _hiprandGetVersion__funptr = NULL
# \brief Returns the version number of the cuRAND or rocRAND library.
# Returns in \p version the version number of the underlying cuRAND or
# rocRAND library.
# \param version - Version of the library
# \return
# - HIPRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
# - HIPRAND_STATUS_SUCCESS if the version number was successfully returned \n
cdef hiprandStatus hiprandGetVersion(int * version) nogil:
    global _hiprandGetVersion__funptr
    __init_symbol(&_hiprandGetVersion__funptr,"hiprandGetVersion")
    return (<hiprandStatus (*)(int *) nogil> _hiprandGetVersion__funptr)(version)


cdef void* _hiprandCreatePoissonDistribution__funptr = NULL
# \brief Construct the histogram for a Poisson distribution.
# Construct the histogram for the Poisson distribution with lambda \p lambda.
# \param lambda - lambda for the Poisson distribution
# \param discrete_distribution - pointer to the histogram in device memory
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
# - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
# - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
cdef hiprandStatus hiprandCreatePoissonDistribution(double lambda_,hiprandDiscreteDistribution_t* discrete_distribution) nogil:
    global _hiprandCreatePoissonDistribution__funptr
    __init_symbol(&_hiprandCreatePoissonDistribution__funptr,"hiprandCreatePoissonDistribution")
    return (<hiprandStatus (*)(double,hiprandDiscreteDistribution_t*) nogil> _hiprandCreatePoissonDistribution__funptr)(lambda_,discrete_distribution)


cdef void* _hiprandDestroyDistribution__funptr = NULL
# \brief Destroy the histogram array for a discrete distribution.
# Destroy the histogram array for a discrete distribution created by
# hiprandCreatePoissonDistribution.
# \param discrete_distribution - pointer to the histogram in device memory
# \return
# - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
# - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
cdef hiprandStatus hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution) nogil:
    global _hiprandDestroyDistribution__funptr
    __init_symbol(&_hiprandDestroyDistribution__funptr,"hiprandDestroyDistribution")
    return (<hiprandStatus (*)(hiprandDiscreteDistribution_t) nogil> _hiprandDestroyDistribution__funptr)(discrete_distribution)
