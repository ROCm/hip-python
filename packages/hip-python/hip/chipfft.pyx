# AMD_COPYRIGHT
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libhipfft.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _hipfftPlan1d__funptr = NULL
# ! @brief Create a new one-dimensional FFT plan.
# @details Allocate and initialize a new one-dimensional FFT plan.
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx FFT length.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to compute.
cdef hipfftResult_t hipfftPlan1d(hipfftHandle* plan,int nx,hipfftType_t type,int batch) nogil:
    global _hipfftPlan1d__funptr
    __init_symbol(&_hipfftPlan1d__funptr,"hipfftPlan1d")
    return (<hipfftResult_t (*)(hipfftHandle*,int,hipfftType_t,int) nogil> _hipfftPlan1d__funptr)(plan,nx,type,batch)


cdef void* _hipfftPlan2d__funptr = NULL
# ! @brief Create a new two-dimensional FFT plan.
# @details Allocate and initialize a new two-dimensional FFT plan.
# Two-dimensional data should be stored in C ordering (row-major
# format), so that indexes in y-direction (j index) vary the
# fastest.
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx Number of elements in the x-direction (slow index).
# @param[in] ny Number of elements in the y-direction (fast index).
# @param[in] type FFT type.
cdef hipfftResult_t hipfftPlan2d(hipfftHandle* plan,int nx,int ny,hipfftType_t type) nogil:
    global _hipfftPlan2d__funptr
    __init_symbol(&_hipfftPlan2d__funptr,"hipfftPlan2d")
    return (<hipfftResult_t (*)(hipfftHandle*,int,int,hipfftType_t) nogil> _hipfftPlan2d__funptr)(plan,nx,ny,type)


cdef void* _hipfftPlan3d__funptr = NULL
# ! @brief Create a new three-dimensional FFT plan.
# @details Allocate and initialize a new three-dimensional FFT plan.
# Three-dimensional data should be stored in C ordering (row-major
# format), so that indexes in z-direction (k index) vary the
# fastest.
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] nx Number of elements in the x-direction (slowest index).
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction (fastest index).
# @param[in] type FFT type.
cdef hipfftResult_t hipfftPlan3d(hipfftHandle* plan,int nx,int ny,int nz,hipfftType_t type) nogil:
    global _hipfftPlan3d__funptr
    __init_symbol(&_hipfftPlan3d__funptr,"hipfftPlan3d")
    return (<hipfftResult_t (*)(hipfftHandle*,int,int,int,hipfftType_t) nogil> _hipfftPlan3d__funptr)(plan,nx,ny,nz,type)


cdef void* _hipfftPlanMany__funptr = NULL
# ! @brief Create a new batched rank-dimensional FFT plan with advanced data layout.
# @details Allocate and initialize a new batched rank-dimensional
# FFT plan. The number of elements to transform in each direction of
# the input data is specified in n.
# The batch parameter tells hipFFT how many transforms to perform. 
# The distance between the first elements of two consecutive batches 
# of the input and output data are specified with the idist and odist 
# parameters.
# The inembed and onembed parameters define the input and output data
# layouts. The number of elements in the data is assumed to be larger 
# than the number of elements in the transform. Strided data layouts 
# are also supported. Strides along the fastest direction in the input
# and output data are specified via the istride and ostride parameters.  
# If both inembed and onembed parameters are set to NULL, all the 
# advanced data layout parameters are ignored and reverted to default 
# values, i.e., the batched transform is performed with non-strided data
# access and the number of data/transform elements are assumed to be  
# equivalent.
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] rank Dimension of transform (1, 2, or 3).
# @param[in] n Number of elements to transform in the x/y/z directions.
# @param[in] inembed Number of elements in the input data in the x/y/z directions.
# @param[in] istride Distance between two successive elements in the input data.
# @param[in] idist Distance between input batches.
# @param[in] onembed Number of elements in the output data in the x/y/z directions.
# @param[in] ostride Distance between two successive elements in the output data.
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
cdef hipfftResult_t hipfftPlanMany(hipfftHandle* plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch) nogil:
    global _hipfftPlanMany__funptr
    __init_symbol(&_hipfftPlanMany__funptr,"hipfftPlanMany")
    return (<hipfftResult_t (*)(hipfftHandle*,int,int *,int *,int,int,int *,int,int,hipfftType_t,int) nogil> _hipfftPlanMany__funptr)(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch)


cdef void* _hipfftCreate__funptr = NULL
# ! @brief Allocate a new plan.
cdef hipfftResult_t hipfftCreate(hipfftHandle* plan) nogil:
    global _hipfftCreate__funptr
    __init_symbol(&_hipfftCreate__funptr,"hipfftCreate")
    return (<hipfftResult_t (*)(hipfftHandle*) nogil> _hipfftCreate__funptr)(plan)


cdef void* _hipfftExtPlanScaleFactor__funptr = NULL
# ! @brief Set scaling factor.
# @details hipFFT multiplies each element of the result by the given factor at the end of the transform.
# The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.
# This function must be called after the plan is allocated using
# ::hipfftCreate, but before the plan is initialized by any of the
# "MakePlan" functions.
cdef hipfftResult_t hipfftExtPlanScaleFactor(hipfftHandle plan,double scalefactor) nogil:
    global _hipfftExtPlanScaleFactor__funptr
    __init_symbol(&_hipfftExtPlanScaleFactor__funptr,"hipfftExtPlanScaleFactor")
    return (<hipfftResult_t (*)(hipfftHandle,double) nogil> _hipfftExtPlanScaleFactor__funptr)(plan,scalefactor)


cdef void* _hipfftMakePlan1d__funptr = NULL
# ! @brief Initialize a new one-dimensional FFT plan.
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# @param[in] plan Handle of the FFT plan.
# @param[in] nx FFT length.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to compute.
cdef hipfftResult_t hipfftMakePlan1d(hipfftHandle plan,int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftMakePlan1d__funptr
    __init_symbol(&_hipfftMakePlan1d__funptr,"hipfftMakePlan1d")
    return (<hipfftResult_t (*)(hipfftHandle,int,hipfftType_t,int,unsigned long *) nogil> _hipfftMakePlan1d__funptr)(plan,nx,type,batch,workSize)


cdef void* _hipfftMakePlan2d__funptr = NULL
# ! @brief Initialize a new two-dimensional FFT plan.
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# Two-dimensional data should be stored in C ordering (row-major
# format), so that indexes in y-direction (j index) vary the
# fastest.
# @param[in] plan Handle of the FFT plan.
# @param[in] nx Number of elements in the x-direction (slow index).
# @param[in] ny Number of elements in the y-direction (fast index).
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlan2d(hipfftHandle plan,int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftMakePlan2d__funptr
    __init_symbol(&_hipfftMakePlan2d__funptr,"hipfftMakePlan2d")
    return (<hipfftResult_t (*)(hipfftHandle,int,int,hipfftType_t,unsigned long *) nogil> _hipfftMakePlan2d__funptr)(plan,nx,ny,type,workSize)


cdef void* _hipfftMakePlan3d__funptr = NULL
# ! @brief Initialize a new two-dimensional FFT plan.
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle.
# Three-dimensional data should be stored in C ordering (row-major
# format), so that indexes in z-direction (k index) vary the
# fastest.
# @param[in] plan Handle of the FFT plan.
# @param[in] nx Number of elements in the x-direction (slowest index).
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction (fastest index).
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlan3d(hipfftHandle plan,int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftMakePlan3d__funptr
    __init_symbol(&_hipfftMakePlan3d__funptr,"hipfftMakePlan3d")
    return (<hipfftResult_t (*)(hipfftHandle,int,int,int,hipfftType_t,unsigned long *) nogil> _hipfftMakePlan3d__funptr)(plan,nx,ny,nz,type,workSize)


cdef void* _hipfftMakePlanMany__funptr = NULL
# ! @brief Initialize a new batched rank-dimensional FFT plan with advanced data layout.
# @details Assumes that the plan has been created already, and
# modifies the plan associated with the plan handle. The number 
# of elements to transform in each direction of the input data 
# in the FFT plan is specified in n.
# The batch parameter tells hipFFT how many transforms to perform. 
# The distance between the first elements of two consecutive batches 
# of the input and output data are specified with the idist and odist 
# parameters.
# The inembed and onembed parameters define the input and output data
# layouts. The number of elements in the data is assumed to be larger 
# than the number of elements in the transform. Strided data layouts 
# are also supported. Strides along the fastest direction in the input
# and output data are specified via the istride and ostride parameters.  
# If both inembed and onembed parameters are set to NULL, all the 
# advanced data layout parameters are ignored and reverted to default 
# values, i.e., the batched transform is performed with non-strided data
# access and the number of data/transform elements are assumed to be  
# equivalent.
# @param[out] plan Pointer to the FFT plan handle.
# @param[in] rank Dimension of transform (1, 2, or 3).
# @param[in] n Number of elements to transform in the x/y/z directions.
# @param[in] inembed Number of elements in the input data in the x/y/z directions.
# @param[in] istride Distance between two successive elements in the input data.
# @param[in] idist Distance between input batches.
# @param[in] onembed Number of elements in the output data in the x/y/z directions.
# @param[in] ostride Distance between two successive elements in the output data.
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftMakePlanMany(hipfftHandle plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftMakePlanMany__funptr
    __init_symbol(&_hipfftMakePlanMany__funptr,"hipfftMakePlanMany")
    return (<hipfftResult_t (*)(hipfftHandle,int,int *,int *,int,int,int *,int,int,hipfftType_t,int,unsigned long *) nogil> _hipfftMakePlanMany__funptr)(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize)


cdef void* _hipfftMakePlanMany64__funptr = NULL
cdef hipfftResult_t hipfftMakePlanMany64(hipfftHandle plan,int rank,long long * n,long long * inembed,long long istride,long long idist,long long * onembed,long long ostride,long long odist,hipfftType_t type,long long batch,unsigned long * workSize) nogil:
    global _hipfftMakePlanMany64__funptr
    __init_symbol(&_hipfftMakePlanMany64__funptr,"hipfftMakePlanMany64")
    return (<hipfftResult_t (*)(hipfftHandle,int,long long *,long long *,long long,long long,long long *,long long,long long,hipfftType_t,long long,unsigned long *) nogil> _hipfftMakePlanMany64__funptr)(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize)


cdef void* _hipfftEstimate1d__funptr = NULL
# ! @brief Return an estimate of the work area size required for a 1D plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate1d(int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftEstimate1d__funptr
    __init_symbol(&_hipfftEstimate1d__funptr,"hipfftEstimate1d")
    return (<hipfftResult_t (*)(int,hipfftType_t,int,unsigned long *) nogil> _hipfftEstimate1d__funptr)(nx,type,batch,workSize)


cdef void* _hipfftEstimate2d__funptr = NULL
# ! @brief Return an estimate of the work area size required for a 2D plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate2d(int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftEstimate2d__funptr
    __init_symbol(&_hipfftEstimate2d__funptr,"hipfftEstimate2d")
    return (<hipfftResult_t (*)(int,int,hipfftType_t,unsigned long *) nogil> _hipfftEstimate2d__funptr)(nx,ny,type,workSize)


cdef void* _hipfftEstimate3d__funptr = NULL
# ! @brief Return an estimate of the work area size required for a 3D plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimate3d(int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftEstimate3d__funptr
    __init_symbol(&_hipfftEstimate3d__funptr,"hipfftEstimate3d")
    return (<hipfftResult_t (*)(int,int,int,hipfftType_t,unsigned long *) nogil> _hipfftEstimate3d__funptr)(nx,ny,nz,type,workSize)


cdef void* _hipfftEstimateMany__funptr = NULL
# ! @brief Return an estimate of the work area size required for a rank-dimensional plan.
# @param[in] rank Dimension of FFT transform (1, 2, or 3).
# @param[in] n Number of elements in the x/y/z directions.
# @param[in] inembed
# @param[in] istride
# @param[in] idist Distance between input batches.
# @param[in] onembed
# @param[in] ostride
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftEstimateMany(int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftEstimateMany__funptr
    __init_symbol(&_hipfftEstimateMany__funptr,"hipfftEstimateMany")
    return (<hipfftResult_t (*)(int,int *,int *,int,int,int *,int,int,hipfftType_t,int,unsigned long *) nogil> _hipfftEstimateMany__funptr)(rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize)


cdef void* _hipfftGetSize1d__funptr = NULL
# ! @brief Return size of the work area size required for a 1D plan.
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize1d(hipfftHandle plan,int nx,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftGetSize1d__funptr
    __init_symbol(&_hipfftGetSize1d__funptr,"hipfftGetSize1d")
    return (<hipfftResult_t (*)(hipfftHandle,int,hipfftType_t,int,unsigned long *) nogil> _hipfftGetSize1d__funptr)(plan,nx,type,batch,workSize)


cdef void* _hipfftGetSize2d__funptr = NULL
# ! @brief Return size of the work area size required for a 2D plan.
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize2d(hipfftHandle plan,int nx,int ny,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftGetSize2d__funptr
    __init_symbol(&_hipfftGetSize2d__funptr,"hipfftGetSize2d")
    return (<hipfftResult_t (*)(hipfftHandle,int,int,hipfftType_t,unsigned long *) nogil> _hipfftGetSize2d__funptr)(plan,nx,ny,type,workSize)


cdef void* _hipfftGetSize3d__funptr = NULL
# ! @brief Return size of the work area size required for a 3D plan.
# @param[in] plan Pointer to the FFT plan.
# @param[in] nx Number of elements in the x-direction.
# @param[in] ny Number of elements in the y-direction.
# @param[in] nz Number of elements in the z-direction.
# @param[in] type FFT type.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSize3d(hipfftHandle plan,int nx,int ny,int nz,hipfftType_t type,unsigned long * workSize) nogil:
    global _hipfftGetSize3d__funptr
    __init_symbol(&_hipfftGetSize3d__funptr,"hipfftGetSize3d")
    return (<hipfftResult_t (*)(hipfftHandle,int,int,int,hipfftType_t,unsigned long *) nogil> _hipfftGetSize3d__funptr)(plan,nx,ny,nz,type,workSize)


cdef void* _hipfftGetSizeMany__funptr = NULL
# ! @brief Return size of the work area size required for a rank-dimensional plan.
# @param[in] plan Pointer to the FFT plan.
# @param[in] rank Dimension of FFT transform (1, 2, or 3).
# @param[in] n Number of elements in the x/y/z directions.
# @param[in] inembed
# @param[in] istride
# @param[in] idist Distance between input batches.
# @param[in] onembed
# @param[in] ostride
# @param[in] odist Distance between output batches.
# @param[in] type FFT type.
# @param[in] batch Number of batched transforms to perform.
# @param[out] workSize Pointer to work area size (returned value).
cdef hipfftResult_t hipfftGetSizeMany(hipfftHandle plan,int rank,int * n,int * inembed,int istride,int idist,int * onembed,int ostride,int odist,hipfftType_t type,int batch,unsigned long * workSize) nogil:
    global _hipfftGetSizeMany__funptr
    __init_symbol(&_hipfftGetSizeMany__funptr,"hipfftGetSizeMany")
    return (<hipfftResult_t (*)(hipfftHandle,int,int *,int *,int,int,int *,int,int,hipfftType_t,int,unsigned long *) nogil> _hipfftGetSizeMany__funptr)(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize)


cdef void* _hipfftGetSizeMany64__funptr = NULL
cdef hipfftResult_t hipfftGetSizeMany64(hipfftHandle plan,int rank,long long * n,long long * inembed,long long istride,long long idist,long long * onembed,long long ostride,long long odist,hipfftType_t type,long long batch,unsigned long * workSize) nogil:
    global _hipfftGetSizeMany64__funptr
    __init_symbol(&_hipfftGetSizeMany64__funptr,"hipfftGetSizeMany64")
    return (<hipfftResult_t (*)(hipfftHandle,int,long long *,long long *,long long,long long,long long *,long long,long long,hipfftType_t,long long,unsigned long *) nogil> _hipfftGetSizeMany64__funptr)(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize)


cdef void* _hipfftGetSize__funptr = NULL
# ! @brief Return size of the work area size required for a rank-dimensional plan.
# @param[in] plan Pointer to the FFT plan.
cdef hipfftResult_t hipfftGetSize(hipfftHandle plan,unsigned long * workSize) nogil:
    global _hipfftGetSize__funptr
    __init_symbol(&_hipfftGetSize__funptr,"hipfftGetSize")
    return (<hipfftResult_t (*)(hipfftHandle,unsigned long *) nogil> _hipfftGetSize__funptr)(plan,workSize)


cdef void* _hipfftSetAutoAllocation__funptr = NULL
# ! @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
# @param[in] plan Pointer to the FFT plan.
# @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
cdef hipfftResult_t hipfftSetAutoAllocation(hipfftHandle plan,int autoAllocate) nogil:
    global _hipfftSetAutoAllocation__funptr
    __init_symbol(&_hipfftSetAutoAllocation__funptr,"hipfftSetAutoAllocation")
    return (<hipfftResult_t (*)(hipfftHandle,int) nogil> _hipfftSetAutoAllocation__funptr)(plan,autoAllocate)


cdef void* _hipfftSetWorkArea__funptr = NULL
# ! @brief Set the plan's work area.
# @param[in] plan Pointer to the FFT plan.
# @param[in] workArea Pointer to the work area (on device).
cdef hipfftResult_t hipfftSetWorkArea(hipfftHandle plan,void * workArea) nogil:
    global _hipfftSetWorkArea__funptr
    __init_symbol(&_hipfftSetWorkArea__funptr,"hipfftSetWorkArea")
    return (<hipfftResult_t (*)(hipfftHandle,void *) nogil> _hipfftSetWorkArea__funptr)(plan,workArea)


cdef void* _hipfftExecC2C__funptr = NULL
# ! @brief Execute a (float) complex-to-complex FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
# @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
cdef hipfftResult_t hipfftExecC2C(hipfftHandle plan,float2 * idata,float2 * odata,int direction) nogil:
    global _hipfftExecC2C__funptr
    __init_symbol(&_hipfftExecC2C__funptr,"hipfftExecC2C")
    return (<hipfftResult_t (*)(hipfftHandle,float2 *,float2 *,int) nogil> _hipfftExecC2C__funptr)(plan,idata,odata,direction)


cdef void* _hipfftExecR2C__funptr = NULL
# ! @brief Execute a (float) real-to-complex FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecR2C(hipfftHandle plan,float * idata,float2 * odata) nogil:
    global _hipfftExecR2C__funptr
    __init_symbol(&_hipfftExecR2C__funptr,"hipfftExecR2C")
    return (<hipfftResult_t (*)(hipfftHandle,float *,float2 *) nogil> _hipfftExecR2C__funptr)(plan,idata,odata)


cdef void* _hipfftExecC2R__funptr = NULL
# ! @brief Execute a (float) complex-to-real FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecC2R(hipfftHandle plan,float2 * idata,float * odata) nogil:
    global _hipfftExecC2R__funptr
    __init_symbol(&_hipfftExecC2R__funptr,"hipfftExecC2R")
    return (<hipfftResult_t (*)(hipfftHandle,float2 *,float *) nogil> _hipfftExecC2R__funptr)(plan,idata,odata)


cdef void* _hipfftExecZ2Z__funptr = NULL
# ! @brief Execute a (double) complex-to-complex FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
# @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
cdef hipfftResult_t hipfftExecZ2Z(hipfftHandle plan,double2 * idata,double2 * odata,int direction) nogil:
    global _hipfftExecZ2Z__funptr
    __init_symbol(&_hipfftExecZ2Z__funptr,"hipfftExecZ2Z")
    return (<hipfftResult_t (*)(hipfftHandle,double2 *,double2 *,int) nogil> _hipfftExecZ2Z__funptr)(plan,idata,odata,direction)


cdef void* _hipfftExecD2Z__funptr = NULL
# ! @brief Execute a (double) real-to-complex FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecD2Z(hipfftHandle plan,double * idata,double2 * odata) nogil:
    global _hipfftExecD2Z__funptr
    __init_symbol(&_hipfftExecD2Z__funptr,"hipfftExecD2Z")
    return (<hipfftResult_t (*)(hipfftHandle,double *,double2 *) nogil> _hipfftExecD2Z__funptr)(plan,idata,odata)


cdef void* _hipfftExecZ2D__funptr = NULL
# ! @brief Execute a (double) complex-to-real FFT.
# @details If the input and output buffers are equal, an in-place
# transform is performed.
# @param plan The FFT plan.
# @param idata Input data (on device).
# @param odata Output data (on device).
cdef hipfftResult_t hipfftExecZ2D(hipfftHandle plan,double2 * idata,double * odata) nogil:
    global _hipfftExecZ2D__funptr
    __init_symbol(&_hipfftExecZ2D__funptr,"hipfftExecZ2D")
    return (<hipfftResult_t (*)(hipfftHandle,double2 *,double *) nogil> _hipfftExecZ2D__funptr)(plan,idata,odata)


cdef void* _hipfftSetStream__funptr = NULL
# ! @brief Set HIP stream to execute plan on.
# @details Associates a HIP stream with a hipFFT plan.  All kernels
# launched by this plan are associated with the provided stream.
# @param plan The FFT plan.
# @param stream The HIP stream.
cdef hipfftResult_t hipfftSetStream(hipfftHandle plan,hipStream_t stream) nogil:
    global _hipfftSetStream__funptr
    __init_symbol(&_hipfftSetStream__funptr,"hipfftSetStream")
    return (<hipfftResult_t (*)(hipfftHandle,hipStream_t) nogil> _hipfftSetStream__funptr)(plan,stream)


cdef void* _hipfftDestroy__funptr = NULL
# ! @brief Destroy and deallocate an existing plan.
cdef hipfftResult_t hipfftDestroy(hipfftHandle plan) nogil:
    global _hipfftDestroy__funptr
    __init_symbol(&_hipfftDestroy__funptr,"hipfftDestroy")
    return (<hipfftResult_t (*)(hipfftHandle) nogil> _hipfftDestroy__funptr)(plan)


cdef void* _hipfftGetVersion__funptr = NULL
# ! @brief Get rocFFT/cuFFT version.
# @param[out] version cuFFT/rocFFT version (returned value).
cdef hipfftResult_t hipfftGetVersion(int * version) nogil:
    global _hipfftGetVersion__funptr
    __init_symbol(&_hipfftGetVersion__funptr,"hipfftGetVersion")
    return (<hipfftResult_t (*)(int *) nogil> _hipfftGetVersion__funptr)(version)


cdef void* _hipfftGetProperty__funptr = NULL
# ! @brief Get library property.
# @param[in] type Property type.
# @param[out] value Returned value.
cdef hipfftResult_t hipfftGetProperty(hipfftLibraryPropertyType_t type,int * value) nogil:
    global _hipfftGetProperty__funptr
    __init_symbol(&_hipfftGetProperty__funptr,"hipfftGetProperty")
    return (<hipfftResult_t (*)(hipfftLibraryPropertyType_t,int *) nogil> _hipfftGetProperty__funptr)(type,value)
