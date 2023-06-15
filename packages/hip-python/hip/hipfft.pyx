# AMD_COPYRIGHT

"""
Attributes:
    HIPFFT_FORWARD (`~.int`):
        Macro constant.

    HIPFFT_BACKWARD (`~.int`):
        Macro constant.

    hipfftResult:
        alias of `~.hipfftResult_t`

    hipfftType:
        alias of `~.hipfftType_t`

    hipfftLibraryPropertyType:
        alias of `~.hipfftLibraryPropertyType_t`

    hipfftHandle:
        alias of `~.hipfftHandle_t`

    hipfftComplex:
        alias of `~.float2`

    hipfftDoubleComplex:
        alias of `~.double2`

"""

import cython
import ctypes
import enum
HIPFFT_FORWARD = chipfft.HIPFFT_FORWARD

HIPFFT_BACKWARD = chipfft.HIPFFT_BACKWARD

class _hipfftResult_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipfftResult_t(_hipfftResult_t__Base):
    HIPFFT_SUCCESS = chipfft.HIPFFT_SUCCESS
    HIPFFT_INVALID_PLAN = chipfft.HIPFFT_INVALID_PLAN
    HIPFFT_ALLOC_FAILED = chipfft.HIPFFT_ALLOC_FAILED
    HIPFFT_INVALID_TYPE = chipfft.HIPFFT_INVALID_TYPE
    HIPFFT_INVALID_VALUE = chipfft.HIPFFT_INVALID_VALUE
    HIPFFT_INTERNAL_ERROR = chipfft.HIPFFT_INTERNAL_ERROR
    HIPFFT_EXEC_FAILED = chipfft.HIPFFT_EXEC_FAILED
    HIPFFT_SETUP_FAILED = chipfft.HIPFFT_SETUP_FAILED
    HIPFFT_INVALID_SIZE = chipfft.HIPFFT_INVALID_SIZE
    HIPFFT_UNALIGNED_DATA = chipfft.HIPFFT_UNALIGNED_DATA
    HIPFFT_INCOMPLETE_PARAMETER_LIST = chipfft.HIPFFT_INCOMPLETE_PARAMETER_LIST
    HIPFFT_INVALID_DEVICE = chipfft.HIPFFT_INVALID_DEVICE
    HIPFFT_PARSE_ERROR = chipfft.HIPFFT_PARSE_ERROR
    HIPFFT_NO_WORKSPACE = chipfft.HIPFFT_NO_WORKSPACE
    HIPFFT_NOT_IMPLEMENTED = chipfft.HIPFFT_NOT_IMPLEMENTED
    HIPFFT_NOT_SUPPORTED = chipfft.HIPFFT_NOT_SUPPORTED
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftResult = hipfftResult_t

class _hipfftType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipfftType_t(_hipfftType_t__Base):
    HIPFFT_R2C = chipfft.HIPFFT_R2C
    HIPFFT_C2R = chipfft.HIPFFT_C2R
    HIPFFT_C2C = chipfft.HIPFFT_C2C
    HIPFFT_D2Z = chipfft.HIPFFT_D2Z
    HIPFFT_Z2D = chipfft.HIPFFT_Z2D
    HIPFFT_Z2Z = chipfft.HIPFFT_Z2Z
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftType = hipfftType_t

class _hipfftLibraryPropertyType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipfftLibraryPropertyType_t(_hipfftLibraryPropertyType_t__Base):
    HIPFFT_MAJOR_VERSION = chipfft.HIPFFT_MAJOR_VERSION
    HIPFFT_MINOR_VERSION = chipfft.HIPFFT_MINOR_VERSION
    HIPFFT_PATCH_LEVEL = chipfft.HIPFFT_PATCH_LEVEL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftLibraryPropertyType = hipfftLibraryPropertyType_t

cdef class hipfftHandle_t:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipfftHandle_t from_ptr(chipfft.hipfftHandle_t* ptr, bint owner=False):
        """Factory function to create ``hipfftHandle_t`` objects from
        given ``chipfft.hipfftHandle_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipfftHandle_t wrapper = hipfftHandle_t.__new__(hipfftHandle_t)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipfftHandle_t from_pyobj(object pyobj):
        """Derives a hipfftHandle_t from a Python object.

        Derives a hipfftHandle_t from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipfftHandle_t`` reference, this method
        returns it directly. No new ``hipfftHandle_t`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `hipfftHandle_t`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipfftHandle_t!
        """
        cdef hipfftHandle_t wrapper = hipfftHandle_t.__new__(hipfftHandle_t)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipfftHandle_t):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipfft.hipfftHandle_t*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    def __int__(self):
        """Returns the data's address as long integer.
        """
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<hipfftHandle_t object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`
        """
        return ctypes.c_void_p(int(self))
    @staticmethod
    def PROPERTIES():
        return []

    def __contains__(self,item):
        properties = self.PROPERTIES()
        return item in properties

    def __getitem__(self,item):
        properties = self.PROPERTIES()
        if isinstance(item,int):
            if item < 0 or item >= len(properties):
                raise IndexError()
            return getattr(self,properties[item])
        raise ValueError("'item' type must be 'int'")


hipfftHandle = hipfftHandle_t

hipfftComplex = float2

hipfftDoubleComplex = double2

@cython.embedsignature(True)
def hipfftPlan1d(int nx, object type, int batch):
    r"""Create a new one-dimensional FFT plan.

    Allocate and initialize a new one-dimensional FFT plan.

    Args:
        nx (`~.int`): **[in]** FFT length.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to compute.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.hipfftHandle_t`: Pointer to the FFT plan handle.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")
    _hipfftPlan1d__retval = hipfftResult_t(chipfft.hipfftPlan1d(&plan._ptr,nx,type.value,batch))    # fully specified
    return (_hipfftPlan1d__retval,plan)


@cython.embedsignature(True)
def hipfftPlan2d(int nx, int ny, object type):
    r"""Create a new two-dimensional FFT plan.

    Allocate and initialize a new two-dimensional FFT plan.
    Two-dimensional data should be stored in C ordering (row-major
    format), so that indexes in y-direction (j index) vary the
    fastest.

    Args:
        nx (`~.int`): **[in]** Number of elements in the x-direction (slow index).

        ny (`~.int`): **[in]** Number of elements in the y-direction (fast index).

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.hipfftHandle_t`: Pointer to the FFT plan handle.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")
    _hipfftPlan2d__retval = hipfftResult_t(chipfft.hipfftPlan2d(&plan._ptr,nx,ny,type.value))    # fully specified
    return (_hipfftPlan2d__retval,plan)


@cython.embedsignature(True)
def hipfftPlan3d(int nx, int ny, int nz, object type):
    r"""Create a new three-dimensional FFT plan.

    Allocate and initialize a new three-dimensional FFT plan.
    Three-dimensional data should be stored in C ordering (row-major
    format), so that indexes in z-direction (k index) vary the
    fastest.

    Args:
        nx (`~.int`): **[in]** Number of elements in the x-direction (slowest index).

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        nz (`~.int`): **[in]** Number of elements in the z-direction (fastest index).

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.hipfftHandle_t`: Pointer to the FFT plan handle.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")
    _hipfftPlan3d__retval = hipfftResult_t(chipfft.hipfftPlan3d(&plan._ptr,nx,ny,nz,type.value))    # fully specified
    return (_hipfftPlan3d__retval,plan)


@cython.embedsignature(True)
def hipfftPlanMany(int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    r"""Create a new batched rank-dimensional FFT plan with advanced data layout.

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
        rank (`~.int`): **[in]** Dimension of transform (1, 2, or 3).

        n (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements to transform in the x/y/z directions.

        inembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the input data in the x/y/z directions.

        istride (`~.int`): **[in]** Distance between two successive elements in the input data.

        idist (`~.int`): **[in]** Distance between input batches.

        onembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the output data in the x/y/z directions.

        ostride (`~.int`): **[in]** Distance between two successive elements in the output data.

        odist (`~.int`): **[in]** Distance between output batches.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to perform.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.hipfftHandle_t`: Pointer to the FFT plan handle.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")
    _hipfftPlanMany__retval = hipfftResult_t(chipfft.hipfftPlanMany(&plan._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch))    # fully specified
    return (_hipfftPlanMany__retval,plan)


@cython.embedsignature(True)
def hipfftCreate():
    r"""Allocate a new plan.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    _hipfftCreate__retval = hipfftResult_t(chipfft.hipfftCreate(&plan._ptr))    # fully specified
    return (_hipfftCreate__retval,plan)


@cython.embedsignature(True)
def hipfftExtPlanScaleFactor(object plan, double scalefactor):
    r"""Set scaling factor.

    hipFFT multiplies each element of the result by the given factor at the end of the transform.

    The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.

    This function must be called after the plan is allocated using
    `~.hipfftCreate`, but before the plan is initialized by any of the
    "MakePlan" functions.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExtPlanScaleFactor__retval = hipfftResult_t(chipfft.hipfftExtPlanScaleFactor(
        hipfftHandle_t.from_pyobj(plan)._ptr,scalefactor))    # fully specified
    return (_hipfftExtPlanScaleFactor__retval,)


@cython.embedsignature(True)
def hipfftMakePlan1d(object plan, int nx, object type, int batch):
    r"""Initialize a new one-dimensional FFT plan.

    Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Handle of the FFT plan.

        nx (`~.int`): **[in]** FFT length.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to compute.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftMakePlan1d__retval = hipfftResult_t(chipfft.hipfftMakePlan1d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlan1d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlan2d(object plan, int nx, int ny, object type):
    r"""Initialize a new two-dimensional FFT plan.

    Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.
    Two-dimensional data should be stored in C ordering (row-major
    format), so that indexes in y-direction (j index) vary the
    fastest.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Handle of the FFT plan.

        nx (`~.int`): **[in]** Number of elements in the x-direction (slow index).

        ny (`~.int`): **[in]** Number of elements in the y-direction (fast index).

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftMakePlan2d__retval = hipfftResult_t(chipfft.hipfftMakePlan2d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftMakePlan2d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlan3d(object plan, int nx, int ny, int nz, object type):
    r"""Initialize a new two-dimensional FFT plan.

    Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.
    Three-dimensional data should be stored in C ordering (row-major
    format), so that indexes in z-direction (k index) vary the
    fastest.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Handle of the FFT plan.

        nx (`~.int`): **[in]** Number of elements in the x-direction (slowest index).

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        nz (`~.int`): **[in]** Number of elements in the z-direction (fastest index).

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftMakePlan3d__retval = hipfftResult_t(chipfft.hipfftMakePlan3d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftMakePlan3d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlanMany(object plan, int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    r"""Initialize a new batched rank-dimensional FFT plan with advanced data layout.

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
        plan (`~.hipfftHandle_t`/`~.object`): **[out]** Pointer to the FFT plan handle.

        rank (`~.int`): **[in]** Dimension of transform (1, 2, or 3).

        n (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements to transform in the x/y/z directions.

        inembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the input data in the x/y/z directions.

        istride (`~.int`): **[in]** Distance between two successive elements in the input data.

        idist (`~.int`): **[in]** Distance between input batches.

        onembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the output data in the x/y/z directions.

        ostride (`~.int`): **[in]** Distance between two successive elements in the output data.

        odist (`~.int`): **[in]** Distance between output batches.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to perform.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftMakePlanMany__retval = hipfftResult_t(chipfft.hipfftMakePlanMany(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlanMany__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlanMany64(object plan, int rank, object n, object inembed, long long istride, long long idist, object onembed, long long ostride, long long odist, object type, long long batch):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftMakePlanMany64__retval = hipfftResult_t(chipfft.hipfftMakePlanMany64(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <long long *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <long long *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <long long *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlanMany64__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate1d(int nx, object type, int batch):
    r"""Return an estimate of the work area size required for a 1D plan.

    Args:
        nx (`~.int`): **[in]** Number of elements in the x-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftEstimate1d__retval = hipfftResult_t(chipfft.hipfftEstimate1d(nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftEstimate1d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate2d(int nx, int ny, object type):
    r"""Return an estimate of the work area size required for a 2D plan.

    Args:
        nx (`~.int`): **[in]** Number of elements in the x-direction.

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftEstimate2d__retval = hipfftResult_t(chipfft.hipfftEstimate2d(nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftEstimate2d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate3d(int nx, int ny, int nz, object type):
    r"""Return an estimate of the work area size required for a 3D plan.

    Args:
        nx (`~.int`): **[in]** Number of elements in the x-direction.

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        nz (`~.int`): **[in]** Number of elements in the z-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftEstimate3d__retval = hipfftResult_t(chipfft.hipfftEstimate3d(nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftEstimate3d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimateMany(int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    r"""Return an estimate of the work area size required for a rank-dimensional plan.

    Args:
        rank (`~.int`): **[in]** Dimension of FFT transform (1, 2, or 3).

        n (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the x/y/z directions.

        inembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** 

        istride (`~.int`): **[in]** 

        idist (`~.int`): **[in]** Distance between input batches.

        onembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** 

        ostride (`~.int`): **[in]** 

        odist (`~.int`): **[in]** Distance between output batches.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to perform.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftEstimateMany__retval = hipfftResult_t(chipfft.hipfftEstimateMany(rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftEstimateMany__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize1d(object plan, int nx, object type, int batch):
    r"""Return size of the work area size required for a 1D plan.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        nx (`~.int`): **[in]** Number of elements in the x-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftGetSize1d__retval = hipfftResult_t(chipfft.hipfftGetSize1d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSize1d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize2d(object plan, int nx, int ny, object type):
    r"""Return size of the work area size required for a 2D plan.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        nx (`~.int`): **[in]** Number of elements in the x-direction.

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftGetSize2d__retval = hipfftResult_t(chipfft.hipfftGetSize2d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftGetSize2d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize3d(object plan, int nx, int ny, int nz, object type):
    r"""Return size of the work area size required for a 3D plan.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        nx (`~.int`): **[in]** Number of elements in the x-direction.

        ny (`~.int`): **[in]** Number of elements in the y-direction.

        nz (`~.int`): **[in]** Number of elements in the z-direction.

        type (`~.hipfftType_t`): **[in]** FFT type.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftGetSize3d__retval = hipfftResult_t(chipfft.hipfftGetSize3d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftGetSize3d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSizeMany(object plan, int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    r"""Return size of the work area size required for a rank-dimensional plan.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        rank (`~.int`): **[in]** Dimension of FFT transform (1, 2, or 3).

        n (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Number of elements in the x/y/z directions.

        inembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** 

        istride (`~.int`): **[in]** 

        idist (`~.int`): **[in]** Distance between input batches.

        onembed (`~.hip._util.types.DataHandle`/`~.object`): **[in]** 

        ostride (`~.int`): **[in]** 

        odist (`~.int`): **[in]** Distance between output batches.

        type (`~.hipfftType_t`): **[in]** FFT type.

        batch (`~.int`): **[in]** Number of batched transforms to perform.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hipfftResult_t`
        * `~.int`: Pointer to work area size (returned value).
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftGetSizeMany__retval = hipfftResult_t(chipfft.hipfftGetSizeMany(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSizeMany__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSizeMany64(object plan, int rank, object n, object inembed, long long istride, long long idist, object onembed, long long ostride, long long odist, object type, long long batch):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    if not isinstance(type,_hipfftType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftType_t__Base'")                    
    cdef unsigned long workSize
    _hipfftGetSizeMany64__retval = hipfftResult_t(chipfft.hipfftGetSizeMany64(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <long long *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <long long *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <long long *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSizeMany64__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize(object plan):
    r"""Return size of the work area size required for a rank-dimensional plan.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    cdef unsigned long workSize
    _hipfftGetSize__retval = hipfftResult_t(chipfft.hipfftGetSize(
        hipfftHandle_t.from_pyobj(plan)._ptr,&workSize))    # fully specified
    return (_hipfftGetSize__retval,workSize)


@cython.embedsignature(True)
def hipfftSetAutoAllocation(object plan, int autoAllocate):
    r"""Set the plan's auto-allocation flag.  The plan will allocate its own workarea.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        autoAllocate (`~.int`): **[in]** 0 to disable auto-allocation, non-zero to enable.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftSetAutoAllocation__retval = hipfftResult_t(chipfft.hipfftSetAutoAllocation(
        hipfftHandle_t.from_pyobj(plan)._ptr,autoAllocate))    # fully specified
    return (_hipfftSetAutoAllocation__retval,)


@cython.embedsignature(True)
def hipfftSetWorkArea(object plan, object workArea):
    r"""Set the plan's work area.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): **[in]** Pointer to the FFT plan.

        workArea (`~.hip._util.types.DataHandle`/`~.object`): **[in]** Pointer to the work area (on device).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftSetWorkArea__retval = hipfftResult_t(chipfft.hipfftSetWorkArea(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(workArea)._ptr))    # fully specified
    return (_hipfftSetWorkArea__retval,)


@cython.embedsignature(True)
def hipfftExecC2C(object plan, object idata, object odata, int direction):
    r"""Execute a (float) complex-to-complex FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.float2`/`~.object`): Input data (on device).

        odata (`~.float2`/`~.object`): Output data (on device).

        direction (`~.int`): Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecC2C__retval = hipfftResult_t(chipfft.hipfftExecC2C(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        float2.from_pyobj(idata)._ptr,
        float2.from_pyobj(odata)._ptr,direction))    # fully specified
    return (_hipfftExecC2C__retval,)


@cython.embedsignature(True)
def hipfftExecR2C(object plan, object idata, object odata):
    r"""Execute a (float) real-to-complex FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.hip._util.types.DataHandle`/`~.object`): Input data (on device).

        odata (`~.float2`/`~.object`): Output data (on device).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecR2C__retval = hipfftResult_t(chipfft.hipfftExecR2C(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(idata)._ptr,
        float2.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecR2C__retval,)


@cython.embedsignature(True)
def hipfftExecC2R(object plan, object idata, object odata):
    r"""Execute a (float) complex-to-real FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.float2`/`~.object`): Input data (on device).

        odata (`~.hip._util.types.DataHandle`/`~.object`): Output data (on device).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecC2R__retval = hipfftResult_t(chipfft.hipfftExecC2R(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        float2.from_pyobj(idata)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecC2R__retval,)


@cython.embedsignature(True)
def hipfftExecZ2Z(object plan, object idata, object odata, int direction):
    r"""Execute a (double) complex-to-complex FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.double2`/`~.object`): Input data (on device).

        odata (`~.double2`/`~.object`): Output data (on device).

        direction (`~.int`): Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecZ2Z__retval = hipfftResult_t(chipfft.hipfftExecZ2Z(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        double2.from_pyobj(idata)._ptr,
        double2.from_pyobj(odata)._ptr,direction))    # fully specified
    return (_hipfftExecZ2Z__retval,)


@cython.embedsignature(True)
def hipfftExecD2Z(object plan, object idata, object odata):
    r"""Execute a (double) real-to-complex FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.hip._util.types.DataHandle`/`~.object`): Input data (on device).

        odata (`~.double2`/`~.object`): Output data (on device).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecD2Z__retval = hipfftResult_t(chipfft.hipfftExecD2Z(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(idata)._ptr,
        double2.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecD2Z__retval,)


@cython.embedsignature(True)
def hipfftExecZ2D(object plan, object idata, object odata):
    r"""Execute a (double) complex-to-real FFT.

    If the input and output buffers are equal, an in-place
    transform is performed.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        idata (`~.double2`/`~.object`): Input data (on device).

        odata (`~.hip._util.types.DataHandle`/`~.object`): Output data (on device).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftExecZ2D__retval = hipfftResult_t(chipfft.hipfftExecZ2D(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        double2.from_pyobj(idata)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecZ2D__retval,)


@cython.embedsignature(True)
def hipfftSetStream(object plan, object stream):
    r"""Set HIP stream to execute plan on.

    Associates a HIP stream with a hipFFT plan.  All kernels
    launched by this plan are associated with the provided stream.

    Args:
        plan (`~.hipfftHandle_t`/`~.object`): The FFT plan.

        stream (`~.ihipStream_t`/`~.object`): The HIP stream.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftSetStream__retval = hipfftResult_t(chipfft.hipfftSetStream(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipfftSetStream__retval,)


@cython.embedsignature(True)
def hipfftDestroy(object plan):
    r"""Destroy and deallocate an existing plan.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftDestroy__retval = hipfftResult_t(chipfft.hipfftDestroy(
        hipfftHandle_t.from_pyobj(plan)._ptr))    # fully specified
    return (_hipfftDestroy__retval,)


@cython.embedsignature(True)
def hipfftGetVersion(object version):
    r"""Get rocFFT/cuFFT version.

    Args:
        version (`~.hip._util.types.DataHandle`/`~.object`): **[out]** cuFFT/rocFFT version (returned value).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    _hipfftGetVersion__retval = hipfftResult_t(chipfft.hipfftGetVersion(
        <int *>hip._util.types.DataHandle.from_pyobj(version)._ptr))    # fully specified
    return (_hipfftGetVersion__retval,)


@cython.embedsignature(True)
def hipfftGetProperty(object type, object value):
    r"""Get library property.

    Args:
        type (`~.hipfftLibraryPropertyType_t`): **[in]** Property type.

        value (`~.hip._util.types.DataHandle`/`~.object`): **[out]** Returned value.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipfftResult_t`
    """
    if not isinstance(type,_hipfftLibraryPropertyType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipfftLibraryPropertyType_t__Base'")
    _hipfftGetProperty__retval = hipfftResult_t(chipfft.hipfftGetProperty(type.value,
        <int *>hip._util.types.DataHandle.from_pyobj(value)._ptr))    # fully specified
    return (_hipfftGetProperty__retval,)
