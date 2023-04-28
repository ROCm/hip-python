# AMD_COPYRIGHT
# c imports
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
# python imports
import cython
import ctypes
import enum
from hip._util.datahandle cimport DataHandle
#ctypedef int16_t __int16_t
#ctypedef uint16_t __uint16_t
from .hip cimport ihipStream_t

from . cimport chipblas
hipblasVersionMajor = chipblas.hipblasVersionMajor

hipblaseVersionMinor = chipblas.hipblaseVersionMinor

hipblasVersionMinor = chipblas.hipblasVersionMinor

hipblasVersionPatch = chipblas.hipblasVersionPatch

cdef class hipblasHandle_t:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipblasHandle_t from_ptr(void * ptr, bint owner=False):
        """Factory function to create ``hipblasHandle_t`` objects from
        given ``void`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasHandle_t wrapper = hipblasHandle_t.__new__(hipblasHandle_t)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipblasHandle_t from_pyobj(object pyobj):
        """Derives a hipblasHandle_t from a Python object.

        Derives a hipblasHandle_t from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipblasHandle_t`` reference, this method
        returns it directly. No new ``hipblasHandle_t`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``hipblasHandle_t``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipblasHandle_t!
        """
        cdef hipblasHandle_t wrapper = hipblasHandle_t.__new__(hipblasHandle_t)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipblasHandle_t):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <void *>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <void *>cpython.long.PyLong_AsVoidPtr(pyobj.value)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <void *>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <void *>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<hipblasHandle_t object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)


cdef class hipblasBfloat16:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipblasBfloat16 from_ptr(chipblas.hipblasBfloat16* ptr, bint owner=False):
        """Factory function to create ``hipblasBfloat16`` objects from
        given ``chipblas.hipblasBfloat16`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasBfloat16 wrapper = hipblasBfloat16.__new__(hipblasBfloat16)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipblasBfloat16 from_pyobj(object pyobj):
        """Derives a hipblasBfloat16 from a Python object.

        Derives a hipblasBfloat16 from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipblasBfloat16`` reference, this method
        returns it directly. No new ``hipblasBfloat16`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``hipblasBfloat16``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipblasBfloat16!
        """
        cdef hipblasBfloat16 wrapper = hipblasBfloat16.__new__(hipblasBfloat16)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipblasBfloat16):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipblas.hipblasBfloat16*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipblas.hipblasBfloat16*>cpython.long.PyLong_AsVoidPtr(pyobj.value)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasBfloat16*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipblas.hipblasBfloat16*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef __allocate(chipblas.hipblasBfloat16** ptr):
        ptr[0] = <chipblas.hipblasBfloat16*>stdlib.malloc(sizeof(chipblas.hipblasBfloat16))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasBfloat16 new():
        """Factory function to create hipblasBfloat16 objects with
        newly allocated chipblas.hipblasBfloat16"""
        cdef chipblas.hipblasBfloat16* ptr;
        hipblasBfloat16.__allocate(&ptr)
        return hipblasBfloat16.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasBfloat16.__allocate(&self._ptr)
       self.ptr_owner = True
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<hipblasBfloat16 object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_data(self, i):
        """Get value ``data`` of ``self._ptr[i]``.
        """
        return self._ptr[i].data
    def set_data(self, i, uint16_t value):
        """Set value ``data`` of ``self._ptr[i]``.
        """
        self._ptr[i].data = value
    @property
    def data(self):
        return self.get_data(0)
    @data.setter
    def data(self, uint16_t value):
        self.set_data(0,value)


cdef class hipblasComplex:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipblasComplex from_ptr(chipblas.hipblasComplex* ptr, bint owner=False):
        """Factory function to create ``hipblasComplex`` objects from
        given ``chipblas.hipblasComplex`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasComplex wrapper = hipblasComplex.__new__(hipblasComplex)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipblasComplex from_pyobj(object pyobj):
        """Derives a hipblasComplex from a Python object.

        Derives a hipblasComplex from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipblasComplex`` reference, this method
        returns it directly. No new ``hipblasComplex`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``hipblasComplex``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipblasComplex!
        """
        cdef hipblasComplex wrapper = hipblasComplex.__new__(hipblasComplex)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipblasComplex):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipblas.hipblasComplex*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipblas.hipblasComplex*>cpython.long.PyLong_AsVoidPtr(pyobj.value)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasComplex*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipblas.hipblasComplex*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef __allocate(chipblas.hipblasComplex** ptr):
        ptr[0] = <chipblas.hipblasComplex*>stdlib.malloc(sizeof(chipblas.hipblasComplex))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasComplex new():
        """Factory function to create hipblasComplex objects with
        newly allocated chipblas.hipblasComplex"""
        cdef chipblas.hipblasComplex* ptr;
        hipblasComplex.__allocate(&ptr)
        return hipblasComplex.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasComplex.__allocate(&self._ptr)
       self.ptr_owner = True
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<hipblasComplex object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, float value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, float value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, float value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, float value):
        self.set_y(0,value)


cdef class hipblasDoubleComplex:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipblasDoubleComplex from_ptr(chipblas.hipblasDoubleComplex* ptr, bint owner=False):
        """Factory function to create ``hipblasDoubleComplex`` objects from
        given ``chipblas.hipblasDoubleComplex`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipblasDoubleComplex wrapper = hipblasDoubleComplex.__new__(hipblasDoubleComplex)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipblasDoubleComplex from_pyobj(object pyobj):
        """Derives a hipblasDoubleComplex from a Python object.

        Derives a hipblasDoubleComplex from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipblasDoubleComplex`` reference, this method
        returns it directly. No new ``hipblasDoubleComplex`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``hipblasDoubleComplex``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipblasDoubleComplex!
        """
        cdef hipblasDoubleComplex wrapper = hipblasDoubleComplex.__new__(hipblasDoubleComplex)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipblasDoubleComplex):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>cpython.long.PyLong_AsVoidPtr(pyobj.value)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef __allocate(chipblas.hipblasDoubleComplex** ptr):
        ptr[0] = <chipblas.hipblasDoubleComplex*>stdlib.malloc(sizeof(chipblas.hipblasDoubleComplex))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef hipblasDoubleComplex new():
        """Factory function to create hipblasDoubleComplex objects with
        newly allocated chipblas.hipblasDoubleComplex"""
        cdef chipblas.hipblasDoubleComplex* ptr;
        hipblasDoubleComplex.__allocate(&ptr)
        return hipblasDoubleComplex.from_ptr(ptr, owner=True)
    
    def __init__(self):
       hipblasDoubleComplex.__allocate(&self._ptr)
       self.ptr_owner = True
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<hipblasDoubleComplex object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, double value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, double value):
        self.set_x(0,value)
    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, double value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, double value):
        self.set_y(0,value)


class hipblasStatus_t(enum.IntEnum):
    HIPBLAS_STATUS_SUCCESS = chipblas.HIPBLAS_STATUS_SUCCESS
    HIPBLAS_STATUS_NOT_INITIALIZED = chipblas.HIPBLAS_STATUS_NOT_INITIALIZED
    HIPBLAS_STATUS_ALLOC_FAILED = chipblas.HIPBLAS_STATUS_ALLOC_FAILED
    HIPBLAS_STATUS_INVALID_VALUE = chipblas.HIPBLAS_STATUS_INVALID_VALUE
    HIPBLAS_STATUS_MAPPING_ERROR = chipblas.HIPBLAS_STATUS_MAPPING_ERROR
    HIPBLAS_STATUS_EXECUTION_FAILED = chipblas.HIPBLAS_STATUS_EXECUTION_FAILED
    HIPBLAS_STATUS_INTERNAL_ERROR = chipblas.HIPBLAS_STATUS_INTERNAL_ERROR
    HIPBLAS_STATUS_NOT_SUPPORTED = chipblas.HIPBLAS_STATUS_NOT_SUPPORTED
    HIPBLAS_STATUS_ARCH_MISMATCH = chipblas.HIPBLAS_STATUS_ARCH_MISMATCH
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = chipblas.HIPBLAS_STATUS_HANDLE_IS_NULLPTR
    HIPBLAS_STATUS_INVALID_ENUM = chipblas.HIPBLAS_STATUS_INVALID_ENUM
    HIPBLAS_STATUS_UNKNOWN = chipblas.HIPBLAS_STATUS_UNKNOWN

class hipblasOperation_t(enum.IntEnum):
    HIPBLAS_OP_N = chipblas.HIPBLAS_OP_N
    HIPBLAS_OP_T = chipblas.HIPBLAS_OP_T
    HIPBLAS_OP_C = chipblas.HIPBLAS_OP_C

class hipblasPointerMode_t(enum.IntEnum):
    HIPBLAS_POINTER_MODE_HOST = chipblas.HIPBLAS_POINTER_MODE_HOST
    HIPBLAS_POINTER_MODE_DEVICE = chipblas.HIPBLAS_POINTER_MODE_DEVICE

class hipblasFillMode_t(enum.IntEnum):
    HIPBLAS_FILL_MODE_UPPER = chipblas.HIPBLAS_FILL_MODE_UPPER
    HIPBLAS_FILL_MODE_LOWER = chipblas.HIPBLAS_FILL_MODE_LOWER
    HIPBLAS_FILL_MODE_FULL = chipblas.HIPBLAS_FILL_MODE_FULL

class hipblasDiagType_t(enum.IntEnum):
    HIPBLAS_DIAG_NON_UNIT = chipblas.HIPBLAS_DIAG_NON_UNIT
    HIPBLAS_DIAG_UNIT = chipblas.HIPBLAS_DIAG_UNIT

class hipblasSideMode_t(enum.IntEnum):
    HIPBLAS_SIDE_LEFT = chipblas.HIPBLAS_SIDE_LEFT
    HIPBLAS_SIDE_RIGHT = chipblas.HIPBLAS_SIDE_RIGHT
    HIPBLAS_SIDE_BOTH = chipblas.HIPBLAS_SIDE_BOTH

class hipblasDatatype_t(enum.IntEnum):
    HIPBLAS_R_16F = chipblas.HIPBLAS_R_16F
    HIPBLAS_R_32F = chipblas.HIPBLAS_R_32F
    HIPBLAS_R_64F = chipblas.HIPBLAS_R_64F
    HIPBLAS_C_16F = chipblas.HIPBLAS_C_16F
    HIPBLAS_C_32F = chipblas.HIPBLAS_C_32F
    HIPBLAS_C_64F = chipblas.HIPBLAS_C_64F
    HIPBLAS_R_8I = chipblas.HIPBLAS_R_8I
    HIPBLAS_R_8U = chipblas.HIPBLAS_R_8U
    HIPBLAS_R_32I = chipblas.HIPBLAS_R_32I
    HIPBLAS_R_32U = chipblas.HIPBLAS_R_32U
    HIPBLAS_C_8I = chipblas.HIPBLAS_C_8I
    HIPBLAS_C_8U = chipblas.HIPBLAS_C_8U
    HIPBLAS_C_32I = chipblas.HIPBLAS_C_32I
    HIPBLAS_C_32U = chipblas.HIPBLAS_C_32U
    HIPBLAS_R_16B = chipblas.HIPBLAS_R_16B
    HIPBLAS_C_16B = chipblas.HIPBLAS_C_16B

class hipblasGemmAlgo_t(enum.IntEnum):
    HIPBLAS_GEMM_DEFAULT = chipblas.HIPBLAS_GEMM_DEFAULT

class hipblasAtomicsMode_t(enum.IntEnum):
    HIPBLAS_ATOMICS_NOT_ALLOWED = chipblas.HIPBLAS_ATOMICS_NOT_ALLOWED
    HIPBLAS_ATOMICS_ALLOWED = chipblas.HIPBLAS_ATOMICS_ALLOWED

class hipblasInt8Datatype_t(enum.IntEnum):
    HIPBLAS_INT8_DATATYPE_DEFAULT = chipblas.HIPBLAS_INT8_DATATYPE_DEFAULT
    HIPBLAS_INT8_DATATYPE_INT8 = chipblas.HIPBLAS_INT8_DATATYPE_INT8
    HIPBLAS_INT8_DATATYPE_PACK_INT8x4 = chipblas.HIPBLAS_INT8_DATATYPE_PACK_INT8x4

@cython.embedsignature(True)
def hipblasCreate():
    """! \brief Create hipblas handle. */
    """
    handle = DataHandle.from_ptr(NULL)
    _hipblasCreate__retval = hipblasStatus_t(chipblas.hipblasCreate(
        <void **>&handle._ptr))    # fully specified
    return (_hipblasCreate__retval,handle)


@cython.embedsignature(True)
def hipblasDestroy(object handle):
    """! \brief Destroys the library context created using hipblasCreate() */
    """
    _hipblasDestroy__retval = hipblasStatus_t(chipblas.hipblasDestroy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr))    # fully specified
    return (_hipblasDestroy__retval,)


@cython.embedsignature(True)
def hipblasSetStream(object handle, object streamId):
    """! \brief Set stream for handle */
    """
    _hipblasSetStream__retval = hipblasStatus_t(chipblas.hipblasSetStream(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        ihipStream_t.from_pyobj(streamId)._ptr))    # fully specified
    return (_hipblasSetStream__retval,)


@cython.embedsignature(True)
def hipblasGetStream(object handle, object streamId):
    """! \brief Get stream[0] for handle */
    """
    _hipblasGetStream__retval = hipblasStatus_t(chipblas.hipblasGetStream(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipStream_t*>DataHandle.from_pyobj(streamId)._ptr))    # fully specified
    return (_hipblasGetStream__retval,)


@cython.embedsignature(True)
def hipblasSetPointerMode(object handle, object mode):
    """! \brief Set hipblas pointer mode */
    """
    if not isinstance(mode,hipblasPointerMode_t):
        raise TypeError("argument 'mode' must be of type 'hipblasPointerMode_t'")
    _hipblasSetPointerMode__retval = hipblasStatus_t(chipblas.hipblasSetPointerMode(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,mode.value))    # fully specified
    return (_hipblasSetPointerMode__retval,)


@cython.embedsignature(True)
def hipblasGetPointerMode(object handle):
    """! \brief Get hipblas pointer mode */
    """
    pass

@cython.embedsignature(True)
def hipblasSetInt8Datatype(object handle, object int8Type):
    """! \brief Set hipblas int8 Datatype */
    """
    if not isinstance(int8Type,hipblasInt8Datatype_t):
        raise TypeError("argument 'int8Type' must be of type 'hipblasInt8Datatype_t'")
    _hipblasSetInt8Datatype__retval = hipblasStatus_t(chipblas.hipblasSetInt8Datatype(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,int8Type.value))    # fully specified
    return (_hipblasSetInt8Datatype__retval,)


@cython.embedsignature(True)
def hipblasGetInt8Datatype(object handle):
    """! \brief Get hipblas int8 Datatype*/
    """
    pass

@cython.embedsignature(True)
def hipblasSetVector(int n, int elemSize, object x, int incx, object y, int incy):
    """! \brief copy vector from host to device
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize    [int]
                    Size of both vectors in bytes
        @param[in]
        x           pointer to vector on the host
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the device
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
    """
    _hipblasSetVector__retval = hipblasStatus_t(chipblas.hipblasSetVector(n,elemSize,
        <const void *>DataHandle.from_pyobj(x)._ptr,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSetVector__retval,)


@cython.embedsignature(True)
def hipblasGetVector(int n, int elemSize, object x, int incx, object y, int incy):
    """! \brief copy vector from device to host
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize    [int]
                    Size of both vectors in bytes
        @param[in]
        x           pointer to vector on the device
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the host
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
    """
    _hipblasGetVector__retval = hipblasStatus_t(chipblas.hipblasGetVector(n,elemSize,
        <const void *>DataHandle.from_pyobj(x)._ptr,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasGetVector__retval,)


@cython.embedsignature(True)
def hipblasSetMatrix(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb):
    """! \brief copy matrix from host to device
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the host
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the GPU
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
    """
    _hipblasSetMatrix__retval = hipblasStatus_t(chipblas.hipblasSetMatrix(rows,cols,elemSize,
        <const void *>DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasSetMatrix__retval,)


@cython.embedsignature(True)
def hipblasGetMatrix(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb):
    """! \brief copy matrix from device to host
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the GPU
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP          pointer to matrix on the host
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
    """
    _hipblasGetMatrix__retval = hipblasStatus_t(chipblas.hipblasGetMatrix(rows,cols,elemSize,
        <const void *>DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasGetMatrix__retval,)


@cython.embedsignature(True)
def hipblasSetVectorAsync(int n, int elemSize, object x, int incx, object y, int incy, object stream):
    """! \brief asynchronously copy vector from host to device
        \details
        hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        x           pointer to vector on the host
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the device
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    _hipblasSetVectorAsync__retval = hipblasStatus_t(chipblas.hipblasSetVectorAsync(n,elemSize,
        <const void *>DataHandle.from_pyobj(x)._ptr,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,incy,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasSetVectorAsync__retval,)


@cython.embedsignature(True)
def hipblasGetVectorAsync(int n, int elemSize, object x, int incx, object y, int incy, object stream):
    """! \brief asynchronously copy vector from device to host
        \details
        hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        n           [int]
                    number of elements in the vector
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        x           pointer to vector on the device
        @param[in]
        incx        [int]
                    specifies the increment for the elements of the vector
        @param[out]
        y           pointer to vector on the host
        @param[in]
        incy        [int]
                    specifies the increment for the elements of the vector
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    _hipblasGetVectorAsync__retval = hipblasStatus_t(chipblas.hipblasGetVectorAsync(n,elemSize,
        <const void *>DataHandle.from_pyobj(x)._ptr,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,incy,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasGetVectorAsync__retval,)


@cython.embedsignature(True)
def hipblasSetMatrixAsync(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb, object stream):
    """! \brief asynchronously copy matrix from host to device
        \details
        hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP           pointer to matrix on the host
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the GPU
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    _hipblasSetMatrixAsync__retval = hipblasStatus_t(chipblas.hipblasSetMatrixAsync(rows,cols,elemSize,
        <const void *>DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>DataHandle.from_pyobj(BP)._ptr,ldb,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasSetMatrixAsync__retval,)


@cython.embedsignature(True)
def hipblasGetMatrixAsync(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb, object stream):
    """! \brief asynchronously copy matrix from device to host
        \details
        hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
        Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
        @param[in]
        rows        [int]
                    number of rows in matrices
        @param[in]
        cols        [int]
                    number of columns in matrices
        @param[in]
        elemSize   [int]
                    number of bytes per element in the matrix
        @param[in]
        AP          pointer to matrix on the GPU
        @param[in]
        lda         [int]
                    specifies the leading dimension of A, lda >= rows
        @param[out]
        BP           pointer to matrix on the host
        @param[in]
        ldb         [int]
                    specifies the leading dimension of B, ldb >= rows
        @param[in]
        stream      specifies the stream into which this transfer request is queued
    """
    _hipblasGetMatrixAsync__retval = hipblasStatus_t(chipblas.hipblasGetMatrixAsync(rows,cols,elemSize,
        <const void *>DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>DataHandle.from_pyobj(BP)._ptr,ldb,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasGetMatrixAsync__retval,)


@cython.embedsignature(True)
def hipblasSetAtomicsMode(object handle, object atomics_mode):
    """! \brief Set hipblasSetAtomicsMode*/
    """
    if not isinstance(atomics_mode,hipblasAtomicsMode_t):
        raise TypeError("argument 'atomics_mode' must be of type 'hipblasAtomicsMode_t'")
    _hipblasSetAtomicsMode__retval = hipblasStatus_t(chipblas.hipblasSetAtomicsMode(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,atomics_mode.value))    # fully specified
    return (_hipblasSetAtomicsMode__retval,)


@cython.embedsignature(True)
def hipblasGetAtomicsMode(object handle):
    """! \brief Get hipblasSetAtomicsMode*/
    """
    pass

@cython.embedsignature(True)
def hipblasIsamax(object handle, int n, object x, int incx, object result):
    """! @{
        \brief BLAS Level 1 API

        \details
        amax finds the first index of the element of maximum magnitude of a vector x.

        - Supported precisions in rocBLAS : s,d,c,z.
        - Supported precisions in cuBLAS  : s,d,c,z.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the amax index.
                  return is 0.0 if n, incx<=0.
    """
    _hipblasIsamax__retval = hipblasStatus_t(chipblas.hipblasIsamax(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamax__retval,)


@cython.embedsignature(True)
def hipblasIdamax(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIdamax__retval = hipblasStatus_t(chipblas.hipblasIdamax(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamax__retval,)


@cython.embedsignature(True)
def hipblasIcamax(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIcamax__retval = hipblasStatus_t(chipblas.hipblasIcamax(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamax__retval,)


@cython.embedsignature(True)
def hipblasIzamax(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIzamax__retval = hipblasStatus_t(chipblas.hipblasIzamax(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamax__retval,)


@cython.embedsignature(True)
def hipblasIsamin(object handle, int n, object x, int incx, object result):
    """! @{
        \brief BLAS Level 1 API

        \details
        amin finds the first index of the element of minimum magnitude of a vector x.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the amin index.
                  return is 0.0 if n, incx<=0.
    """
    _hipblasIsamin__retval = hipblasStatus_t(chipblas.hipblasIsamin(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamin__retval,)


@cython.embedsignature(True)
def hipblasIdamin(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIdamin__retval = hipblasStatus_t(chipblas.hipblasIdamin(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamin__retval,)


@cython.embedsignature(True)
def hipblasIcamin(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIcamin__retval = hipblasStatus_t(chipblas.hipblasIcamin(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamin__retval,)


@cython.embedsignature(True)
def hipblasIzamin(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasIzamin__retval = hipblasStatus_t(chipblas.hipblasIzamin(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <int *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamin__retval,)


@cython.embedsignature(True)
def hipblasSasum(object handle, int n, object x, int incx, object result):
    """! @{
        \brief BLAS Level 1 API

        \details
        asum computes the sum of the magnitudes of elements of a real vector x,
             or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x. incx must be > 0.
        @param[inout]
        result
                  device pointer or host pointer to store the asum product.
                  return is 0.0 if n <= 0.
    """
    _hipblasSasum__retval = hipblasStatus_t(chipblas.hipblasSasum(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSasum__retval,)


@cython.embedsignature(True)
def hipblasDasum(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasDasum__retval = hipblasStatus_t(chipblas.hipblasDasum(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDasum__retval,)


@cython.embedsignature(True)
def hipblasScasum(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasScasum__retval = hipblasStatus_t(chipblas.hipblasScasum(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScasum__retval,)


@cython.embedsignature(True)
def hipblasDzasum(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasDzasum__retval = hipblasStatus_t(chipblas.hipblasDzasum(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDzasum__retval,)


@cython.embedsignature(True)
def hipblasHaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        axpy   computes constant alpha multiplied by vector x, plus vector y

            y := alpha * x + y

        - Supported precisions in rocBLAS : h,s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        alpha     device pointer or host pointer to specify the scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[out]
        y         device pointer storing vector y.
        @param[inout]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    _hipblasHaxpy__retval = hipblasStatus_t(chipblas.hipblasHaxpy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasHaxpy__retval,)


@cython.embedsignature(True)
def hipblasSaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """
    """
    _hipblasSaxpy__retval = hipblasStatus_t(chipblas.hipblasSaxpy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSaxpy__retval,)


@cython.embedsignature(True)
def hipblasDaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """
    """
    _hipblasDaxpy__retval = hipblasStatus_t(chipblas.hipblasDaxpy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDaxpy__retval,)


@cython.embedsignature(True)
def hipblasCaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """
    """
    _hipblasCaxpy__retval = hipblasStatus_t(chipblas.hipblasCaxpy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCaxpy__retval,)


@cython.embedsignature(True)
def hipblasZaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """
    """
    _hipblasZaxpy__retval = hipblasStatus_t(chipblas.hipblasZaxpy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZaxpy__retval,)


@cython.embedsignature(True)
def hipblasScopy(object handle, int n, object x, int incx, object y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        copy  copies each element x[i] into y[i], for  i = 1 , ... , n

            y := x,

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x to be copied to y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[out]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    _hipblasScopy__retval = hipblasStatus_t(chipblas.hipblasScopy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasScopy__retval,)


@cython.embedsignature(True)
def hipblasDcopy(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasDcopy__retval = hipblasStatus_t(chipblas.hipblasDcopy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDcopy__retval,)


@cython.embedsignature(True)
def hipblasCcopy(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasCcopy__retval = hipblasStatus_t(chipblas.hipblasCcopy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCcopy__retval,)


@cython.embedsignature(True)
def hipblasZcopy(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasZcopy__retval = hipblasStatus_t(chipblas.hipblasZcopy(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZcopy__retval,)


@cython.embedsignature(True)
def hipblasHdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """! @{
        \brief BLAS Level 1 API

        \details
        dot(u)  performs the dot product of vectors x and y

            result = x * y;

        dotc  performs the dot product of the conjugate of complex vector x and complex vector y

            result = conjugate (x) * y;

        - Supported precisions in rocBLAS : h,bf,s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the dot product.
                  return is 0.0 if n <= 0.
    """
    _hipblasHdot__retval = hipblasStatus_t(chipblas.hipblasHdot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasHdot__retval,)


@cython.embedsignature(True)
def hipblasBfdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasBfdot__retval = hipblasStatus_t(chipblas.hipblasBfdot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasBfloat16.from_pyobj(x)._ptr,incx,
        hipblasBfloat16.from_pyobj(y)._ptr,incy,
        hipblasBfloat16.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasBfdot__retval,)


@cython.embedsignature(True)
def hipblasSdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasSdot__retval = hipblasStatus_t(chipblas.hipblasSdot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(y)._ptr,incy,
        <float *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSdot__retval,)


@cython.embedsignature(True)
def hipblasDdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasDdot__retval = hipblasStatus_t(chipblas.hipblasDdot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(y)._ptr,incy,
        <double *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDdot__retval,)


@cython.embedsignature(True)
def hipblasCdotc(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasCdotc__retval = hipblasStatus_t(chipblas.hipblasCdotc(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotc__retval,)


@cython.embedsignature(True)
def hipblasCdotu(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasCdotu__retval = hipblasStatus_t(chipblas.hipblasCdotu(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotu__retval,)


@cython.embedsignature(True)
def hipblasZdotc(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasZdotc__retval = hipblasStatus_t(chipblas.hipblasZdotc(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotc__retval,)


@cython.embedsignature(True)
def hipblasZdotu(object handle, int n, object x, int incx, object y, int incy, object result):
    """
    """
    _hipblasZdotu__retval = hipblasStatus_t(chipblas.hipblasZdotu(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotu__retval,)


@cython.embedsignature(True)
def hipblasSnrm2(object handle, int n, object x, int incx, object result):
    """! @{
        \brief BLAS Level 1 API

        \details
        nrm2 computes the euclidean norm of a real or complex vector

                  result := sqrt( x'*x ) for real vectors
                  result := sqrt( x**H*x ) for complex vectors

        - Supported precisions in rocBLAS : s,d,c,z,sc,dz
        - Supported precisions in cuBLAS  : s,d,sc,dz

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the nrm2 product.
                  return is 0.0 if n, incx<=0.
    """
    _hipblasSnrm2__retval = hipblasStatus_t(chipblas.hipblasSnrm2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSnrm2__retval,)


@cython.embedsignature(True)
def hipblasDnrm2(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasDnrm2__retval = hipblasStatus_t(chipblas.hipblasDnrm2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDnrm2__retval,)


@cython.embedsignature(True)
def hipblasScnrm2(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasScnrm2__retval = hipblasStatus_t(chipblas.hipblasScnrm2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScnrm2__retval,)


@cython.embedsignature(True)
def hipblasDznrm2(object handle, int n, object x, int incx, object result):
    """
    """
    _hipblasDznrm2__retval = hipblasStatus_t(chipblas.hipblasDznrm2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDznrm2__retval,)


@cython.embedsignature(True)
def hipblasSrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """! @{
        \brief BLAS Level 1 API

        \details
        rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
            Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

        - Supported precisions in rocBLAS : s,d,c,z,sc,dz
        - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
        @param[in]
        s       device pointer or host pointer storing scalar sine component of the rotation matrix.
    """
    _hipblasSrot__retval = hipblasStatus_t(chipblas.hipblasSrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>DataHandle.from_pyobj(c)._ptr,
        <const float *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasSrot__retval,)


@cython.embedsignature(True)
def hipblasDrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """
    """
    _hipblasDrot__retval = hipblasStatus_t(chipblas.hipblasDrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>DataHandle.from_pyobj(c)._ptr,
        <const double *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasDrot__retval,)


@cython.embedsignature(True)
def hipblasCrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """
    """
    _hipblasCrot__retval = hipblasStatus_t(chipblas.hipblasCrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        <const float *>DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCrot__retval,)


@cython.embedsignature(True)
def hipblasCsrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """
    """
    _hipblasCsrot__retval = hipblasStatus_t(chipblas.hipblasCsrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        <const float *>DataHandle.from_pyobj(c)._ptr,
        <const float *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCsrot__retval,)


@cython.embedsignature(True)
def hipblasZrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """
    """
    _hipblasZrot__retval = hipblasStatus_t(chipblas.hipblasZrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        <const double *>DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZrot__retval,)


@cython.embedsignature(True)
def hipblasZdrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """
    """
    _hipblasZdrot__retval = hipblasStatus_t(chipblas.hipblasZdrot(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        <const double *>DataHandle.from_pyobj(c)._ptr,
        <const double *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZdrot__retval,)


@cython.embedsignature(True)
def hipblasSrotg(object handle, object a, object b, object c, object s):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotg creates the Givens rotation matrix for the vector (a b).
             Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
             If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
             If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[inout]
        a       device pointer or host pointer to input vector element, overwritten with r.
        @param[inout]
        b       device pointer or host pointer to input vector element, overwritten with z.
        @param[inout]
        c       device pointer or host pointer to cosine element of Givens rotation.
        @param[inout]
        s       device pointer or host pointer sine element of Givens rotation.
    """
    _hipblasSrotg__retval = hipblasStatus_t(chipblas.hipblasSrotg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        <float *>DataHandle.from_pyobj(a)._ptr,
        <float *>DataHandle.from_pyobj(b)._ptr,
        <float *>DataHandle.from_pyobj(c)._ptr,
        <float *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasSrotg__retval,)


@cython.embedsignature(True)
def hipblasDrotg(object handle, object a, object b, object c, object s):
    """
    """
    _hipblasDrotg__retval = hipblasStatus_t(chipblas.hipblasDrotg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        <double *>DataHandle.from_pyobj(a)._ptr,
        <double *>DataHandle.from_pyobj(b)._ptr,
        <double *>DataHandle.from_pyobj(c)._ptr,
        <double *>DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasDrotg__retval,)


@cython.embedsignature(True)
def hipblasCrotg(object handle, object a, object b, object c, object s):
    """
    """
    _hipblasCrotg__retval = hipblasStatus_t(chipblas.hipblasCrotg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        hipblasComplex.from_pyobj(a)._ptr,
        hipblasComplex.from_pyobj(b)._ptr,
        <float *>DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCrotg__retval,)


@cython.embedsignature(True)
def hipblasZrotg(object handle, object a, object b, object c, object s):
    """
    """
    _hipblasZrotg__retval = hipblasStatus_t(chipblas.hipblasZrotg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        hipblasDoubleComplex.from_pyobj(a)._ptr,
        hipblasDoubleComplex.from_pyobj(b)._ptr,
        <double *>DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZrotg__retval,)


@cython.embedsignature(True)
def hipblasSrotm(object handle, int n, object x, int incx, object y, int incy, object param):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotm applies the modified Givens rotation matrix defined by param to vectors x and y.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        param   device vector or host vector of 5 elements defining the rotation.
                param[0] = flag
                param[1] = H11
                param[2] = H21
                param[3] = H12
                param[4] = H22
                The flag parameter defines the form of H:
                flag = -1 => H = ( H11 H12 H21 H22 )
                flag =  0 => H = ( 1.0 H12 H21 1.0 )
                flag =  1 => H = ( H11 1.0 -1.0 H22 )
                flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
                param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
    """
    _hipblasSrotm__retval = hipblasStatus_t(chipblas.hipblasSrotm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasSrotm__retval,)


@cython.embedsignature(True)
def hipblasDrotm(object handle, int n, object x, int incx, object y, int incy, object param):
    """
    """
    _hipblasDrotm__retval = hipblasStatus_t(chipblas.hipblasDrotm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasDrotm__retval,)


@cython.embedsignature(True)
def hipblasSrotmg(object handle, object d1, object d2, object x1, object y1, object param):
    """! @{
        \brief BLAS Level 1 API

        \details
        rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
              Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
              If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
              If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[inout]
        d1      device pointer or host pointer to input scalar that is overwritten.
        @param[inout]
        d2      device pointer or host pointer to input scalar that is overwritten.
        @param[inout]
        x1      device pointer or host pointer to input scalar that is overwritten.
        @param[in]
        y1      device pointer or host pointer to input scalar.
        @param[out]
        param   device vector or host vector of 5 elements defining the rotation.
                param[0] = flag
                param[1] = H11
                param[2] = H21
                param[3] = H12
                param[4] = H22
                The flag parameter defines the form of H:
                flag = -1 => H = ( H11 H12 H21 H22 )
                flag =  0 => H = ( 1.0 H12 H21 1.0 )
                flag =  1 => H = ( H11 1.0 -1.0 H22 )
                flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
                param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
    """
    _hipblasSrotmg__retval = hipblasStatus_t(chipblas.hipblasSrotmg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        <float *>DataHandle.from_pyobj(d1)._ptr,
        <float *>DataHandle.from_pyobj(d2)._ptr,
        <float *>DataHandle.from_pyobj(x1)._ptr,
        <const float *>DataHandle.from_pyobj(y1)._ptr,
        <float *>DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasSrotmg__retval,)


@cython.embedsignature(True)
def hipblasDrotmg(object handle, object d1, object d2, object x1, object y1, object param):
    """
    """
    _hipblasDrotmg__retval = hipblasStatus_t(chipblas.hipblasDrotmg(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,
        <double *>DataHandle.from_pyobj(d1)._ptr,
        <double *>DataHandle.from_pyobj(d2)._ptr,
        <double *>DataHandle.from_pyobj(x1)._ptr,
        <const double *>DataHandle.from_pyobj(y1)._ptr,
        <double *>DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasDrotmg__retval,)


@cython.embedsignature(True)
def hipblasSscal(object handle, int n, object alpha, object x, int incx):
    """! @{
        \brief BLAS Level 1 API

        \details
        scal  scales each element of vector x with scalar alpha.

            x := alpha * x

        - Supported precisions in rocBLAS : s,d,c,z,cs,zd
        - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        alpha     device pointer or host pointer for the scalar alpha.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    _hipblasSscal__retval = hipblasStatus_t(chipblas.hipblasSscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasSscal__retval,)


@cython.embedsignature(True)
def hipblasDscal(object handle, int n, object alpha, object x, int incx):
    """
    """
    _hipblasDscal__retval = hipblasStatus_t(chipblas.hipblasDscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDscal__retval,)


@cython.embedsignature(True)
def hipblasCscal(object handle, int n, object alpha, object x, int incx):
    """
    """
    _hipblasCscal__retval = hipblasStatus_t(chipblas.hipblasCscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCscal__retval,)


@cython.embedsignature(True)
def hipblasCsscal(object handle, int n, object alpha, object x, int incx):
    """
    """
    _hipblasCsscal__retval = hipblasStatus_t(chipblas.hipblasCsscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCsscal__retval,)


@cython.embedsignature(True)
def hipblasZscal(object handle, int n, object alpha, object x, int incx):
    """
    """
    _hipblasZscal__retval = hipblasStatus_t(chipblas.hipblasZscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZscal__retval,)


@cython.embedsignature(True)
def hipblasZdscal(object handle, int n, object alpha, object x, int incx):
    """
    """
    _hipblasZdscal__retval = hipblasStatus_t(chipblas.hipblasZdscal(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZdscal__retval,)


@cython.embedsignature(True)
def hipblasSswap(object handle, int n, object x, int incx, object y, int incy):
    """! @{
        \brief BLAS Level 1 API

        \details
        swap  interchanges vectors x and y.

            y := x; x := y

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    _hipblasSswap__retval = hipblasStatus_t(chipblas.hipblasSswap(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSswap__retval,)


@cython.embedsignature(True)
def hipblasDswap(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasDswap__retval = hipblasStatus_t(chipblas.hipblasDswap(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDswap__retval,)


@cython.embedsignature(True)
def hipblasCswap(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasCswap__retval = hipblasStatus_t(chipblas.hipblasCswap(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCswap__retval,)


@cython.embedsignature(True)
def hipblasZswap(object handle, int n, object x, int incx, object y, int incy):
    """
    """
    _hipblasZswap__retval = hipblasStatus_t(chipblas.hipblasZswap(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZswap__retval,)


@cython.embedsignature(True)
def hipblasSgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        gbmv performs one of the matrix-vector operations

            y := alpha*A*x    + beta*y,   or
            y := alpha*A**T*x + beta*y,   or
            y := alpha*A**H*x + beta*y,

        where alpha and beta are scalars, x and y are vectors and A is an
        m by n banded matrix with kl sub-diagonals and ku super-diagonals.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        trans     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not
        @param[in]
        m         [int]
                  number of rows of matrix A
        @param[in]
        n         [int]
                  number of columns of matrix A
        @param[in]
        kl        [int]
                  number of sub-diagonals of A
        @param[in]
        ku        [int]
                  number of super-diagonals of A
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
            AP    device pointer storing banded matrix A.
                  Leading (kl + ku + 1) by n part of the matrix contains the coefficients
                  of the banded matrix. The leading diagonal resides in row (ku + 1) with
                  the first super-diagonal above on the RHS of row ku. The first sub-diagonal
                  resides below on the LHS of row ku + 2. This propogates up and down across
                  sub/super-diagonals.
                    Ex: (m = n = 7; ku = 2, kl = 2)
                    1 2 3 0 0 0 0             0 0 3 3 3 3 3
                    4 1 2 3 0 0 0             0 2 2 2 2 2 2
                    5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                    0 5 4 1 2 3 0             4 4 4 4 4 4 0
                    0 0 5 4 1 2 0             5 5 5 5 5 0 0
                    0 0 0 5 4 1 2             0 0 0 0 0 0 0
                    0 0 0 0 5 4 1             0 0 0 0 0 0 0
                  Note that the empty elements which don't correspond to data will not
                  be referenced.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be >= (kl + ku + 1)
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasSgbmv__retval = hipblasStatus_t(chipblas.hipblasSgbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSgbmv__retval,)


@cython.embedsignature(True)
def hipblasDgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasDgbmv__retval = hipblasStatus_t(chipblas.hipblasDgbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDgbmv__retval,)


@cython.embedsignature(True)
def hipblasCgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasCgbmv__retval = hipblasStatus_t(chipblas.hipblasCgbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCgbmv__retval,)


@cython.embedsignature(True)
def hipblasZgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasZgbmv__retval = hipblasStatus_t(chipblas.hipblasZgbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZgbmv__retval,)


@cython.embedsignature(True)
def hipblasSgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        gemv performs one of the matrix-vector operations

            y := alpha*A*x    + beta*y,   or
            y := alpha*A**T*x + beta*y,   or
            y := alpha*A**H*x + beta*y,

        where alpha and beta are scalars, x and y are vectors and A is an
        m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        trans     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not
        @param[in]
        m         [int]
                  number of rows of matrix A
        @param[in]
        n         [int]
                  number of columns of matrix A
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasSgemv__retval = hipblasStatus_t(chipblas.hipblasSgemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSgemv__retval,)


@cython.embedsignature(True)
def hipblasDgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasDgemv__retval = hipblasStatus_t(chipblas.hipblasDgemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDgemv__retval,)


@cython.embedsignature(True)
def hipblasCgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasCgemv__retval = hipblasStatus_t(chipblas.hipblasCgemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCgemv__retval,)


@cython.embedsignature(True)
def hipblasZgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasZgemv__retval = hipblasStatus_t(chipblas.hipblasZgemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZgemv__retval,)


@cython.embedsignature(True)
def hipblasSger(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        ger,geru,gerc performs the matrix-vector operations

            A := A + alpha*x*y**T , OR
            A := A + alpha*x*y**H for gerc

        where alpha is a scalar, x and y are vectors, and A is an
        m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        m         [int]
                  the number of rows of the matrix A.
        @param[in]
        n         [int]
                  the number of columns of the matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    _hipblasSger__retval = hipblasStatus_t(chipblas.hipblasSger(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(y)._ptr,incy,
        <float *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSger__retval,)


@cython.embedsignature(True)
def hipblasDger(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    _hipblasDger__retval = hipblasStatus_t(chipblas.hipblasDger(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(y)._ptr,incy,
        <double *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDger__retval,)


@cython.embedsignature(True)
def hipblasCgeru(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    _hipblasCgeru__retval = hipblasStatus_t(chipblas.hipblasCgeru(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCgeru__retval,)


@cython.embedsignature(True)
def hipblasCgerc(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    _hipblasCgerc__retval = hipblasStatus_t(chipblas.hipblasCgerc(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCgerc__retval,)


@cython.embedsignature(True)
def hipblasZgeru(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    _hipblasZgeru__retval = hipblasStatus_t(chipblas.hipblasZgeru(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZgeru__retval,)


@cython.embedsignature(True)
def hipblasZgerc(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    _hipblasZgerc__retval = hipblasStatus_t(chipblas.hipblasZgerc(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZgerc__retval,)


@cython.embedsignature(True)
def hipblasChbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hbmv performs the matrix-vector operations

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian band matrix, with k super-diagonals.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
        @param[in]
        n         [int]
                  the order of the matrix A.
        @param[in]
        k         [int]
                  the number of super-diagonals of the matrix A. Must be >= 0.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A. Of dimension (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The leading (k + 1) by n part of A must contain the upper
                    triangular band part of the Hermitian matrix, with the leading
                    diagonal in row (k + 1), the first super-diagonal on the RHS
                    of row k, etc.
                    The top left k by x triangle of A will not be referenced.
                        Ex (upper, lda = n = 4, k = 1):
                        A                             Represented matrix
                        (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                        (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                        (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                        (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The leading (k + 1) by n part of A must contain the lower
                    triangular band part of the Hermitian matrix, with the leading
                    diagonal in row (1), the first sub-diagonal on the LHS of
                    row 2, etc.
                    The bottom right k by k triangle of A will not be referenced.
                        Ex (lower, lda = 2, n = 4, k = 1):
                        A                               Represented matrix
                        (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                        (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                        (0, 0) (6, 8) (3, 0) (7,-7)
                                                        (0, 0) (0, 0) (7, 7) (4, 0)

                  As a Hermitian matrix, the imaginary part of the main diagonal
                  of A will not be referenced and is assumed to be == 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. must be >= k + 1
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChbmv__retval = hipblasStatus_t(chipblas.hipblasChbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChbmv__retval,)


@cython.embedsignature(True)
def hipblasZhbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhbmv__retval = hipblasStatus_t(chipblas.hipblasZhbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhbmv__retval,)


@cython.embedsignature(True)
def hipblasChemv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hemv performs one of the matrix-vector operations

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
                  HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
        @param[in]
        n         [int]
                  the order of the matrix A.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing matrix A. Of dimension (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular part of A must contain
                    the upper triangular part of a Hermitian matrix. The lower
                    triangular part of A will not be referenced.
                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular part of A must contain
                    the lower triangular part of a Hermitian matrix. The upper
                    triangular part of A will not be referenced.
                  As a Hermitian matrix, the imaginary part of the main diagonal
                  of A will not be referenced and is assumed to be == 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. must be >= max(1, n)
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChemv__retval = hipblasStatus_t(chipblas.hipblasChemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChemv__retval,)


@cython.embedsignature(True)
def hipblasZhemv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhemv__retval = hipblasStatus_t(chipblas.hipblasZhemv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhemv__retval,)


@cython.embedsignature(True)
def hipblasCher(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        her performs the matrix-vector operations

            A := A + alpha*x*x**H

        where alpha is a real scalar, x is a vector, and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the specified triangular portion of
                  the Hermitian matrix A. Of size (lda * n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied. The lower
                    triangluar portion will not be touched.
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied. The upper
                    triangular portion will not be touched.
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be at least max(1, n).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCher__retval = hipblasStatus_t(chipblas.hipblasCher(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCher__retval,)


@cython.embedsignature(True)
def hipblasZher(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZher__retval = hipblasStatus_t(chipblas.hipblasZher(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZher__retval,)


@cython.embedsignature(True)
def hipblasCher2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        her2 performs the matrix-vector operations

            A := A + alpha*x*y**H + conj(alpha)*y*x**H

        where alpha is a complex scalar, x and y are vectors, and A is an
        n by n Hermitian matrix.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing the specified triangular portion of
                  the Hermitian matrix A. Of size (lda, n).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
                    portion of A will not be touched.
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
                    portion of A will not be touched.
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. Must be at least max(lda, 1).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCher2__retval = hipblasStatus_t(chipblas.hipblasCher2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCher2__retval,)


@cython.embedsignature(True)
def hipblasZher2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZher2__retval = hipblasStatus_t(chipblas.hipblasZher2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZher2__retval,)


@cython.embedsignature(True)
def hipblasChpmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpmv performs the matrix-vector operation

            y := alpha*A*x + beta*y

        where alpha and beta are scalars, x and y are n element vectors and A is an
        n by n Hermitian matrix, supplied in packed form (see description below).

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
        @param[in]
        n         [int]
                  the order of the matrix A, must be >= 0.
        @param[in]
        alpha     device pointer or host pointer to scalar alpha.
        @param[in]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (3, 2)
                            (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                            (3,-2) (5, 1) (6, 0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (3, 2)
                            (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                            (3,-2) (5, 1) (6, 0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        beta      device pointer or host pointer to scalar beta.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChpmv__retval = hipblasStatus_t(chipblas.hipblasChpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChpmv__retval,)


@cython.embedsignature(True)
def hipblasZhpmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhpmv__retval = hipblasStatus_t(chipblas.hipblasZhpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhpmv__retval,)


@cython.embedsignature(True)
def hipblasChpr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpr performs the matrix-vector operations

            A := A + alpha*x*x**H

        where alpha is a real scalar, x is a vector, and A is an
        n by n Hermitian matrix, supplied in packed form.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChpr__retval = hipblasStatus_t(chipblas.hipblasChpr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasChpr__retval,)


@cython.embedsignature(True)
def hipblasZhpr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhpr__retval = hipblasStatus_t(chipblas.hipblasZhpr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZhpr__retval,)


@cython.embedsignature(True)
def hipblasChpr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        hpr2 performs the matrix-vector operations

            A := A + alpha*x*y**H + conj(alpha)*y*x**H

        where alpha is a complex scalar, x and y are vectors, and A is an
        n by n Hermitian matrix, supplied in packed form.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the Hermitian matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
                            (1, 0) (2, 1) (4,9)
                            (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                            (4,-9) (5,-3) (6,0)
                Note that the imaginary part of the diagonal elements are not accessed and are assumed
                to be 0.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChpr2__retval = hipblasStatus_t(chipblas.hipblasChpr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasChpr2__retval,)


@cython.embedsignature(True)
def hipblasZhpr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhpr2__retval = hipblasStatus_t(chipblas.hipblasZhpr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZhpr2__retval,)


@cython.embedsignature(True)
def hipblasSsbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        sbmv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n symmetric banded matrix.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        k         [int]
                  specifies the number of sub- and super-diagonals
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        lda       [int]
                  specifies the leading dimension of matrix A
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSsbmv__retval = hipblasStatus_t(chipblas.hipblasSsbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSsbmv__retval,)


@cython.embedsignature(True)
def hipblasDsbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDsbmv__retval = hipblasStatus_t(chipblas.hipblasDsbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDsbmv__retval,)


@cython.embedsignature(True)
def hipblasSspmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        spmv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n packed symmetric matrix.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSspmv__retval = hipblasStatus_t(chipblas.hipblasSspmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSspmv__retval,)


@cython.embedsignature(True)
def hipblasDspmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDspmv__retval = hipblasStatus_t(chipblas.hipblasDspmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDspmv__retval,)


@cython.embedsignature(True)
def hipblasSspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        spr performs the matrix-vector operations

            A := A + alpha*x*x**T

        where alpha is a scalar, x is a vector, and A is an
        n by n symmetric matrix, supplied in packed form.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
                            1 2 4 7
                            2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            4 5 6 9
                            7 8 9 0
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(2) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
                            1 2 3 4
                            2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            3 6 8 9
                            4 7 9 0
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSspr__retval = hipblasStatus_t(chipblas.hipblasSspr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasSspr__retval,)


@cython.embedsignature(True)
def hipblasDspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDspr__retval = hipblasStatus_t(chipblas.hipblasDspr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasDspr__retval,)


@cython.embedsignature(True)
def hipblasCspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCspr__retval = hipblasStatus_t(chipblas.hipblasCspr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasCspr__retval,)


@cython.embedsignature(True)
def hipblasZspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZspr__retval = hipblasStatus_t(chipblas.hipblasZspr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZspr__retval,)


@cython.embedsignature(True)
def hipblasSspr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """! @{
        \brief BLAS Level 2 API

        \details
        spr2 performs the matrix-vector operation

            A := A + alpha*x*y**T + alpha*y*x**T

        where alpha is a scalar, x and y are vectors, and A is an
        n by n symmetric matrix, supplied in packed form.

        - Supported precisions in rocBLAS : s,d
        - Supported precisions in cuBLAS  : s,d

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
                  HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
        @param[in]
        n         [int]
                  the number of rows and columns of matrix A, must be at least 0.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP        device pointer storing the packed version of the specified triangular portion of
                  the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The upper triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(0,1)
                    AP(2) = A(1,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
                            1 2 4 7
                            2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            4 5 6 9
                            7 8 9 0
                if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The lower triangular portion of the symmetric matrix A is supplied.
                    The matrix is compacted so that AP contains the triangular portion column-by-column
                    so that:
                    AP(0) = A(0,0)
                    AP(1) = A(1,0)
                    AP(n) = A(2,1), etc.
                        Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
                            1 2 3 4
                            2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                            3 6 8 9
                            4 7 9 0
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSspr2__retval = hipblasStatus_t(chipblas.hipblasSspr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(y)._ptr,incy,
        <float *>DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasSspr2__retval,)


@cython.embedsignature(True)
def hipblasDspr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDspr2__retval = hipblasStatus_t(chipblas.hipblasDspr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(y)._ptr,incy,
        <double *>DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasDspr2__retval,)


@cython.embedsignature(True)
def hipblasSsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """! @{
        \brief BLAS Level 2 API

        \details
        symv performs the matrix-vector operation:

            y := alpha*A*x + beta*y,

        where alpha and beta are scalars, x and y are n element vectors and
        A should contain an upper or lower triangular n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        n         [int]
        @param[in]
        alpha
                  specifies the scalar alpha
        @param[in]
        AP         pointer storing matrix A on the GPU
        @param[in]
        lda       [int]
                  specifies the leading dimension of A
        @param[in]
        x         pointer storing vector x on the GPU
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x
        @param[in]
        beta      specifies the scalar beta
        @param[out]
        y         pointer storing vector y on the GPU
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSsymv__retval = hipblasStatus_t(chipblas.hipblasSsymv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSsymv__retval,)


@cython.embedsignature(True)
def hipblasDsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDsymv__retval = hipblasStatus_t(chipblas.hipblasDsymv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDsymv__retval,)


@cython.embedsignature(True)
def hipblasCsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCsymv__retval = hipblasStatus_t(chipblas.hipblasCsymv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCsymv__retval,)


@cython.embedsignature(True)
def hipblasZsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZsymv__retval = hipblasStatus_t(chipblas.hipblasZsymv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZsymv__retval,)


@cython.embedsignature(True)
def hipblasSsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        syr performs the matrix-vector operations

            A := A + alpha*x*x**T

        where alpha is a scalar, x is a vector, and A is an
        n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced

        @param[in]
        n         [int]
                  the number of rows and columns of matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSsyr__retval = hipblasStatus_t(chipblas.hipblasSsyr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSsyr__retval,)


@cython.embedsignature(True)
def hipblasDsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDsyr__retval = hipblasStatus_t(chipblas.hipblasDsyr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDsyr__retval,)


@cython.embedsignature(True)
def hipblasCsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCsyr__retval = hipblasStatus_t(chipblas.hipblasCsyr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCsyr__retval,)


@cython.embedsignature(True)
def hipblasZsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZsyr__retval = hipblasStatus_t(chipblas.hipblasZsyr(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZsyr__retval,)


@cython.embedsignature(True)
def hipblasSsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """! @{
        \brief BLAS Level 2 API

        \details
        syr2 performs the matrix-vector operations

            A := A + alpha*x*y**T + alpha*y*x**T

        where alpha is a scalar, x and y are vectors, and A is an
        n by n symmetric matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : No support

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced

        @param[in]
        n         [int]
                  the number of rows and columns of matrix A.
        @param[in]
        alpha
                  device pointer or host pointer to scalar alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSsyr2__retval = hipblasStatus_t(chipblas.hipblasSsyr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>DataHandle.from_pyobj(y)._ptr,incy,
        <float *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSsyr2__retval,)


@cython.embedsignature(True)
def hipblasDsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDsyr2__retval = hipblasStatus_t(chipblas.hipblasDsyr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>DataHandle.from_pyobj(y)._ptr,incy,
        <double *>DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDsyr2__retval,)


@cython.embedsignature(True)
def hipblasCsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCsyr2__retval = hipblasStatus_t(chipblas.hipblasCsyr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCsyr2__retval,)


@cython.embedsignature(True)
def hipblasZsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZsyr2__retval = hipblasStatus_t(chipblas.hipblasZsyr2(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZsyr2__retval,)


@cython.embedsignature(True)
def hipblasStbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tbmv performs one of the matrix-vector operations

            x := A*x      or
            x := A**T*x   or
            x := A**H*x,

        x is a vectors and A is a banded m by m matrix (see description below).

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
                  HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
        @param[in]
        transA     [hipblasOperation_t]
                  indicates whether matrix A is tranposed (conjugated) or not.
        @param[in]
        diag      [hipblasDiagType_t]
                  HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
                                         1's and is not referenced.
                  HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
        @param[in]
        m         [int]
                  the number of rows and columns of the matrix represented by A.
        @param[in]
        k         [int]
                  if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
                  of the matrix A.
                  if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
                  of the matrix A.
                  k must satisfy k > 0 && k < lda.
        @param[in]
        AP         device pointer storing banded triangular matrix A.
                  if uplo == HIPBLAS_FILL_MODE_UPPER:
                    The matrix represented is an upper banded triangular matrix
                    with the main diagonal and k super-diagonals, everything
                    else can be assumed to be 0.
                    The matrix is compacted so that the main diagonal resides on the k'th
                    row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                    with the k'th diagonal on the RHS of the 0'th row.
                       Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
                          1 6 9 0 0              0 0 9 8 7
                          0 2 7 8 0              0 6 7 8 9
                          0 0 3 8 7     ---->    1 2 3 4 5
                          0 0 0 4 9              0 0 0 0 0
                          0 0 0 0 5              0 0 0 0 0
                  if uplo == HIPBLAS_FILL_MODE_LOWER:
                    The matrix represnted is a lower banded triangular matrix
                    with the main diagonal and k sub-diagonals, everything else can be
                    assumed to be 0.
                    The matrix is compacted so that the main diagonal resides on the 0'th row,
                    working up to the k'th diagonal residing on the LHS of the k'th row.
                       Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
                          1 0 0 0 0              1 2 3 4 5
                          6 2 0 0 0              6 7 8 9 0
                          9 7 3 0 0     ---->    9 8 7 0 0
                          0 8 8 4 0              0 0 0 0 0
                          0 0 7 9 5              0 0 0 0 0
        @param[in]
        lda       [int]
                  specifies the leading dimension of A. lda must satisfy lda > k.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStbmv__retval = hipblasStatus_t(chipblas.hipblasStbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStbmv__retval,)


@cython.embedsignature(True)
def hipblasDtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtbmv__retval = hipblasStatus_t(chipblas.hipblasDtbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtbmv__retval,)


@cython.embedsignature(True)
def hipblasCtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtbmv__retval = hipblasStatus_t(chipblas.hipblasCtbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtbmv__retval,)


@cython.embedsignature(True)
def hipblasZtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtbmv__retval = hipblasStatus_t(chipblas.hipblasZtbmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtbmv__retval,)


@cython.embedsignature(True)
def hipblasStbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tbsv solves

             A*x = b or A**T*x = b or A**H*x = b,

        where x and b are vectors and A is a banded triangular matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]
                   HIPBLAS_OP_N: Solves A*x = b
                   HIPBLAS_OP_T: Solves A**T*x = b
                   HIPBLAS_OP_C: Solves A**H*x = b

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
                                           of A are not used in computations).
                HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.

        @param[in]
        n         [int]
                  n specifies the number of rows of b. n >= 0.
        @param[in]
        k         [int]
                  if(uplo == HIPBLAS_FILL_MODE_UPPER)
                    k specifies the number of super-diagonals of A.
                  if(uplo == HIPBLAS_FILL_MODE_LOWER)
                    k specifies the number of sub-diagonals of A.
                  k >= 0.

        @param[in]
        AP         device pointer storing the matrix A in banded format.

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda >= (k + 1).

        @param[inout]
        x         device pointer storing input vector b. Overwritten by the output vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStbsv__retval = hipblasStatus_t(chipblas.hipblasStbsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStbsv__retval,)


@cython.embedsignature(True)
def hipblasDtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtbsv__retval = hipblasStatus_t(chipblas.hipblasDtbsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtbsv__retval,)


@cython.embedsignature(True)
def hipblasCtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtbsv__retval = hipblasStatus_t(chipblas.hipblasCtbsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtbsv__retval,)


@cython.embedsignature(True)
def hipblasZtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtbsv__retval = hipblasStatus_t(chipblas.hipblasZtbsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtbsv__retval,)


@cython.embedsignature(True)
def hipblasStpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tpmv performs one of the matrix-vector operations

             x = A*x or x = A**T*x,

        where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.

        The vector x is overwritten.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of A. m >= 0.

        @param[in]
        AP       device pointer storing matrix A,
                of dimension at leat ( m * ( m + 1 ) / 2 ).
              Before entry with uplo = HIPBLAS_FILL_MODE_UPPER, the array A
              must contain the upper triangular matrix packed sequentially,
              column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
              a_{0,1} and a_{1, 1} respectively, and so on.
              Before entry with uplo = HIPBLAS_FILL_MODE_LOWER, the array A
              must contain the lower triangular matrix packed sequentially,
              column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
              a_{1,0} and a_{2,0} respectively, and so on.
              Note that when DIAG = HIPBLAS_DIAG_UNIT, the diagonal elements of A are
              not referenced, but are assumed to be unity.

        @param[in]
        x       device pointer storing vector x.

        @param[in]
        incx    [int]
                specifies the increment for the elements of x. incx must not be zero.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStpmv__retval = hipblasStatus_t(chipblas.hipblasStpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>DataHandle.from_pyobj(AP)._ptr,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStpmv__retval,)


@cython.embedsignature(True)
def hipblasDtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtpmv__retval = hipblasStatus_t(chipblas.hipblasDtpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>DataHandle.from_pyobj(AP)._ptr,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtpmv__retval,)


@cython.embedsignature(True)
def hipblasCtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtpmv__retval = hipblasStatus_t(chipblas.hipblasCtpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtpmv__retval,)


@cython.embedsignature(True)
def hipblasZtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtpmv__retval = hipblasStatus_t(chipblas.hipblasZtpmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtpmv__retval,)


@cython.embedsignature(True)
def hipblasStpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        tpsv solves

             A*x = b or A**T*x = b, or A**H*x = b,

        where x and b are vectors and A is a triangular matrix stored in the packed format.

        The input vector b is overwritten by the output vector x.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: Solves A*x = b
                HIPBLAS_OP_T: Solves A**T*x = b
                HIPBLAS_OP_C: Solves A**H*x = b

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
                                           of A are not used in computations).
                HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of b. m >= 0.

        @param[in]
        AP        device pointer storing the packed version of matrix A,
                  of dimension >= (n * (n + 1) / 2)

        @param[inout]
        x         device pointer storing vector b on input, overwritten by x on output.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStpsv__retval = hipblasStatus_t(chipblas.hipblasStpsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>DataHandle.from_pyobj(AP)._ptr,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStpsv__retval,)


@cython.embedsignature(True)
def hipblasDtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtpsv__retval = hipblasStatus_t(chipblas.hipblasDtpsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>DataHandle.from_pyobj(AP)._ptr,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtpsv__retval,)


@cython.embedsignature(True)
def hipblasCtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtpsv__retval = hipblasStatus_t(chipblas.hipblasCtpsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtpsv__retval,)


@cython.embedsignature(True)
def hipblasZtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtpsv__retval = hipblasStatus_t(chipblas.hipblasZtpsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtpsv__retval,)


@cython.embedsignature(True)
def hipblasStrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        trmv performs one of the matrix-vector operations

             x = A*x or x = A**T*x,

        where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.

        The vector x is overwritten.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of A. m >= 0.

        @param[in]
        AP        device pointer storing matrix A,
                  of dimension ( lda, m )

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda = max( 1, m ).

        @param[in]
        x         device pointer storing vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStrmv__retval = hipblasStatus_t(chipblas.hipblasStrmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStrmv__retval,)


@cython.embedsignature(True)
def hipblasDtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtrmv__retval = hipblasStatus_t(chipblas.hipblasDtrmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtrmv__retval,)


@cython.embedsignature(True)
def hipblasCtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtrmv__retval = hipblasStatus_t(chipblas.hipblasCtrmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtrmv__retval,)


@cython.embedsignature(True)
def hipblasZtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtrmv__retval = hipblasStatus_t(chipblas.hipblasZtrmv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtrmv__retval,)


@cython.embedsignature(True)
def hipblasStrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """! @{
        \brief BLAS Level 2 API

        \details
        trsv solves

             A*x = b or A**T*x = b,

        where x and b are vectors and A is a triangular matrix.

        The vector x is overwritten on b.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA     [hipblasOperation_t]

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m         [int]
                  m specifies the number of rows of b. m >= 0.

        @param[in]
        AP        device pointer storing matrix A,
                  of dimension ( lda, m )

        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
                  lda = max( 1, m ).

        @param[in]
        x         device pointer storing vector x.

        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStrsv__retval = hipblasStatus_t(chipblas.hipblasStrsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStrsv__retval,)


@cython.embedsignature(True)
def hipblasDtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtrsv__retval = hipblasStatus_t(chipblas.hipblasDtrsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtrsv__retval,)


@cython.embedsignature(True)
def hipblasCtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtrsv__retval = hipblasStatus_t(chipblas.hipblasCtrsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtrsv__retval,)


@cython.embedsignature(True)
def hipblasZtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtrsv__retval = hipblasStatus_t(chipblas.hipblasZtrsv(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtrsv__retval,)


@cython.embedsignature(True)
def hipblasHgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        gemm performs one of the matrix-matrix operations

            C = alpha*op( A )*op( B ) + beta*C,

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B and C are matrices, with
        op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.

        - Supported precisions in rocBLAS : h,s,d,c,z
        - Supported precisions in cuBLAS  : h,s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]

                  .
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A )
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B )
        @param[in]
        m         [int]
                  number or rows of matrices op( A ) and C
        @param[in]
        n         [int]
                  number of columns of matrices op( B ) and C
        @param[in]
        k         [int]
                  number of columns of matrix op( A ) and number of rows of matrix op( B )
        @param[in]
        alpha     device pointer or host pointer specifying the scalar alpha.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        BP         device pointer storing matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in]
        beta      device pointer or host pointer specifying the scalar beta.
        @param[in, out]
        CP         device pointer storing matrix C on the GPU.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasHgemm__retval = hipblasStatus_t(chipblas.hipblasHgemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasHalf *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasHgemm__retval,)


@cython.embedsignature(True)
def hipblasSgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasSgemm__retval = hipblasStatus_t(chipblas.hipblasSgemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSgemm__retval,)


@cython.embedsignature(True)
def hipblasDgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasDgemm__retval = hipblasStatus_t(chipblas.hipblasDgemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDgemm__retval,)


@cython.embedsignature(True)
def hipblasCgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasCgemm__retval = hipblasStatus_t(chipblas.hipblasCgemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCgemm__retval,)


@cython.embedsignature(True)
def hipblasZgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasZgemm__retval = hipblasStatus_t(chipblas.hipblasZgemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZgemm__retval,)


@cython.embedsignature(True)
def hipblasCherk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        herk performs one of the matrix-matrix operations for a Hermitian rank-k update

        C := alpha*op( A )*op( A )^H + beta*C

        where  alpha and beta are scalars, op(A) is an n by k matrix, and
        C is a n x n Hermitian matrix stored as either upper or lower.

            op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
            op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op(A) = A^H
                HIPBLAS_ON_N:  op(A) = A

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCherk__retval = hipblasStatus_t(chipblas.hipblasCherk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCherk__retval,)


@cython.embedsignature(True)
def hipblasZherk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZherk__retval = hipblasStatus_t(chipblas.hipblasZherk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZherk__retval,)


@cython.embedsignature(True)
def hipblasCherkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        herkx performs one of the matrix-matrix operations for a Hermitian rank-k update

        C := alpha*op( A )*op( B )^H + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
        C is a n x n Hermitian matrix stored as either upper or lower.
        This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.


            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
                HIPBLAS_OP_N:  op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP      pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCherkx__retval = hipblasStatus_t(chipblas.hipblasCherkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCherkx__retval,)


@cython.embedsignature(True)
def hipblasZherkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZherkx__retval = hipblasStatus_t(chipblas.hipblasZherkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZherkx__retval,)


@cython.embedsignature(True)
def hipblasCher2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update

        C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
        C is a n x n Hermitian matrix stored as either upper or lower.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
                HIPBLAS_OP_N:  op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                The imaginary component of the diagonal elements are not used but are set to zero unless quick return.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCher2k__retval = hipblasStatus_t(chipblas.hipblasCher2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCher2k__retval,)


@cython.embedsignature(True)
def hipblasZher2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZher2k__retval = hipblasStatus_t(chipblas.hipblasZher2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZher2k__retval,)


@cython.embedsignature(True)
def hipblasSsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        symm performs one of the matrix-matrix operations:

        C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
        C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

        where alpha and beta are scalars, B and C are m by n matrices, and
        A is a symmetric matrix stored as either upper or lower.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side  [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
                HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix

        @param[in]
        m       [int]
                m specifies the number of rows of B and C. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B and C. n >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A and B are not referenced.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                A is m by m if side == HIPBLAS_SIDE_LEFT
                A is n by n if side == HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                otherwise lda >= max( 1, n ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B. ldb >= max( 1, m )

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, m )
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasSsymm__retval = hipblasStatus_t(chipblas.hipblasSsymm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsymm__retval,)


@cython.embedsignature(True)
def hipblasDsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasDsymm__retval = hipblasStatus_t(chipblas.hipblasDsymm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsymm__retval,)


@cython.embedsignature(True)
def hipblasCsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasCsymm__retval = hipblasStatus_t(chipblas.hipblasCsymm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsymm__retval,)


@cython.embedsignature(True)
def hipblasZsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZsymm__retval = hipblasStatus_t(chipblas.hipblasZsymm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsymm__retval,)


@cython.embedsignature(True)
def hipblasSsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syrk performs one of the matrix-matrix operations for a symmetric rank-k update

        C := alpha*op( A )*op( A )^T + beta*C

        where  alpha and beta are scalars, op(A) is an n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.

            op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
            op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T: op(A) = A^T
                HIPBLAS_OP_N: op(A) = A
                HIPBLAS_OP_C: op(A) = A^T

                HIPBLAS_OP_C is not supported for complex types, see cherk
                and zherk.

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasSsyrk__retval = hipblasStatus_t(chipblas.hipblasSsyrk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyrk__retval,)


@cython.embedsignature(True)
def hipblasDsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasDsyrk__retval = hipblasStatus_t(chipblas.hipblasDsyrk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyrk__retval,)


@cython.embedsignature(True)
def hipblasCsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCsyrk__retval = hipblasStatus_t(chipblas.hipblasCsyrk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyrk__retval,)


@cython.embedsignature(True)
def hipblasZsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZsyrk__retval = hipblasStatus_t(chipblas.hipblasZsyrk(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyrk__retval,)


@cython.embedsignature(True)
def hipblasSsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update

        C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
                HIPBLAS_OP_N:           op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A) and op(B). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).
        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).
        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasSsyr2k__retval = hipblasStatus_t(chipblas.hipblasSsyr2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyr2k__retval,)


@cython.embedsignature(True)
def hipblasDsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasDsyr2k__retval = hipblasStatus_t(chipblas.hipblasDsyr2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyr2k__retval,)


@cython.embedsignature(True)
def hipblasCsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCsyr2k__retval = hipblasStatus_t(chipblas.hipblasCsyr2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyr2k__retval,)


@cython.embedsignature(True)
def hipblasZsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZsyr2k__retval = hipblasStatus_t(chipblas.hipblasZsyr2k(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyr2k__retval,)


@cython.embedsignature(True)
def hipblasSsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        syrkx performs one of the matrix-matrix operations for a symmetric rank-k update

        C := alpha*op( A )*op( B )^T + beta*C

        where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
        C is a symmetric n x n matrix stored as either upper or lower.
        This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.

            op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
            op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
                HIPBLAS_OP_N:           op( A ) = A, op( B ) = B

        @param[in]
        n       [int]
                n specifies the number of rows and columns of C. n >= 0.

        @param[in]
        k       [int]
                k specifies the number of columns of op(A) and op(B). k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and A need not be set before
                entry.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
                otherwise lda >= max( 1, k ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
                only the upper/lower triangular part is accessed.

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B.
                if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
                otherwise ldb >= max( 1, k ).

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, n ).
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasSsyrkx__retval = hipblasStatus_t(chipblas.hipblasSsyrkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyrkx__retval,)


@cython.embedsignature(True)
def hipblasDsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasDsyrkx__retval = hipblasStatus_t(chipblas.hipblasDsyrkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyrkx__retval,)


@cython.embedsignature(True)
def hipblasCsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasCsyrkx__retval = hipblasStatus_t(chipblas.hipblasCsyrkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyrkx__retval,)


@cython.embedsignature(True)
def hipblasZsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")
    _hipblasZsyrkx__retval = hipblasStatus_t(chipblas.hipblasZsyrkx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyrkx__retval,)


@cython.embedsignature(True)
def hipblasSgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        geam performs one of the matrix-matrix operations

            C = alpha*op( A ) + beta*op( B ),

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B and C are matrices, with
        op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A )
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B )
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        alpha     device pointer or host pointer specifying the scalar alpha.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        beta      device pointer or host pointer specifying the scalar beta.
        @param[in]
        BP         device pointer storing matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in, out]
        CP         device pointer storing matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasSgeam__retval = hipblasStatus_t(chipblas.hipblasSgeam(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(beta)._ptr,
        <const float *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSgeam__retval,)


@cython.embedsignature(True)
def hipblasDgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasDgeam__retval = hipblasStatus_t(chipblas.hipblasDgeam(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(beta)._ptr,
        <const double *>DataHandle.from_pyobj(BP)._ptr,ldb,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDgeam__retval,)


@cython.embedsignature(True)
def hipblasCgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasCgeam__retval = hipblasStatus_t(chipblas.hipblasCgeam(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCgeam__retval,)


@cython.embedsignature(True)
def hipblasZgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")
    _hipblasZgeam__retval = hipblasStatus_t(chipblas.hipblasZgeam(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZgeam__retval,)


@cython.embedsignature(True)
def hipblasChemm(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details

        hemm performs one of the matrix-matrix operations:

        C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
        C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

        where alpha and beta are scalars, B and C are m by n matrices, and
        A is a Hermitian matrix stored as either upper or lower.

        - Supported precisions in rocBLAS : c,z
        - Supported precisions in cuBLAS  : c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side  [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
                HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix

        @param[in]
        n       [int]
                n specifies the number of rows of B and C. n >= 0.

        @param[in]
        k       [int]
                n specifies the number of columns of B and C. k >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A and B are not referenced.

        @param[in]
        AP       pointer storing matrix A on the GPU.
                A is m by m if side == HIPBLAS_SIDE_LEFT
                A is n by n if side == HIPBLAS_SIDE_RIGHT
                Only the upper/lower triangular part is accessed.
                The imaginary component of the diagonal elements is not used.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                otherwise lda >= max( 1, n ).

        @param[in]
        BP       pointer storing matrix B on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldb     [int]
                ldb specifies the first dimension of B. ldb >= max( 1, m )

        @param[in]
        beta
                beta specifies the scalar beta. When beta is
                zero then C need not be set before entry.

        @param[in]
        CP       pointer storing matrix C on the GPU.
                Matrix dimension is m by n

        @param[in]
        ldc    [int]
               ldc specifies the first dimension of C. ldc >= max( 1, m )
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasChemm__retval = hipblasStatus_t(chipblas.hipblasChemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasChemm__retval,)


@cython.embedsignature(True)
def hipblasZhemm(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")
    _hipblasZhemm__retval = hipblasStatus_t(chipblas.hipblasZhemm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZhemm__retval,)


@cython.embedsignature(True)
def hipblasStrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """! @{
        \brief BLAS Level 3 API

        \details

        trmm performs one of the matrix-matrix operations

        B := alpha*op( A )*B,   or   B := alpha*B*op( A )

        where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
        non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.


        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                Specifies whether op(A) multiplies B from the left or right as follows:
                HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
                HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).

        @param[in]
        uplo    [hipblasFillMode_t]
                Specifies whether the matrix A is an upper or lower triangular matrix as follows:
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                Specifies the form of op(A) to be used in the matrix multiplication as follows:
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_OP_C:  op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                Specifies whether or not A is unit triangular as follows:
                HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha
                alpha specifies the scalar alpha. When alpha is
                zero then A is not referenced and B need not be set before
                entry.

        @param[in]
        AP       Device pointer to matrix A on the GPU.
                A has dimension ( lda, k ), where k is m
                when  side == HIPBLAS_SIDE_LEFT  and
                is  n  when  side == HIPBLAS_SIDE_RIGHT.

            When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
            upper triangular part of the array  A must contain the upper
            triangular matrix  and the strictly lower triangular part of
            A is not referenced.

            When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
            lower triangular part of the array  A must contain the lower
            triangular matrix  and the strictly upper triangular part of
            A is not referenced.

            Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
            A  are not referenced either,  but are assumed to be  unity.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[inout]
        BP       Device pointer to the first matrix B_0 on the GPU.
                On entry,  the leading  m by n part of the array  B must
               contain the matrix  B,  and  on exit  is overwritten  by the
               transformed matrix.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStrmm__retval = hipblasStatus_t(chipblas.hipblasStrmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasStrmm__retval,)


@cython.embedsignature(True)
def hipblasDtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtrmm__retval = hipblasStatus_t(chipblas.hipblasDtrmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasDtrmm__retval,)


@cython.embedsignature(True)
def hipblasCtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtrmm__retval = hipblasStatus_t(chipblas.hipblasCtrmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasCtrmm__retval,)


@cython.embedsignature(True)
def hipblasZtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtrmm__retval = hipblasStatus_t(chipblas.hipblasZtrmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasZtrmm__retval,)


@cython.embedsignature(True)
def hipblasStrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """! @{
        \brief BLAS Level 3 API

        \details

        trsm solves

            op(A)*X = alpha*B or  X*op(A) = alpha*B,

        where alpha is a scalar, X and B are m by n matrices,
        A is triangular matrix and op(A) is one of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

        The matrix X is overwritten on B.

        Note about memory allocation:
        When trsm is launched with a k evenly divisible by the internal block size of 128,
        and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
        memory found in the handle to increase overall performance. This memory can be managed by using
        the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
        used for temporary storage will default to 1 MB and may result in chunking, which in turn may
        reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
        to the desired chunk of right hand sides to be used at a time.

        (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
                HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_OP_C: op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha
                device pointer or host pointer specifying the scalar alpha. When alpha is
                &zero then A is not referenced and B need not be set before
                entry.

        @param[in]
        AP       device pointer storing matrix A.
                of dimension ( lda, k ), where k is m
                when  HIPBLAS_SIDE_LEFT  and
                is  n  when  HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[in,out]
        BP       device pointer storing matrix B.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStrsm__retval = hipblasStatus_t(chipblas.hipblasStrsm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>DataHandle.from_pyobj(alpha)._ptr,
        <float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasStrsm__retval,)


@cython.embedsignature(True)
def hipblasDtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtrsm__retval = hipblasStatus_t(chipblas.hipblasDtrsm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>DataHandle.from_pyobj(alpha)._ptr,
        <double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasDtrsm__retval,)


@cython.embedsignature(True)
def hipblasCtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtrsm__retval = hipblasStatus_t(chipblas.hipblasCtrsm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasCtrsm__retval,)


@cython.embedsignature(True)
def hipblasZtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtrsm__retval = hipblasStatus_t(chipblas.hipblasZtrsm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasZtrsm__retval,)


@cython.embedsignature(True)
def hipblasStrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """! @{
        \brief BLAS Level 3 API

        \details
        trtri  compute the inverse of a matrix A, namely, invA

            and write the result into invA;

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : No support

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        uplo      [hipblasFillMode_t]
                  specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
                  if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
                  if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
        @param[in]
        diag      [hipblasDiagType_t]
                  = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
                  = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
        @param[in]
        n         [int]
                  size of matrix A and invA
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[out]
        invA      device pointer storing matrix invA.
        @param[in]
        ldinvA    [int]
                  specifies the leading dimension of invA.
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasStrtri__retval = hipblasStatus_t(chipblas.hipblasStrtri(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>DataHandle.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasStrtri__retval,)


@cython.embedsignature(True)
def hipblasDtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasDtrtri__retval = hipblasStatus_t(chipblas.hipblasDtrtri(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>DataHandle.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasDtrtri__retval,)


@cython.embedsignature(True)
def hipblasCtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasCtrtri__retval = hipblasStatus_t(chipblas.hipblasCtrtri(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasCtrtri__retval,)


@cython.embedsignature(True)
def hipblasZtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """
    """
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")
    _hipblasZtrtri__retval = hipblasStatus_t(chipblas.hipblasZtrtri(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasZtrtri__retval,)


@cython.embedsignature(True)
def hipblasSdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """! @{
        \brief BLAS Level 3 API

        \details
        dgmm performs one of the matrix-matrix operations

            C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
            C = diag(x) * A if side == HIPBLAS_SIDE_LEFT

        where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
        and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
        if side == HIPBLAS_SIDE_LEFT.

        - Supported precisions in rocBLAS : s,d,c,z
        - Supported precisions in cuBLAS  : s,d,c,z

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        side      [hipblasSideMode_t]
                  specifies the side of diag(x)
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        AP         device pointer storing matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        incx      [int]
                  specifies the increment between values of x
        @param[in, out]
        CP         device pointer storing matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    _hipblasSdgmm__retval = hipblasStatus_t(chipblas.hipblasSdgmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const float *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>DataHandle.from_pyobj(x)._ptr,incx,
        <float *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSdgmm__retval,)


@cython.embedsignature(True)
def hipblasDdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    _hipblasDdgmm__retval = hipblasStatus_t(chipblas.hipblasDdgmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const double *>DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>DataHandle.from_pyobj(x)._ptr,incx,
        <double *>DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDdgmm__retval,)


@cython.embedsignature(True)
def hipblasCdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    _hipblasCdgmm__retval = hipblasStatus_t(chipblas.hipblasCdgmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCdgmm__retval,)


@cython.embedsignature(True)
def hipblasZdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")
    _hipblasZdgmm__retval = hipblasStatus_t(chipblas.hipblasZdgmm(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZdgmm__retval,)


@cython.embedsignature(True)
def hipblasSgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """! @{
        \brief SOLVER API

        \details
        getrf computes the LU factorization of a general n-by-n matrix A
        using partial pivoting with row interchanges. The LU factorization can
        be done without pivoting if ipiv is passed as a nullptr.

        In the case that ipiv is not null, the factorization has the form:

        \f[
            A = PLU
        \f]

        where P is a permutation matrix, L is lower triangular with unit
        diagonal elements, and U is upper triangular.

        In the case that ipiv is null, the factorization is done without pivoting:

        \f[
            A = LU
        \f]

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z

        @param[in]
        handle    hipblasHandle_t.
        @param[in]
        n         int. n >= 0.\n
                  The number of columns and rows of the matrix A.
        @param[inout]
        A         pointer to type. Array on the GPU of dimension lda*n.\n
                  On entry, the n-by-n matrix A to be factored.
                  On exit, the factors L and U from the factorization.
                  The unit diagonal elements of L are not stored.
        @param[in]
        lda       int. lda >= n.\n
                  Specifies the leading dimension of A.
        @param[out]
        ipiv      pointer to int. Array on the GPU of dimension n.\n
                  The vector of pivot indices. Elements of ipiv are 1-based indices.
                  For 1 <= i <= n, the row i of the
                  matrix was interchanged with row ipiv[i].
                  Matrix P of the factorization can be derived from ipiv.
                  The factorization here can be done without pivoting if ipiv is passed
                  in as a nullptr.
        @param[out]
        info      pointer to a int on the GPU.\n
                  If info = 0, successful exit.
                  If info = j > 0, U is singular. U[j,j] is the first zero pivot.
    """
    _hipblasSgetrf__retval = hipblasStatus_t(chipblas.hipblasSgetrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <float *>DataHandle.from_pyobj(A)._ptr,lda,
        <int *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgetrf__retval,)


@cython.embedsignature(True)
def hipblasDgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasDgetrf__retval = hipblasStatus_t(chipblas.hipblasDgetrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <double *>DataHandle.from_pyobj(A)._ptr,lda,
        <int *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgetrf__retval,)


@cython.embedsignature(True)
def hipblasCgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasCgetrf__retval = hipblasStatus_t(chipblas.hipblasCgetrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        <int *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgetrf__retval,)


@cython.embedsignature(True)
def hipblasZgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasZgetrf__retval = hipblasStatus_t(chipblas.hipblasZgetrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        <int *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgetrf__retval,)


@cython.embedsignature(True)
def hipblasSgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """! @{
        \brief SOLVER API

        \details
        getrs solves a system of n linear equations on n variables in its factorized form.

        It solves one of the following systems, depending on the value of trans:

        \f[
            \begin{array}{cl}
            A X = B & \: \text{not transposed,}\\
            A^T X = B & \: \text{transposed, or}\\
            A^H X = B & \: \text{conjugate transposed.}
            \end{array}
        \f]

        Matrix A is defined by its triangular factors as returned by \ref hipblasSgetrf "getrf".

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z


        @param[in]
        handle      hipblasHandle_t.
        @param[in]
        trans       hipblasOperation_t.\n
                    Specifies the form of the system of equations.
        @param[in]
        n           int. n >= 0.\n
                    The order of the system, i.e. the number of columns and rows of A.
        @param[in]
        nrhs        int. nrhs >= 0.\n
                    The number of right hand sides, i.e., the number of columns
                    of the matrix B.
        @param[in]
        A           pointer to type. Array on the GPU of dimension lda*n.\n
                    The factors L and U of the factorization A = P*L*U returned by \ref hipblasSgetrf "getrf".
        @param[in]
        lda         int. lda >= n.\n
                    The leading dimension of A.
        @param[in]
        ipiv        pointer to int. Array on the GPU of dimension n.\n
                    The pivot indices returned by \ref hipblasSgetrf "getrf".
        @param[in,out]
        B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                    On entry, the right hand side matrix B.
                    On exit, the solution matrix X.
        @param[in]
        ldb         int. ldb >= n.\n
                    The leading dimension of B.
        @param[out]
        info      pointer to a int on the host.\n
                  If info = 0, successful exit.
                  If info = j < 0, the j-th argument is invalid.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasSgetrs__retval = hipblasStatus_t(chipblas.hipblasSgetrs(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <float *>DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>DataHandle.from_pyobj(ipiv)._ptr,
        <float *>DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgetrs__retval,)


@cython.embedsignature(True)
def hipblasDgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasDgetrs__retval = hipblasStatus_t(chipblas.hipblasDgetrs(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <double *>DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>DataHandle.from_pyobj(ipiv)._ptr,
        <double *>DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgetrs__retval,)


@cython.embedsignature(True)
def hipblasCgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasCgetrs__retval = hipblasStatus_t(chipblas.hipblasCgetrs(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        <const int *>DataHandle.from_pyobj(ipiv)._ptr,
        hipblasComplex.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgetrs__retval,)


@cython.embedsignature(True)
def hipblasZgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasZgetrs__retval = hipblasStatus_t(chipblas.hipblasZgetrs(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        <const int *>DataHandle.from_pyobj(ipiv)._ptr,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgetrs__retval,)


@cython.embedsignature(True)
def hipblasSgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """! @{
        \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
        matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
        factorization computed by "GELQF").

        \details
        Depending on the value of trans, the problem solved by this function is either of the form

        \f[
            \begin{array}{cl}
            A X = B & \: \text{not transposed, or}\\
            A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
            \end{array}
        \f]

        If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
        and a least-squares solution approximating X is found by minimizing

        \f[
            || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
        \f]

        If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
        and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : currently unsupported

        @param[in]
        handle      hipblasHandle_t.
        @param[in]
        trans       hipblasOperation_t.\n
                    Specifies the form of the system of equations.
        @param[in]
        m           int. m >= 0.\n
                    The number of rows of matrix A.
        @param[in]
        n           int. n >= 0.\n
                    The number of columns of matrix A.
        @param[in]
        nrhs        int. nrhs >= 0.\n
                    The number of columns of matrices B and X;
                    i.e., the columns on the right hand side.
        @param[inout]
        A           pointer to type. Array on the GPU of dimension lda*n.\n
                    On entry, the matrix A.
                    On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").
        @param[in]
        lda         int. lda >= m.\n
                    Specifies the leading dimension of matrix A.
        @param[inout]
        B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
                    On entry, the matrix B.
                    On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
                    the overdetermined cases) stored as columns.
        @param[in]
        ldb         int. ldb >= max(m,n).\n
                    Specifies the leading dimension of matrix B.
        @param[out]
        info        pointer to an int on the host.\n
                    If info = 0, successful exit.
                    If info = j < 0, the j-th argument is invalid.
        @param[out]
        deviceInfo  pointer to int on the GPU.\n
                    If info = 0, successful exit.
                    If info = i > 0, the solution could not be computed because input matrix A is
                    rank deficient; the i-th diagonal element of its triangular factor is zero.
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasSgels__retval = hipblasStatus_t(chipblas.hipblasSgels(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <float *>DataHandle.from_pyobj(A)._ptr,lda,
        <float *>DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr,
        <int *>DataHandle.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasSgels__retval,)


@cython.embedsignature(True)
def hipblasDgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasDgels__retval = hipblasStatus_t(chipblas.hipblasDgels(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <double *>DataHandle.from_pyobj(A)._ptr,lda,
        <double *>DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr,
        <int *>DataHandle.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasDgels__retval,)


@cython.embedsignature(True)
def hipblasCgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasCgels__retval = hipblasStatus_t(chipblas.hipblasCgels(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        hipblasComplex.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr,
        <int *>DataHandle.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasCgels__retval,)


@cython.embedsignature(True)
def hipblasZgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """
    """
    if not isinstance(trans,hipblasOperation_t):
        raise TypeError("argument 'trans' must be of type 'hipblasOperation_t'")
    _hipblasZgels__retval = hipblasStatus_t(chipblas.hipblasZgels(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,
        <int *>DataHandle.from_pyobj(info)._ptr,
        <int *>DataHandle.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasZgels__retval,)


@cython.embedsignature(True)
def hipblasSgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """! @{
        \brief SOLVER API

        \details
        geqrf computes a QR factorization of a general m-by-n matrix A.

        The factorization has the form

        \f[
            A = Q\left[\begin{array}{c}
            R\\
            0
            \end{array}\right]
        \f]

        where R is upper triangular (upper trapezoidal if m < n), and Q is
        a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

        \f[
            Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
        \f]

        Each Householder matrix \f$H_i\f$ is given by

        \f[
            H_i = I - \text{ipiv}[i] \cdot v_i v_i'
        \f]

        where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.

        - Supported precisions in rocSOLVER : s,d,c,z
        - Supported precisions in cuBLAS    : s,d,c,z

        @param[in]
        handle    hipblasHandle_t.
        @param[in]
        m         int. m >= 0.\n
                  The number of rows of the matrix A.
        @param[in]
        n         int. n >= 0.\n
                  The number of columns of the matrix A.
        @param[inout]
        A         pointer to type. Array on the GPU of dimension lda*n.\n
                  On entry, the m-by-n matrix to be factored.
                  On exit, the elements on and above the diagonal contain the
                  factor R; the elements below the diagonal are the last m - i elements
                  of Householder vector v_i.
        @param[in]
        lda       int. lda >= m.\n
                  Specifies the leading dimension of A.
        @param[out]
        ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
                  The Householder scalars.
        @param[out]
        info      pointer to a int on the host.\n
                  If info = 0, successful exit.
                  If info = j < 0, the j-th argument is invalid.
    """
    _hipblasSgeqrf__retval = hipblasStatus_t(chipblas.hipblasSgeqrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        <float *>DataHandle.from_pyobj(A)._ptr,lda,
        <float *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgeqrf__retval,)


@cython.embedsignature(True)
def hipblasDgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasDgeqrf__retval = hipblasStatus_t(chipblas.hipblasDgeqrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        <double *>DataHandle.from_pyobj(A)._ptr,lda,
        <double *>DataHandle.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgeqrf__retval,)


@cython.embedsignature(True)
def hipblasCgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasCgeqrf__retval = hipblasStatus_t(chipblas.hipblasCgeqrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        hipblasComplex.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgeqrf__retval,)


@cython.embedsignature(True)
def hipblasZgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """
    """
    _hipblasZgeqrf__retval = hipblasStatus_t(chipblas.hipblasZgeqrf(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(ipiv)._ptr,
        <int *>DataHandle.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgeqrf__retval,)


@cython.embedsignature(True)
def hipblasGemmEx(object handle, object transA, object transB, int m, int n, int k, object alpha, object A, object aType, int lda, object B, object bType, int ldb, object beta, object C, object cType, int ldc, object computeType, object algo):
    """! \brief BLAS EX API

        \details
        gemmEx performs one of the matrix-matrix operations

            C = alpha*op( A )*op( B ) + beta*C,

        where op( X ) is one of

            op( X ) = X      or
            op( X ) = X**T   or
            op( X ) = X**H,

        alpha and beta are scalars, and A, B, and C are matrices, with
        op( A ) an m by k matrix, op( B ) a k by n matrix and C is a m by n matrix.

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
        data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
        format for a given device as documented in rocBLAS.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        transA    [hipblasOperation_t]
                  specifies the form of op( A ).
        @param[in]
        transB    [hipblasOperation_t]
                  specifies the form of op( B ).
        @param[in]
        m         [int]
                  matrix dimension m.
        @param[in]
        n         [int]
                  matrix dimension n.
        @param[in]
        k         [int]
                  matrix dimension k.
        @param[in]
        alpha     [const void *]
                  device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
        @param[in]
        A         [void *]
                  device pointer storing matrix A.
        @param[in]
        aType    [hipblasDatatype_t]
                  specifies the datatype of matrix A.
        @param[in]
        lda       [int]
                  specifies the leading dimension of A.
        @param[in]
        B         [void *]
                  device pointer storing matrix B.
        @param[in]
        bType    [hipblasDatatype_t]
                  specifies the datatype of matrix B.
        @param[in]
        ldb       [int]
                  specifies the leading dimension of B.
        @param[in]
        beta      [const void *]
                  device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
        @param[in]
        C         [void *]
                  device pointer storing matrix C.
        @param[in]
        cType    [hipblasDatatype_t]
                  specifies the datatype of matrix C.
        @param[in]
        ldc       [int]
                  specifies the leading dimension of C.
        @param[in]
        computeType
                  [hipblasDatatype_t]
                  specifies the datatype of computation.
        @param[in]
        algo      [hipblasGemmAlgo_t]
                  enumerant specifying the algorithm type.
    """
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(transB,hipblasOperation_t):
        raise TypeError("argument 'transB' must be of type 'hipblasOperation_t'")                    
    if not isinstance(aType,hipblasDatatype_t):
        raise TypeError("argument 'aType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(bType,hipblasDatatype_t):
        raise TypeError("argument 'bType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(cType,hipblasDatatype_t):
        raise TypeError("argument 'cType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(computeType,hipblasDatatype_t):
        raise TypeError("argument 'computeType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(algo,hipblasGemmAlgo_t):
        raise TypeError("argument 'algo' must be of type 'hipblasGemmAlgo_t'")
    _hipblasGemmEx__retval = hipblasStatus_t(chipblas.hipblasGemmEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const void *>DataHandle.from_pyobj(alpha)._ptr,
        <const void *>DataHandle.from_pyobj(A)._ptr,aType.value,lda,
        <const void *>DataHandle.from_pyobj(B)._ptr,bType.value,ldb,
        <const void *>DataHandle.from_pyobj(beta)._ptr,
        <void *>DataHandle.from_pyobj(C)._ptr,cType.value,ldc,computeType.value,algo.value))    # fully specified
    return (_hipblasGemmEx__retval,)


@cython.embedsignature(True)
def hipblasTrsmEx(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object A, int lda, object B, int ldb, object invA, int invAsize, object computeType):
    """! BLAS EX API

        \details
        trsmEx solves

            op(A)*X = alpha*B or X*op(A) = alpha*B,

        where alpha is a scalar, X and B are m by n matrices,
        A is triangular matrix and op(A) is one of

            op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

        The matrix X is overwritten on B.

        This function gives the user the ability to reuse the invA matrix between runs.
        If invA == NULL, hipblasTrsmEx will automatically calculate invA on every run.

        Setting up invA:
        The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
        matrix A, followed by any smaller diagonal block that remains.
        To set up invA it is recommended that hipblasTrtriBatched be used with matrix A as the input.

        Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
        HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in invA
        should be passed as invAsize.

        To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of
        matrix A. Below are the restricted parameters:
          - n = 128
          - ldinvA = 128
          - stride_invA = 128x128
          - batchCount = k / 128,

        Then any remaining block may be added:
          - n = k % 128
          - invA = invA + stride_invA * previousBatchCount
          - ldinvA = 128
          - batchCount = 1

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.

        @param[in]
        side    [hipblasSideMode_t]
                HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
                HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.

        @param[in]
        uplo    [hipblasFillMode_t]
                HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
                HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.

        @param[in]
        transA  [hipblasOperation_t]
                HIPBLAS_OP_N: op(A) = A.
                HIPBLAS_OP_T: op(A) = A^T.
                HIPBLAS_ON_C: op(A) = A^H.

        @param[in]
        diag    [hipblasDiagType_t]
                HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
                HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.

        @param[in]
        m       [int]
                m specifies the number of rows of B. m >= 0.

        @param[in]
        n       [int]
                n specifies the number of columns of B. n >= 0.

        @param[in]
        alpha   [void *]
                device pointer or host pointer specifying the scalar alpha. When alpha is
                &zero then A is not referenced, and B need not be set before
                entry.

        @param[in]
        A       [void *]
                device pointer storing matrix A.
                of dimension ( lda, k ), where k is m
                when HIPBLAS_SIDE_LEFT and
                is n when HIPBLAS_SIDE_RIGHT
                only the upper/lower triangular part is accessed.

        @param[in]
        lda     [int]
                lda specifies the first dimension of A.
                if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
                if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).

        @param[in, out]
        B       [void *]
                device pointer storing matrix B.
                B is of dimension ( ldb, n ).
                Before entry, the leading m by n part of the array B must
                contain the right-hand side matrix B, and on exit is
                overwritten by the solution matrix X.

        @param[in]
        ldb    [int]
               ldb specifies the first dimension of B. ldb >= max( 1, m ).

        @param[in]
        invA    [void *]
                device pointer storing the inverse diagonal blocks of A.
                invA is of dimension ( ld_invA, k ), where k is m
                when HIPBLAS_SIDE_LEFT and
                is n when HIPBLAS_SIDE_RIGHT.
                ld_invA must be equal to 128.

        @param[in]
        invAsize [int]
                invAsize specifies the number of elements of device memory in invA.

        @param[in]
        computeType [hipblasDatatype_t]
                specifies the datatype of computation
    """
    if not isinstance(side,hipblasSideMode_t):
        raise TypeError("argument 'side' must be of type 'hipblasSideMode_t'")                    
    if not isinstance(uplo,hipblasFillMode_t):
        raise TypeError("argument 'uplo' must be of type 'hipblasFillMode_t'")                    
    if not isinstance(transA,hipblasOperation_t):
        raise TypeError("argument 'transA' must be of type 'hipblasOperation_t'")                    
    if not isinstance(diag,hipblasDiagType_t):
        raise TypeError("argument 'diag' must be of type 'hipblasDiagType_t'")                    
    if not isinstance(computeType,hipblasDatatype_t):
        raise TypeError("argument 'computeType' must be of type 'hipblasDatatype_t'")
    _hipblasTrsmEx__retval = hipblasStatus_t(chipblas.hipblasTrsmEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const void *>DataHandle.from_pyobj(alpha)._ptr,
        <void *>DataHandle.from_pyobj(A)._ptr,lda,
        <void *>DataHandle.from_pyobj(B)._ptr,ldb,
        <const void *>DataHandle.from_pyobj(invA)._ptr,invAsize,computeType.value))    # fully specified
    return (_hipblasTrsmEx__retval,)


@cython.embedsignature(True)
def hipblasAxpyEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, object y, object yType, int incy, object executionType):
    """! \brief BLAS EX API

        \details
        axpyEx computes constant alpha multiplied by vector x, plus vector y

            y := alpha * x + y

            - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        alpha     device pointer or host pointer to specify the scalar alpha.
        @param[in]
        alphaType [hipblasDatatype_t]
                  specifies the datatype of alpha.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[inout]
        y         device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
              specifies the datatype of vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(alphaType,hipblasDatatype_t):
        raise TypeError("argument 'alphaType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasAxpyEx__retval = hipblasStatus_t(chipblas.hipblasAxpyEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <const void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,yType.value,incy,executionType.value))    # fully specified
    return (_hipblasAxpyEx__retval,)


@cython.embedsignature(True)
def hipblasDotEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object result, object resultType, object executionType):
    """! @{
        \brief BLAS EX API

        \details
        dotEx  performs the dot product of vectors x and y

            result = x * y;

        dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y

            result = conjugate (x) * y;

            - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x and y.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[in]
        y         device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
              specifies the datatype of vector y.
        @param[in]
        incy      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the dot product.
                  return is 0.0 if n <= 0.
        @param[in]
        resultType [hipblasDatatype_t]
                    specifies the datatype of the result.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasDotEx__retval = hipblasStatus_t(chipblas.hipblasDotEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <void *>DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotEx__retval,)


@cython.embedsignature(True)
def hipblasDotcEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object result, object resultType, object executionType):
    """
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasDotcEx__retval = hipblasStatus_t(chipblas.hipblasDotcEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <void *>DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotcEx__retval,)


@cython.embedsignature(True)
def hipblasNrm2Ex(object handle, int n, object x, object xType, int incx, object result, object resultType, object executionType):
    """! \brief BLAS_EX API

        \details
        nrm2Ex computes the euclidean norm of a real or complex vector

                  result := sqrt( x'*x ) for real vectors
                  result := sqrt( x**H*x ) for complex vectors

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.


        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of the vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of y.
        @param[inout]
        result
                  device pointer or host pointer to store the nrm2 product.
                  return is 0.0 if n, incx<=0.
        @param[in]
        resultType [hipblasDatatype_t]
                    specifies the datatype of the result.
        @param[in]
        executionType [hipblasDatatype_t]
                      specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(resultType,hipblasDatatype_t):
        raise TypeError("argument 'resultType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasNrm2Ex__retval = hipblasStatus_t(chipblas.hipblasNrm2Ex(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasNrm2Ex__retval,)


@cython.embedsignature(True)
def hipblasRotEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object c, object s, object csType, object executionType):
    """! \brief BLAS EX API

        \details
        rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
            Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

        In the case where cs_type is real:
            x := c * x + s * y
                y := c * y - s * x

        In the case where cs_type is complex, the imaginary part of c is ignored:
            x := real(c) * x + s * y
                y := real(c) * y - conj(s) * x

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle  [hipblasHandle_t]
                handle to the hipblas library context queue.
        @param[in]
        n       [int]
                number of elements in the x and y vectors.
        @param[inout]
        x       device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx    [int]
                specifies the increment between elements of x.
        @param[inout]
        y       device pointer storing vector y.
        @param[in]
        yType [hipblasDatatype_t]
               specifies the datatype of vector y.
        @param[in]
        incy    [int]
                specifies the increment between elements of y.
        @param[in]
        c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
        @param[in]
        s       device pointer or host pointer storing scalar sine component of the rotation matrix.
        @param[in]
        csType [hipblasDatatype_t]
                specifies the datatype of c and s.
        @param[in]
        executionType [hipblasDatatype_t]
                       specifies the datatype of computation.
    """
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(yType,hipblasDatatype_t):
        raise TypeError("argument 'yType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(csType,hipblasDatatype_t):
        raise TypeError("argument 'csType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasRotEx__retval = hipblasStatus_t(chipblas.hipblasRotEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <const void *>DataHandle.from_pyobj(c)._ptr,
        <const void *>DataHandle.from_pyobj(s)._ptr,csType.value,executionType.value))    # fully specified
    return (_hipblasRotEx__retval,)


@cython.embedsignature(True)
def hipblasScalEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, object executionType):
    """! \brief BLAS EX API

        \details
        scalEx  scales each element of vector x with scalar alpha.

            x := alpha * x

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

        @param[in]
        handle    [hipblasHandle_t]
                  handle to the hipblas library context queue.
        @param[in]
        n         [int]
                  the number of elements in x.
        @param[in]
        alpha     device pointer or host pointer for the scalar alpha.
        @param[in]
        alphaType [hipblasDatatype_t]
                   specifies the datatype of alpha.
        @param[inout]
        x         device pointer storing vector x.
        @param[in]
        xType [hipblasDatatype_t]
               specifies the datatype of vector x.
        @param[in]
        incx      [int]
                  specifies the increment for the elements of x.
        @param[in]
        executionType [hipblasDatatype_t]
                       specifies the datatype of computation.
    """
    if not isinstance(alphaType,hipblasDatatype_t):
        raise TypeError("argument 'alphaType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(xType,hipblasDatatype_t):
        raise TypeError("argument 'xType' must be of type 'hipblasDatatype_t'")                    
    if not isinstance(executionType,hipblasDatatype_t):
        raise TypeError("argument 'executionType' must be of type 'hipblasDatatype_t'")
    _hipblasScalEx__retval = hipblasStatus_t(chipblas.hipblasScalEx(
        <chipblas.hipblasHandle_t>DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <void *>DataHandle.from_pyobj(x)._ptr,xType.value,incx,executionType.value))    # fully specified
    return (_hipblasScalEx__retval,)


@cython.embedsignature(True)
def hipblasStatusToString(object status):
    """! HIPBLAS Auxiliary API

        \details
        hipblasStatusToString

        Returns string representing hipblasStatus_t value

        @param[in]
        status  [hipblasStatus_t]
                hipBLAS status to convert to string
    """
    if not isinstance(status,hipblasStatus_t):
        raise TypeError("argument 'status' must be of type 'hipblasStatus_t'")
    cdef const char * _hipblasStatusToString__retval = chipblas.hipblasStatusToString(status.value)    # fully specified
