# AMD_COPYRIGHT
import cython
import ctypes
import enum
hipblasVersionMajor = chipblas.hipblasVersionMajor

hipblaseVersionMinor = chipblas.hipblaseVersionMinor

hipblasVersionMinor = chipblas.hipblasVersionMinor

hipblasVersionPatch = chipblas.hipblasVersionPatch

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
            wrapper._ptr = <chipblas.hipblasBfloat16*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasBfloat16*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
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
        cdef chipblas.hipblasBfloat16* ptr
        hipblasBfloat16.__allocate(&ptr)
        return hipblasBfloat16.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        hipblasBfloat16.__allocate(&self._ptr)
        self.ptr_owner = True
        attribs = self.PROPERTIES()
        used_attribs = set()
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<hipblasBfloat16 object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
    def get_data(self, i):
        """Get value ``data`` of ``self._ptr[i]``.
        """
        return self._ptr[i].data
    def set_data(self, i, unsigned short value):
        """Set value ``data`` of ``self._ptr[i]``.
        """
        self._ptr[i].data = value
    @property
    def data(self):
        return self.get_data(0)
    @data.setter
    def data(self, unsigned short value):
        self.set_data(0,value)

    @staticmethod
    def PROPERTIES():
        return ["data"]

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
            wrapper._ptr = <chipblas.hipblasComplex*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasComplex*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
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
        cdef chipblas.hipblasComplex* ptr
        hipblasComplex.__allocate(&ptr)
        return hipblasComplex.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        hipblasComplex.__allocate(&self._ptr)
        self.ptr_owner = True
        attribs = self.PROPERTIES()
        used_attribs = set()
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<hipblasComplex object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
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

    @staticmethod
    def PROPERTIES():
        return ["x","y"]

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
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipblas.hipblasDoubleComplex*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
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
        cdef chipblas.hipblasDoubleComplex* ptr
        hipblasDoubleComplex.__allocate(&ptr)
        return hipblasDoubleComplex.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        hipblasDoubleComplex.__allocate(&self._ptr)
        self.ptr_owner = True
        attribs = self.PROPERTIES()
        used_attribs = set()
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<hipblasDoubleComplex object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
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

    @staticmethod
    def PROPERTIES():
        return ["x","y"]

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


class _hipblasStatus_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasStatus_t(_hipblasStatus_t__Base):
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
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasOperation_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasOperation_t(_hipblasOperation_t__Base):
    HIPBLAS_OP_N = chipblas.HIPBLAS_OP_N
    HIPBLAS_OP_T = chipblas.HIPBLAS_OP_T
    HIPBLAS_OP_C = chipblas.HIPBLAS_OP_C
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasPointerMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasPointerMode_t(_hipblasPointerMode_t__Base):
    HIPBLAS_POINTER_MODE_HOST = chipblas.HIPBLAS_POINTER_MODE_HOST
    HIPBLAS_POINTER_MODE_DEVICE = chipblas.HIPBLAS_POINTER_MODE_DEVICE
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasFillMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasFillMode_t(_hipblasFillMode_t__Base):
    HIPBLAS_FILL_MODE_UPPER = chipblas.HIPBLAS_FILL_MODE_UPPER
    HIPBLAS_FILL_MODE_LOWER = chipblas.HIPBLAS_FILL_MODE_LOWER
    HIPBLAS_FILL_MODE_FULL = chipblas.HIPBLAS_FILL_MODE_FULL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasDiagType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasDiagType_t(_hipblasDiagType_t__Base):
    HIPBLAS_DIAG_NON_UNIT = chipblas.HIPBLAS_DIAG_NON_UNIT
    HIPBLAS_DIAG_UNIT = chipblas.HIPBLAS_DIAG_UNIT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasSideMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasSideMode_t(_hipblasSideMode_t__Base):
    HIPBLAS_SIDE_LEFT = chipblas.HIPBLAS_SIDE_LEFT
    HIPBLAS_SIDE_RIGHT = chipblas.HIPBLAS_SIDE_RIGHT
    HIPBLAS_SIDE_BOTH = chipblas.HIPBLAS_SIDE_BOTH
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasDatatype_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasDatatype_t(_hipblasDatatype_t__Base):
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
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasGemmAlgo_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasGemmAlgo_t(_hipblasGemmAlgo_t__Base):
    HIPBLAS_GEMM_DEFAULT = chipblas.HIPBLAS_GEMM_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasAtomicsMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasAtomicsMode_t(_hipblasAtomicsMode_t__Base):
    HIPBLAS_ATOMICS_NOT_ALLOWED = chipblas.HIPBLAS_ATOMICS_NOT_ALLOWED
    HIPBLAS_ATOMICS_ALLOWED = chipblas.HIPBLAS_ATOMICS_ALLOWED
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipblasInt8Datatype_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipblasInt8Datatype_t(_hipblasInt8Datatype_t__Base):
    HIPBLAS_INT8_DATATYPE_DEFAULT = chipblas.HIPBLAS_INT8_DATATYPE_DEFAULT
    HIPBLAS_INT8_DATATYPE_INT8 = chipblas.HIPBLAS_INT8_DATATYPE_INT8
    HIPBLAS_INT8_DATATYPE_PACK_INT8x4 = chipblas.HIPBLAS_INT8_DATATYPE_PACK_INT8x4
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def hipblasCreate():
    """Create hipblas handle. */
    """
    handle = hip._util.types.DataHandle.from_ptr(NULL)
    _hipblasCreate__retval = hipblasStatus_t(chipblas.hipblasCreate(
        <void **>&handle._ptr))    # fully specified
    return (_hipblasCreate__retval,handle)


@cython.embedsignature(True)
def hipblasDestroy(object handle):
    """Destroys the library context created using hipblasCreate() */
    """
    _hipblasDestroy__retval = hipblasStatus_t(chipblas.hipblasDestroy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr))    # fully specified
    return (_hipblasDestroy__retval,)


@cython.embedsignature(True)
def hipblasSetStream(object handle, object streamId):
    """Set stream for handle */
    """
    _hipblasSetStream__retval = hipblasStatus_t(chipblas.hipblasSetStream(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        ihipStream_t.from_pyobj(streamId)._ptr))    # fully specified
    return (_hipblasSetStream__retval,)


@cython.embedsignature(True)
def hipblasGetStream(object handle, object streamId):
    """Get stream[0] for handle */
    """
    _hipblasGetStream__retval = hipblasStatus_t(chipblas.hipblasGetStream(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipStream_t*>hip._util.types.DataHandle.from_pyobj(streamId)._ptr))    # fully specified
    return (_hipblasGetStream__retval,)


@cython.embedsignature(True)
def hipblasSetPointerMode(object handle, object mode):
    """Set hipblas pointer mode */
    """
    if not isinstance(mode,_hipblasPointerMode_t__Base):
        raise TypeError("argument 'mode' must be of type '_hipblasPointerMode_t__Base'")
    _hipblasSetPointerMode__retval = hipblasStatus_t(chipblas.hipblasSetPointerMode(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,mode.value))    # fully specified
    return (_hipblasSetPointerMode__retval,)


@cython.embedsignature(True)
def hipblasGetPointerMode(object handle, object mode):
    """Get hipblas pointer mode */
    """
    _hipblasGetPointerMode__retval = hipblasStatus_t(chipblas.hipblasGetPointerMode(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipblasPointerMode_t *>hip._util.types.DataHandle.from_pyobj(mode)._ptr))    # fully specified
    return (_hipblasGetPointerMode__retval,)


@cython.embedsignature(True)
def hipblasSetInt8Datatype(object handle, object int8Type):
    """Set hipblas int8 Datatype */
    """
    if not isinstance(int8Type,_hipblasInt8Datatype_t__Base):
        raise TypeError("argument 'int8Type' must be of type '_hipblasInt8Datatype_t__Base'")
    _hipblasSetInt8Datatype__retval = hipblasStatus_t(chipblas.hipblasSetInt8Datatype(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,int8Type.value))    # fully specified
    return (_hipblasSetInt8Datatype__retval,)


@cython.embedsignature(True)
def hipblasGetInt8Datatype(object handle, object int8Type):
    """Get hipblas int8 Datatype*/
    """
    _hipblasGetInt8Datatype__retval = hipblasStatus_t(chipblas.hipblasGetInt8Datatype(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipblasInt8Datatype_t *>hip._util.types.DataHandle.from_pyobj(int8Type)._ptr))    # fully specified
    return (_hipblasGetInt8Datatype__retval,)


@cython.embedsignature(True)
def hipblasSetVector(int n, int elemSize, object x, int incx, object y, int incy):
    """copy vector from host to device

    Args:
       n: [int]
          number of elements in the vector
       elemSize: [int]
          Size of both vectors in bytes
       x: pointer to vector on the host
       incx: [int]
          specifies the increment for the elements of the vector
       y: pointer to vector on the device
       incy: [int]
          specifies the increment for the elements of the vector
          ******************************************************************
    """
    _hipblasSetVector__retval = hipblasStatus_t(chipblas.hipblasSetVector(n,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSetVector__retval,)


@cython.embedsignature(True)
def hipblasGetVector(int n, int elemSize, object x, int incx, object y, int incy):
    """copy vector from device to host

    Args:
       n: [int]
          number of elements in the vector
       elemSize: [int]
          Size of both vectors in bytes
       x: pointer to vector on the device
       incx: [int]
          specifies the increment for the elements of the vector
       y: pointer to vector on the host
       incy: [int]
          specifies the increment for the elements of the vector
          ******************************************************************
    """
    _hipblasGetVector__retval = hipblasStatus_t(chipblas.hipblasGetVector(n,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasGetVector__retval,)


@cython.embedsignature(True)
def hipblasSetMatrix(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb):
    """copy matrix from host to device

    Args:
       rows: [int]
          number of rows in matrices
       cols: [int]
          number of columns in matrices
       elemSize: [int]
          number of bytes per element in the matrix
       AP: pointer to matrix on the host
       lda: [int]
          specifies the leading dimension of A, lda >= rows
       BP: pointer to matrix on the GPU
       ldb: [int]
          specifies the leading dimension of B, ldb >= rows
          ******************************************************************
    """
    _hipblasSetMatrix__retval = hipblasStatus_t(chipblas.hipblasSetMatrix(rows,cols,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasSetMatrix__retval,)


@cython.embedsignature(True)
def hipblasGetMatrix(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb):
    """copy matrix from device to host

    Args:
       rows: [int]
          number of rows in matrices
       cols: [int]
          number of columns in matrices
       elemSize: [int]
          number of bytes per element in the matrix
       AP: pointer to matrix on the GPU
       lda: [int]
          specifies the leading dimension of A, lda >= rows
       BP: pointer to matrix on the host
       ldb: [int]
          specifies the leading dimension of B, ldb >= rows
          ******************************************************************
    """
    _hipblasGetMatrix__retval = hipblasStatus_t(chipblas.hipblasGetMatrix(rows,cols,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasGetMatrix__retval,)


@cython.embedsignature(True)
def hipblasSetVectorAsync(int n, int elemSize, object x, int incx, object y, int incy, object stream):
    """asynchronously copy vector from host to device

    hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.

    Args:
       n: [int]
          number of elements in the vector
       elemSize: [int]
          number of bytes per element in the matrix
       x: pointer to vector on the host
       incx: [int]
          specifies the increment for the elements of the vector
       y: pointer to vector on the device
       incy: [int]
          specifies the increment for the elements of the vector
       stream: specifies the stream into which this transfer request is queued
          ******************************************************************
    """
    _hipblasSetVectorAsync__retval = hipblasStatus_t(chipblas.hipblasSetVectorAsync(n,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasSetVectorAsync__retval,)


@cython.embedsignature(True)
def hipblasGetVectorAsync(int n, int elemSize, object x, int incx, object y, int incy, object stream):
    """asynchronously copy vector from device to host

    hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.

    Args:
       n: [int]
          number of elements in the vector
       elemSize: [int]
          number of bytes per element in the matrix
       x: pointer to vector on the device
       incx: [int]
          specifies the increment for the elements of the vector
       y: pointer to vector on the host
       incy: [int]
          specifies the increment for the elements of the vector
       stream: specifies the stream into which this transfer request is queued
          ******************************************************************
    """
    _hipblasGetVectorAsync__retval = hipblasStatus_t(chipblas.hipblasGetVectorAsync(n,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasGetVectorAsync__retval,)


@cython.embedsignature(True)
def hipblasSetMatrixAsync(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb, object stream):
    """asynchronously copy matrix from host to device

    hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.

    Args:
       rows: [int]
          number of rows in matrices
       cols: [int]
          number of columns in matrices
       elemSize: [int]
          number of bytes per element in the matrix
       AP: pointer to matrix on the host
       lda: [int]
          specifies the leading dimension of A, lda >= rows
       BP: pointer to matrix on the GPU
       ldb: [int]
          specifies the leading dimension of B, ldb >= rows
       stream: specifies the stream into which this transfer request is queued
          ******************************************************************
    """
    _hipblasSetMatrixAsync__retval = hipblasStatus_t(chipblas.hipblasSetMatrixAsync(rows,cols,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasSetMatrixAsync__retval,)


@cython.embedsignature(True)
def hipblasGetMatrixAsync(int rows, int cols, int elemSize, object AP, int lda, object BP, int ldb, object stream):
    """asynchronously copy matrix from device to host

    hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
    Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.

    Args:
       rows: [int]
          number of rows in matrices
       cols: [int]
          number of columns in matrices
       elemSize: [int]
          number of bytes per element in the matrix
       AP: pointer to matrix on the GPU
       lda: [int]
          specifies the leading dimension of A, lda >= rows
       BP: pointer to matrix on the host
       ldb: [int]
          specifies the leading dimension of B, ldb >= rows
       stream: specifies the stream into which this transfer request is queued
          ******************************************************************
    """
    _hipblasGetMatrixAsync__retval = hipblasStatus_t(chipblas.hipblasGetMatrixAsync(rows,cols,elemSize,
        <const void *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipblasGetMatrixAsync__retval,)


@cython.embedsignature(True)
def hipblasSetAtomicsMode(object handle, object atomics_mode):
    """Set hipblasSetAtomicsMode*/
    """
    if not isinstance(atomics_mode,_hipblasAtomicsMode_t__Base):
        raise TypeError("argument 'atomics_mode' must be of type '_hipblasAtomicsMode_t__Base'")
    _hipblasSetAtomicsMode__retval = hipblasStatus_t(chipblas.hipblasSetAtomicsMode(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,atomics_mode.value))    # fully specified
    return (_hipblasSetAtomicsMode__retval,)


@cython.embedsignature(True)
def hipblasGetAtomicsMode(object handle, object atomics_mode):
    """Get hipblasSetAtomicsMode*/
    """
    _hipblasGetAtomicsMode__retval = hipblasStatus_t(chipblas.hipblasGetAtomicsMode(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipblasAtomicsMode_t *>hip._util.types.DataHandle.from_pyobj(atomics_mode)._ptr))    # fully specified
    return (_hipblasGetAtomicsMode__retval,)


@cython.embedsignature(True)
def hipblasIsamax(object handle, int n, object x, int incx, object result):
    """BLAS Level 1 API

    amax finds the first index of the element of maximum magnitude of a vector x.

    - Supported precisions in rocBLAS : s,d,c,z.
    - Supported precisions in cuBLAS  : s,d,c,z.

    @param[inout]
    result
              device pointer or host pointer to store the amax index.
              return is 0.0 if n, incx<=0.
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of y.
    """
    _hipblasIsamax__retval = hipblasStatus_t(chipblas.hipblasIsamax(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamax__retval,)


@cython.embedsignature(True)
def hipblasIdamax(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIdamax__retval = hipblasStatus_t(chipblas.hipblasIdamax(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamax__retval,)


@cython.embedsignature(True)
def hipblasIcamax(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIcamax__retval = hipblasStatus_t(chipblas.hipblasIcamax(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamax__retval,)


@cython.embedsignature(True)
def hipblasIzamax(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIzamax__retval = hipblasStatus_t(chipblas.hipblasIzamax(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamax__retval,)


@cython.embedsignature(True)
def hipblasIsamaxBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """BLAS Level 1 API

    amaxBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z.
    - Supported precisions in cuBLAS  : No support.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       batchCount: [int]
          number of instances in the batch, must be > 0.
       result: device or host array of pointers of batchCount size for results.
          return is 0 if n, incx<=0.
          ******************************************************************
    """
    _hipblasIsamaxBatched__retval = hipblasStatus_t(chipblas.hipblasIsamaxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamaxBatched__retval,)


@cython.embedsignature(True)
def hipblasIdamaxBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIdamaxBatched__retval = hipblasStatus_t(chipblas.hipblasIdamaxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamaxBatched__retval,)


@cython.embedsignature(True)
def hipblasIcamaxBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIcamaxBatched__retval = hipblasStatus_t(chipblas.hipblasIcamaxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamaxBatched__retval,)


@cython.embedsignature(True)
def hipblasIzamaxBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIzamaxBatched__retval = hipblasStatus_t(chipblas.hipblasIzamaxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamaxBatched__retval,)


@cython.embedsignature(True)
def hipblasIsamaxStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """BLAS Level 1 API

    amaxStridedBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       stridex: [hipblasStride]
          specifies the pointer increment between one x_i and the next x_(i + 1).
       batchCount: [int]
          number of instances in the batch
       result: device or host pointer for storing contiguous batchCount results.
          return is 0 if n <= 0, incx<=0.
          ******************************************************************
    """
    _hipblasIsamaxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIsamaxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamaxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIdamaxStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIdamaxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIdamaxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamaxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIcamaxStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIcamaxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIcamaxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamaxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIzamaxStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIzamaxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIzamaxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamaxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIsamin(object handle, int n, object x, int incx, object result):
    """BLAS Level 1 API

    amin finds the first index of the element of minimum magnitude of a vector x.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    result
              device pointer or host pointer to store the amin index.
              return is 0.0 if n, incx<=0.
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of y.
    """
    _hipblasIsamin__retval = hipblasStatus_t(chipblas.hipblasIsamin(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsamin__retval,)


@cython.embedsignature(True)
def hipblasIdamin(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIdamin__retval = hipblasStatus_t(chipblas.hipblasIdamin(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdamin__retval,)


@cython.embedsignature(True)
def hipblasIcamin(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIcamin__retval = hipblasStatus_t(chipblas.hipblasIcamin(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcamin__retval,)


@cython.embedsignature(True)
def hipblasIzamin(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasIzamin__retval = hipblasStatus_t(chipblas.hipblasIzamin(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzamin__retval,)


@cython.embedsignature(True)
def hipblasIsaminBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """BLAS Level 1 API

    aminBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       batchCount: [int]
          number of instances in the batch, must be > 0.
       result: device or host pointers to array of batchCount size for results.
          return is 0 if n, incx<=0.
          ******************************************************************
    """
    _hipblasIsaminBatched__retval = hipblasStatus_t(chipblas.hipblasIsaminBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsaminBatched__retval,)


@cython.embedsignature(True)
def hipblasIdaminBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIdaminBatched__retval = hipblasStatus_t(chipblas.hipblasIdaminBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdaminBatched__retval,)


@cython.embedsignature(True)
def hipblasIcaminBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIcaminBatched__retval = hipblasStatus_t(chipblas.hipblasIcaminBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcaminBatched__retval,)


@cython.embedsignature(True)
def hipblasIzaminBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasIzaminBatched__retval = hipblasStatus_t(chipblas.hipblasIzaminBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzaminBatched__retval,)


@cython.embedsignature(True)
def hipblasIsaminStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """BLAS Level 1 API

    aminStridedBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       stridex: [hipblasStride]
          specifies the pointer increment between one x_i and the next x_(i + 1)
       batchCount: [int]
          number of instances in the batch
       result: device or host pointer to array for storing contiguous batchCount results.
          return is 0 if n <= 0, incx<=0.
          ******************************************************************
    """
    _hipblasIsaminStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIsaminStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIsaminStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIdaminStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIdaminStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIdaminStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIdaminStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIcaminStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIcaminStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIcaminStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIcaminStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasIzaminStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasIzaminStridedBatched__retval = hipblasStatus_t(chipblas.hipblasIzaminStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <int *>hip._util.types.ListOfInt.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasIzaminStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSasum(object handle, int n, object x, int incx, object result):
    """BLAS Level 1 API

    asum computes the sum of the magnitudes of elements of a real vector x,
         or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    result
              device pointer or host pointer to store the asum product.
              return is 0.0 if n <= 0.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x. incx must be > 0.
    """
    _hipblasSasum__retval = hipblasStatus_t(chipblas.hipblasSasum(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSasum__retval,)


@cython.embedsignature(True)
def hipblasDasum(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasDasum__retval = hipblasStatus_t(chipblas.hipblasDasum(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDasum__retval,)


@cython.embedsignature(True)
def hipblasScasum(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasScasum__retval = hipblasStatus_t(chipblas.hipblasScasum(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScasum__retval,)


@cython.embedsignature(True)
def hipblasDzasum(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasDzasum__retval = hipblasStatus_t(chipblas.hipblasDzasum(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDzasum__retval,)


@cython.embedsignature(True)
def hipblasSasumBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """BLAS Level 1 API

    asumBatched computes the sum of the magnitudes of the elements in a batch of real vectors x_i,
        or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
        vector, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       batchCount: [int]
          number of instances in the batch.
       result: device array or host array of batchCount size for results.
          return is 0.0 if n, incx<=0.
          ******************************************************************
    """
    _hipblasSasumBatched__retval = hipblasStatus_t(chipblas.hipblasSasumBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSasumBatched__retval,)


@cython.embedsignature(True)
def hipblasDasumBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasDasumBatched__retval = hipblasStatus_t(chipblas.hipblasDasumBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDasumBatched__retval,)


@cython.embedsignature(True)
def hipblasScasumBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasScasumBatched__retval = hipblasStatus_t(chipblas.hipblasScasumBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScasumBatched__retval,)


@cython.embedsignature(True)
def hipblasDzasumBatched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasDzasumBatched__retval = hipblasStatus_t(chipblas.hipblasDzasumBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDzasumBatched__retval,)


@cython.embedsignature(True)
def hipblasSasumStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """BLAS Level 1 API

    asumStridedBatched computes the sum of the magnitudes of elements of a real vectors x_i,
        or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
        vector, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each vector x_i
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       batchCount: [int]
          number of instances in the batch
       result: device pointer or host pointer to array for storing contiguous batchCount results.
          return is 0.0 if n, incx<=0.
          ******************************************************************
    """
    _hipblasSasumStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSasumStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSasumStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDasumStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasDasumStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDasumStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDasumStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasScasumStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasScasumStridedBatched__retval = hipblasStatus_t(chipblas.hipblasScasumStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScasumStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDzasumStridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasDzasumStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDzasumStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDzasumStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasHaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """BLAS Level 1 API

    axpy   computes constant alpha multiplied by vector x, plus vector y

        y := alpha * x + y

    - Supported precisions in rocBLAS : h,s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    incy      [int]
              specifies the increment for the elements of y.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       alpha: device pointer or host pointer to specify the scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
    """
    _hipblasHaxpy__retval = hipblasStatus_t(chipblas.hipblasHaxpy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasHaxpy__retval,)


@cython.embedsignature(True)
def hipblasSaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasSaxpy__retval = hipblasStatus_t(chipblas.hipblasSaxpy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSaxpy__retval,)


@cython.embedsignature(True)
def hipblasDaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasDaxpy__retval = hipblasStatus_t(chipblas.hipblasDaxpy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDaxpy__retval,)


@cython.embedsignature(True)
def hipblasCaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasCaxpy__retval = hipblasStatus_t(chipblas.hipblasCaxpy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCaxpy__retval,)


@cython.embedsignature(True)
def hipblasZaxpy(object handle, int n, object alpha, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasZaxpy__retval = hipblasStatus_t(chipblas.hipblasZaxpy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZaxpy__retval,)


@cython.embedsignature(True)
def hipblasHaxpyBatched(object handle, int n, object alpha, object x, int incx, object y, int incy, int batchCount):
    """BLAS Level 1 API

    axpyBatched   compute y := alpha * x + y over a set of batched vectors.

    - Supported precisions in rocBLAS : h,s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    incy      [int]
              specifies the increment for the elements of y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       alpha: specifies the scalar alpha.
       x: pointer storing vector x on the GPU.
       incx: [int]
          specifies the increment for the elements of x.
       y: pointer storing vector y on the GPU.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasHaxpyBatched__retval = hipblasStatus_t(chipblas.hipblasHaxpyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <unsigned short *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasHaxpyBatched__retval,)


@cython.embedsignature(True)
def hipblasSaxpyBatched(object handle, int n, object alpha, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasSaxpyBatched__retval = hipblasStatus_t(chipblas.hipblasSaxpyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSaxpyBatched__retval,)


@cython.embedsignature(True)
def hipblasDaxpyBatched(object handle, int n, object alpha, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasDaxpyBatched__retval = hipblasStatus_t(chipblas.hipblasDaxpyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDaxpyBatched__retval,)


@cython.embedsignature(True)
def hipblasCaxpyBatched(object handle, int n, object alpha, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasCaxpyBatched__retval = hipblasStatus_t(chipblas.hipblasCaxpyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCaxpyBatched__retval,)


@cython.embedsignature(True)
def hipblasZaxpyBatched(object handle, int n, object alpha, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasZaxpyBatched__retval = hipblasStatus_t(chipblas.hipblasZaxpyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZaxpyBatched__retval,)


@cython.embedsignature(True)
def hipblasHaxpyStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """BLAS Level 1 API

    axpyStridedBatched   compute y := alpha * x + y over a set of strided batched vectors.

    - Supported precisions in rocBLAS : h,s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    incy      [int]
              specifies the increment for the elements of y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
       alpha: specifies the scalar alpha.
       x: pointer storing vector x on the GPU.
       incx: [int]
          specifies the increment for the elements of x.
       stridex: [hipblasStride]
          specifies the increment between vectors of x.
       y: pointer storing vector y on the GPU.
       stridey: [hipblasStride]
          specifies the increment between vectors of y.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasHaxpyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasHaxpyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasHaxpyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSaxpyStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasSaxpyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSaxpyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSaxpyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDaxpyStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasDaxpyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDaxpyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDaxpyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCaxpyStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasCaxpyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCaxpyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCaxpyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZaxpyStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasZaxpyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZaxpyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZaxpyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasScopy(object handle, int n, object x, int incx, object y, int incy):
    """BLAS Level 1 API

    copy  copies each element x[i] into y[i], for  i = 1 , ... , n

        y := x,

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x to be copied to y.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    _hipblasScopy__retval = hipblasStatus_t(chipblas.hipblasScopy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasScopy__retval,)


@cython.embedsignature(True)
def hipblasDcopy(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasDcopy__retval = hipblasStatus_t(chipblas.hipblasDcopy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDcopy__retval,)


@cython.embedsignature(True)
def hipblasCcopy(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasCcopy__retval = hipblasStatus_t(chipblas.hipblasCcopy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCcopy__retval,)


@cython.embedsignature(True)
def hipblasZcopy(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasZcopy__retval = hipblasStatus_t(chipblas.hipblasZcopy(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZcopy__retval,)


@cython.embedsignature(True)
def hipblasScopyBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """BLAS Level 1 API

    copyBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount

        y_i := x_i,

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i to be copied to y_i.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each vector x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each vector y_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasScopyBatched__retval = hipblasStatus_t(chipblas.hipblasScopyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasScopyBatched__retval,)


@cython.embedsignature(True)
def hipblasDcopyBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasDcopyBatched__retval = hipblasStatus_t(chipblas.hipblasDcopyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDcopyBatched__retval,)


@cython.embedsignature(True)
def hipblasCcopyBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasCcopyBatched__retval = hipblasStatus_t(chipblas.hipblasCcopyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCcopyBatched__retval,)


@cython.embedsignature(True)
def hipblasZcopyBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasZcopyBatched__retval = hipblasStatus_t(chipblas.hipblasZcopyBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZcopyBatched__retval,)


@cython.embedsignature(True)
def hipblasScopyStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """BLAS Level 1 API

    copyStridedBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount

        y_i := x_i,

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i to be copied to y_i.
       x: device pointer to the first vector (x_1) in the batch.
       incx: [int]
          specifies the increments for the elements of vectors x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       y: device pointer to the first vector (y_1) in the batch.
       incy: [int]
          specifies the increment for the elements of vectors y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stride_y, however the user should
          take care to ensure that stride_y is of appropriate size, for a typical
          case this means stride_y >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasScopyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasScopyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasScopyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDcopyStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasDcopyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDcopyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDcopyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCcopyStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasCcopyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCcopyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCcopyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZcopyStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasZcopyStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZcopyStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZcopyStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasHdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """BLAS Level 1 API

    dot(u)  performs the dot product of vectors x and y

        result = x * y;

    dotc  performs the dot product of the conjugate of complex vector x and complex vector y

        result = conjugate (x) * y;

    - Supported precisions in rocBLAS : h,bf,s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    result
              device pointer or host pointer to store the dot product.
              return is 0.0 if n <= 0.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of y.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
    """
    _hipblasHdot__retval = hipblasStatus_t(chipblas.hipblasHdot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasHdot__retval,)


@cython.embedsignature(True)
def hipblasBfdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasBfdot__retval = hipblasStatus_t(chipblas.hipblasBfdot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasBfloat16.from_pyobj(x)._ptr,incx,
        hipblasBfloat16.from_pyobj(y)._ptr,incy,
        hipblasBfloat16.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasBfdot__retval,)


@cython.embedsignature(True)
def hipblasSdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasSdot__retval = hipblasStatus_t(chipblas.hipblasSdot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSdot__retval,)


@cython.embedsignature(True)
def hipblasDdot(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasDdot__retval = hipblasStatus_t(chipblas.hipblasDdot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDdot__retval,)


@cython.embedsignature(True)
def hipblasCdotc(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasCdotc__retval = hipblasStatus_t(chipblas.hipblasCdotc(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotc__retval,)


@cython.embedsignature(True)
def hipblasCdotu(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasCdotu__retval = hipblasStatus_t(chipblas.hipblasCdotu(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotu__retval,)


@cython.embedsignature(True)
def hipblasZdotc(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasZdotc__retval = hipblasStatus_t(chipblas.hipblasZdotc(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotc__retval,)


@cython.embedsignature(True)
def hipblasZdotu(object handle, int n, object x, int incx, object y, int incy, object result):
    """(No brief)
    """
    _hipblasZdotu__retval = hipblasStatus_t(chipblas.hipblasZdotu(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotu__retval,)


@cython.embedsignature(True)
def hipblasHdotBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """BLAS Level 1 API

    dotBatched(u) performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotcBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : h,bf,s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    result
              device array or host array of batchCount size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch
    """
    _hipblasHdotBatched__retval = hipblasStatus_t(chipblas.hipblasHdotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const unsigned short *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasHdotBatched__retval,)


@cython.embedsignature(True)
def hipblasBfdotBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasBfdotBatched__retval = hipblasStatus_t(chipblas.hipblasBfdotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasBfloat16 *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasBfloat16 *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount,
        hipblasBfloat16.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasBfdotBatched__retval,)


@cython.embedsignature(True)
def hipblasSdotBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasSdotBatched__retval = hipblasStatus_t(chipblas.hipblasSdotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSdotBatched__retval,)


@cython.embedsignature(True)
def hipblasDdotBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasDdotBatched__retval = hipblasStatus_t(chipblas.hipblasDdotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDdotBatched__retval,)


@cython.embedsignature(True)
def hipblasCdotcBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasCdotcBatched__retval = hipblasStatus_t(chipblas.hipblasCdotcBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotcBatched__retval,)


@cython.embedsignature(True)
def hipblasCdotuBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasCdotuBatched__retval = hipblasStatus_t(chipblas.hipblasCdotuBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotuBatched__retval,)


@cython.embedsignature(True)
def hipblasZdotcBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasZdotcBatched__retval = hipblasStatus_t(chipblas.hipblasZdotcBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotcBatched__retval,)


@cython.embedsignature(True)
def hipblasZdotuBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount, object result):
    """(No brief)
    """
    _hipblasZdotuBatched__retval = hipblasStatus_t(chipblas.hipblasZdotuBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotuBatched__retval,)


@cython.embedsignature(True)
def hipblasHdotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """BLAS Level 1 API

    dotStridedBatched(u)  performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotcStridedBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : h,bf,s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    result
              device array or host array of batchCount size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       x: device pointer to the first vector (x_1) in the batch.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1)
       y: device pointer to the first vector (y_1) in the batch.
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1)
       batchCount: [int]
          number of instances in the batch
    """
    _hipblasHdotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasHdotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasHdotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasBfdotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasBfdotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasBfdotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasBfloat16.from_pyobj(x)._ptr,incx,stridex,
        hipblasBfloat16.from_pyobj(y)._ptr,incy,stridey,batchCount,
        hipblasBfloat16.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasBfdotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSdotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasSdotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSdotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSdotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDdotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasDdotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDdotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDdotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCdotcStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasCdotcStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCdotcStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotcStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCdotuStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasCdotuStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCdotuStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount,
        hipblasComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasCdotuStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZdotcStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasZdotcStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZdotcStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotcStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZdotuStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount, object result):
    """(No brief)
    """
    _hipblasZdotuStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZdotuStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount,
        hipblasDoubleComplex.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasZdotuStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSnrm2(object handle, int n, object x, int incx, object result):
    """BLAS Level 1 API

    nrm2 computes the euclidean norm of a real or complex vector

              result := sqrt( x'*x ) for real vectors
              result := sqrt( x**H*x ) for complex vectors

    - Supported precisions in rocBLAS : s,d,c,z,sc,dz
    - Supported precisions in cuBLAS  : s,d,sc,dz

    @param[inout]
    result
              device pointer or host pointer to store the nrm2 product.
              return is 0.0 if n, incx<=0.
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of y.
    """
    _hipblasSnrm2__retval = hipblasStatus_t(chipblas.hipblasSnrm2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSnrm2__retval,)


@cython.embedsignature(True)
def hipblasDnrm2(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasDnrm2__retval = hipblasStatus_t(chipblas.hipblasDnrm2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDnrm2__retval,)


@cython.embedsignature(True)
def hipblasScnrm2(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasScnrm2__retval = hipblasStatus_t(chipblas.hipblasScnrm2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScnrm2__retval,)


@cython.embedsignature(True)
def hipblasDznrm2(object handle, int n, object x, int incx, object result):
    """(No brief)
    """
    _hipblasDznrm2__retval = hipblasStatus_t(chipblas.hipblasDznrm2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDznrm2__retval,)


@cython.embedsignature(True)
def hipblasSnrm2Batched(object handle, int n, object x, int incx, int batchCount, object result):
    """BLAS Level 1 API

    nrm2Batched computes the euclidean norm over a batch of real or complex vectors

              result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
              result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : s,d,c,z,sc,dz
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       batchCount: [int]
          number of instances in the batch
       result: device pointer or host pointer to array of batchCount size for nrm2 results.
          return is 0.0 for each element if n <= 0, incx<=0.
          ******************************************************************
    """
    _hipblasSnrm2Batched__retval = hipblasStatus_t(chipblas.hipblasSnrm2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSnrm2Batched__retval,)


@cython.embedsignature(True)
def hipblasDnrm2Batched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasDnrm2Batched__retval = hipblasStatus_t(chipblas.hipblasDnrm2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDnrm2Batched__retval,)


@cython.embedsignature(True)
def hipblasScnrm2Batched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasScnrm2Batched__retval = hipblasStatus_t(chipblas.hipblasScnrm2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScnrm2Batched__retval,)


@cython.embedsignature(True)
def hipblasDznrm2Batched(object handle, int n, object x, int incx, int batchCount, object result):
    """(No brief)
    """
    _hipblasDznrm2Batched__retval = hipblasStatus_t(chipblas.hipblasDznrm2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDznrm2Batched__retval,)


@cython.embedsignature(True)
def hipblasSnrm2StridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """BLAS Level 1 API

    nrm2StridedBatched computes the euclidean norm over a batch of real or complex vectors

              := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
              := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : s,d,c,z,sc,dz
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i.
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       batchCount: [int]
          number of instances in the batch
       result: device pointer or host pointer to array for storing contiguous batchCount results.
          return is 0.0 for each element if n <= 0, incx<=0.
          ******************************************************************
    """
    _hipblasSnrm2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasSnrm2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasSnrm2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDnrm2StridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasDnrm2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasDnrm2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDnrm2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasScnrm2StridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasScnrm2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasScnrm2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <float *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasScnrm2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDznrm2StridedBatched(object handle, int n, object x, int incx, long stridex, int batchCount, object result):
    """(No brief)
    """
    _hipblasDznrm2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasDznrm2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount,
        <double *>hip._util.types.DataHandle.from_pyobj(result)._ptr))    # fully specified
    return (_hipblasDznrm2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """BLAS Level 1 API

    rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    - Supported precisions in rocBLAS : s,d,c,z,sc,dz
    - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

    @param[inout]
    x       device pointer storing vector x.

    @param[inout]
    y       device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in the x and y vectors.
       incx: [int]
          specifies the increment between elements of x.
       incy: [int]
          specifies the increment between elements of y.
       c: device pointer or host pointer storing scalar cosine component of the rotation matrix.
       s: device pointer or host pointer storing scalar sine component of the rotation matrix.
          ******************************************************************
    """
    _hipblasSrot__retval = hipblasStatus_t(chipblas.hipblasSrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasSrot__retval,)


@cython.embedsignature(True)
def hipblasDrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """(No brief)
    """
    _hipblasDrot__retval = hipblasStatus_t(chipblas.hipblasDrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasDrot__retval,)


@cython.embedsignature(True)
def hipblasCrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """(No brief)
    """
    _hipblasCrot__retval = hipblasStatus_t(chipblas.hipblasCrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCrot__retval,)


@cython.embedsignature(True)
def hipblasCsrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """(No brief)
    """
    _hipblasCsrot__retval = hipblasStatus_t(chipblas.hipblasCsrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCsrot__retval,)


@cython.embedsignature(True)
def hipblasZrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """(No brief)
    """
    _hipblasZrot__retval = hipblasStatus_t(chipblas.hipblasZrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZrot__retval,)


@cython.embedsignature(True)
def hipblasZdrot(object handle, int n, object x, int incx, object y, int incy, object c, object s):
    """(No brief)
    """
    _hipblasZdrot__retval = hipblasStatus_t(chipblas.hipblasZdrot(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZdrot__retval,)


@cython.embedsignature(True)
def hipblasSrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """BLAS Level 1 API

    rotBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    - Supported precisions in rocBLAS : s,d,sc,dz
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x       device array of deivce pointers storing each vector x_i.

    @param[inout]
    y       device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i and y_i vectors.
       incx: [int]
          specifies the increment between elements of each x_i.
       incy: [int]
          specifies the increment between elements of each y_i.
       c: device pointer or host pointer to scalar cosine component of the rotation matrix.
       s: device pointer or host pointer to scalar sine component of the rotation matrix.
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
          ******************************************************************
    """
    _hipblasSrotBatched__retval = hipblasStatus_t(chipblas.hipblasSrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasSrotBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasDrotBatched__retval = hipblasStatus_t(chipblas.hipblasDrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasDrotBatched__retval,)


@cython.embedsignature(True)
def hipblasCrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasCrotBatched__retval = hipblasStatus_t(chipblas.hipblasCrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasCrotBatched__retval,)


@cython.embedsignature(True)
def hipblasCsrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasCsrotBatched__retval = hipblasStatus_t(chipblas.hipblasCsrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasCsrotBatched__retval,)


@cython.embedsignature(True)
def hipblasZrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasZrotBatched__retval = hipblasStatus_t(chipblas.hipblasZrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasZrotBatched__retval,)


@cython.embedsignature(True)
def hipblasZdrotBatched(object handle, int n, object x, int incx, object y, int incy, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasZdrotBatched__retval = hipblasStatus_t(chipblas.hipblasZdrotBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasZdrotBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """BLAS Level 1 API

    rotStridedBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    - Supported precisions in rocBLAS : s,d,sc,dz
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x       device pointer to the first vector x_1.

    @param[inout]
    y       device pointer to the first vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i and y_i vectors.
       incx: [int]
          specifies the increment between elements of each x_i.
       stridex: [hipblasStride]
          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
       incy: [int]
          specifies the increment between elements of each y_i.
       stridey: [hipblasStride]
          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
       c: device pointer or host pointer to scalar cosine component of the rotation matrix.
       s: device pointer or host pointer to scalar sine component of the rotation matrix.
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
          ******************************************************************
    """
    _hipblasSrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasSrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasDrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasDrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasCrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasCrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasCsrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        <const float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasCsrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasZrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasZrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZdrotStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasZdrotStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZdrotStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        <const double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasZdrotStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotg(object handle, object a, object b, object c, object s):
    """BLAS Level 1 API

    rotg creates the Givens rotation matrix for the vector (a b).
         Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    a       device pointer or host pointer to input vector element, overwritten with r.
    @param[inout]
    b       device pointer or host pointer to input vector element, overwritten with z.
    @param[inout]
    c       device pointer or host pointer to cosine element of Givens rotation.
    @param[inout]
    s       device pointer or host pointer sine element of Givens rotation.

     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
    """
    _hipblasSrotg__retval = hipblasStatus_t(chipblas.hipblasSrotg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(a)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(b)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasSrotg__retval,)


@cython.embedsignature(True)
def hipblasDrotg(object handle, object a, object b, object c, object s):
    """(No brief)
    """
    _hipblasDrotg__retval = hipblasStatus_t(chipblas.hipblasDrotg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(a)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(b)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasDrotg__retval,)


@cython.embedsignature(True)
def hipblasCrotg(object handle, object a, object b, object c, object s):
    """(No brief)
    """
    _hipblasCrotg__retval = hipblasStatus_t(chipblas.hipblasCrotg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        hipblasComplex.from_pyobj(a)._ptr,
        hipblasComplex.from_pyobj(b)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasCrotg__retval,)


@cython.embedsignature(True)
def hipblasZrotg(object handle, object a, object b, object c, object s):
    """(No brief)
    """
    _hipblasZrotg__retval = hipblasStatus_t(chipblas.hipblasZrotg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        hipblasDoubleComplex.from_pyobj(a)._ptr,
        hipblasDoubleComplex.from_pyobj(b)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        hipblasDoubleComplex.from_pyobj(s)._ptr))    # fully specified
    return (_hipblasZrotg__retval,)


@cython.embedsignature(True)
def hipblasSrotgBatched(object handle, object a, object b, object c, object s, int batchCount):
    """BLAS Level 1 API

    rotgBatched creates the Givens rotation matrix for the batched vectors (a_i b_i), for i = 1, ..., batchCount.
         a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    a       device array of device pointers storing each single input vector element a_i, overwritten with r_i.
    @param[inout]
    b       device array of device pointers storing each single input vector element b_i, overwritten with z_i.
    @param[inout]
    c       device array of device pointers storing each cosine element of Givens rotation for the batch.
    @param[inout]
    s       device array of device pointers storing each sine element of Givens rotation for the batch.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       batchCount: [int]
          number of batches (length of arrays a, b, c, and s).
          ******************************************************************
    """
    _hipblasSrotgBatched__retval = hipblasStatus_t(chipblas.hipblasSrotgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(a)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(b)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(c)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasSrotgBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotgBatched(object handle, object a, object b, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasDrotgBatched__retval = hipblasStatus_t(chipblas.hipblasDrotgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(a)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(b)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(c)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasDrotgBatched__retval,)


@cython.embedsignature(True)
def hipblasCrotgBatched(object handle, object a, object b, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasCrotgBatched__retval = hipblasStatus_t(chipblas.hipblasCrotgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(a)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(b)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(c)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasCrotgBatched__retval,)


@cython.embedsignature(True)
def hipblasZrotgBatched(object handle, object a, object b, object c, object s, int batchCount):
    """(No brief)
    """
    _hipblasZrotgBatched__retval = hipblasStatus_t(chipblas.hipblasZrotgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(a)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(b)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(c)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(s)._ptr,batchCount))    # fully specified
    return (_hipblasZrotgBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotgStridedBatched(object handle, object a, long stridea, object b, long strideb, object c, long stridec, object s, long strides, int batchCount):
    """BLAS Level 1 API

    rotgStridedBatched creates the Givens rotation matrix for the strided batched vectors (a_i b_i), for i = 1, ..., batchCount.
         a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
         If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    a       device strided_batched pointer or host strided_batched pointer to first single input vector element a_1, overwritten with r.

    @param[inout]
    b       device strided_batched pointer or host strided_batched pointer to first single input vector element b_1, overwritten with z.

    @param[inout]
    c       device strided_batched pointer or host strided_batched pointer to first cosine element of Givens rotations c_1.

    @param[inout]
    s       device strided_batched pointer or host strided_batched pointer to sine element of Givens rotations s_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       stridea: [hipblasStride]
          distance between elements of a in batch (distance between a_i and a_(i + 1))
       strideb: [hipblasStride]
          distance between elements of b in batch (distance between b_i and b_(i + 1))
       stridec: [hipblasStride]
          distance between elements of c in batch (distance between c_i and c_(i + 1))
       strides: [hipblasStride]
          distance between elements of s in batch (distance between s_i and s_(i + 1))
       batchCount: [int]
          number of batches (length of arrays a, b, c, and s).
          ******************************************************************
    """
    _hipblasSrotgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSrotgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(a)._ptr,stridea,
        <float *>hip._util.types.DataHandle.from_pyobj(b)._ptr,strideb,
        <float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,stridec,
        <float *>hip._util.types.DataHandle.from_pyobj(s)._ptr,strides,batchCount))    # fully specified
    return (_hipblasSrotgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotgStridedBatched(object handle, object a, long stridea, object b, long strideb, object c, long stridec, object s, long strides, int batchCount):
    """(No brief)
    """
    _hipblasDrotgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDrotgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(a)._ptr,stridea,
        <double *>hip._util.types.DataHandle.from_pyobj(b)._ptr,strideb,
        <double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,stridec,
        <double *>hip._util.types.DataHandle.from_pyobj(s)._ptr,strides,batchCount))    # fully specified
    return (_hipblasDrotgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCrotgStridedBatched(object handle, object a, long stridea, object b, long strideb, object c, long stridec, object s, long strides, int batchCount):
    """(No brief)
    """
    _hipblasCrotgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCrotgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        hipblasComplex.from_pyobj(a)._ptr,stridea,
        hipblasComplex.from_pyobj(b)._ptr,strideb,
        <float *>hip._util.types.DataHandle.from_pyobj(c)._ptr,stridec,
        hipblasComplex.from_pyobj(s)._ptr,strides,batchCount))    # fully specified
    return (_hipblasCrotgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZrotgStridedBatched(object handle, object a, long stridea, object b, long strideb, object c, long stridec, object s, long strides, int batchCount):
    """(No brief)
    """
    _hipblasZrotgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZrotgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        hipblasDoubleComplex.from_pyobj(a)._ptr,stridea,
        hipblasDoubleComplex.from_pyobj(b)._ptr,strideb,
        <double *>hip._util.types.DataHandle.from_pyobj(c)._ptr,stridec,
        hipblasDoubleComplex.from_pyobj(s)._ptr,strides,batchCount))    # fully specified
    return (_hipblasZrotgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotm(object handle, int n, object x, int incx, object y, int incy, object param):
    """BLAS Level 1 API

    rotm applies the modified Givens rotation matrix defined by param to vectors x and y.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : s,d

    @param[inout]
    x       device pointer storing vector x.

    @param[inout]
    y       device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in the x and y vectors.
       incx: [int]
          specifies the increment between elements of x.
       incy: [int]
          specifies the increment between elements of y.
       param: device vector or host vector of 5 elements defining the rotation.
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
          ******************************************************************
    """
    _hipblasSrotm__retval = hipblasStatus_t(chipblas.hipblasSrotm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const float *>hip._util.types.DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasSrotm__retval,)


@cython.embedsignature(True)
def hipblasDrotm(object handle, int n, object x, int incx, object y, int incy, object param):
    """(No brief)
    """
    _hipblasDrotm__retval = hipblasStatus_t(chipblas.hipblasDrotm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <const double *>hip._util.types.DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasDrotm__retval,)


@cython.embedsignature(True)
def hipblasSrotmBatched(object handle, int n, object x, int incx, object y, int incy, object param, int batchCount):
    """BLAS Level 1 API

    rotmBatched applies the modified Givens rotation matrix defined by param_i to batched vectors x_i and y_i, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x       device array of device pointers storing each vector x_i.

    @param[inout]
    y       device array of device pointers storing each vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in the x and y vectors.
       incx: [int]
          specifies the increment between elements of each x_i.
       incy: [int]
          specifies the increment between elements of each y_i.
       param: device array of device vectors of 5 elements defining the rotation.
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
          param may ONLY be stored on the device for the batched version of this function.
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
          ******************************************************************
    """
    _hipblasSrotmBatched__retval = hipblasStatus_t(chipblas.hipblasSrotmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(param)._ptr,batchCount))    # fully specified
    return (_hipblasSrotmBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotmBatched(object handle, int n, object x, int incx, object y, int incy, object param, int batchCount):
    """(No brief)
    """
    _hipblasDrotmBatched__retval = hipblasStatus_t(chipblas.hipblasDrotmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(param)._ptr,batchCount))    # fully specified
    return (_hipblasDrotmBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotmStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object param, long strideParam, int batchCount):
    """BLAS Level 1 API

    rotmStridedBatched applies the modified Givens rotation matrix defined by param_i to strided batched vectors x_i and y_i, for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x       device pointer pointing to first strided batched vector x_1.

    @param[inout]
    y       device pointer pointing to first strided batched vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in the x and y vectors.
       incx: [int]
          specifies the increment between elements of each x_i.
       stridex: [hipblasStride]
          specifies the increment between the beginning of x_i and x_(i + 1)
       incy: [int]
          specifies the increment between elements of each y_i.
       stridey: [hipblasStride]
          specifies the increment between the beginning of y_i and y_(i + 1)
       param: device pointer pointing to first array of 5 elements defining the rotation (param_1).
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
          param may ONLY be stored on the device for the strided_batched version of this function.
       strideParam: [hipblasStride]
          specifies the increment between the beginning of param_i and param_(i + 1)
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
          ******************************************************************
    """
    _hipblasSrotmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSrotmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <const float *>hip._util.types.DataHandle.from_pyobj(param)._ptr,strideParam,batchCount))    # fully specified
    return (_hipblasSrotmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotmStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, object param, long strideParam, int batchCount):
    """(No brief)
    """
    _hipblasDrotmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDrotmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <const double *>hip._util.types.DataHandle.from_pyobj(param)._ptr,strideParam,batchCount))    # fully specified
    return (_hipblasDrotmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotmg(object handle, object d1, object d2, object x1, object y1, object param):
    """BLAS Level 1 API

    rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
          Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : s,d

    @param[inout]
    d1      device pointer or host pointer to input scalar that is overwritten.
    @param[inout]
    d2      device pointer or host pointer to input scalar that is overwritten.
    @param[inout]
    x1      device pointer or host pointer to input scalar that is overwritten.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       y1: device pointer or host pointer to input scalar.
       param: device vector or host vector of 5 elements defining the rotation.
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
          ******************************************************************
    """
    _hipblasSrotmg__retval = hipblasStatus_t(chipblas.hipblasSrotmg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(d1)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(d2)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(x1)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(y1)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasSrotmg__retval,)


@cython.embedsignature(True)
def hipblasDrotmg(object handle, object d1, object d2, object x1, object y1, object param):
    """(No brief)
    """
    _hipblasDrotmg__retval = hipblasStatus_t(chipblas.hipblasDrotmg(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(d1)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(d2)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(x1)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(y1)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(param)._ptr))    # fully specified
    return (_hipblasDrotmg__retval,)


@cython.embedsignature(True)
def hipblasSrotmgBatched(object handle, object d1, object d2, object x1, object y1, object param, int batchCount):
    """BLAS Level 1 API

    rotmgBatched creates the modified Givens rotation matrix for the batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
          Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    d1      device batched array or host batched array of input scalars that is overwritten.
    @param[inout]
    d2      device batched array or host batched array of input scalars that is overwritten.
    @param[inout]
    x1      device batched array or host batched array of input scalars that is overwritten.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       y1: device batched array or host batched array of input scalars.
       param: device batched array or host batched array of vectors of 5 elements defining the rotation.
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
       batchCount: [int]
          the number of instances in the batch.
          ******************************************************************
    """
    _hipblasSrotmgBatched__retval = hipblasStatus_t(chipblas.hipblasSrotmgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(d1)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(d2)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(x1)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(y1)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(param)._ptr,batchCount))    # fully specified
    return (_hipblasSrotmgBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotmgBatched(object handle, object d1, object d2, object x1, object y1, object param, int batchCount):
    """(No brief)
    """
    _hipblasDrotmgBatched__retval = hipblasStatus_t(chipblas.hipblasDrotmgBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(d1)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(d2)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(x1)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(y1)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(param)._ptr,batchCount))    # fully specified
    return (_hipblasDrotmgBatched__retval,)


@cython.embedsignature(True)
def hipblasSrotmgStridedBatched(object handle, object d1, long strided1, object d2, long strided2, object x1, long stridex1, object y1, long stridey1, object param, long strideParam, int batchCount):
    """BLAS Level 1 API

    rotmgStridedBatched creates the modified Givens rotation matrix for the strided batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
          Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
          If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    d1      device strided_batched array or host strided_batched array of input scalars that is overwritten.

    @param[inout]
    d2      device strided_batched array or host strided_batched array of input scalars that is overwritten.

    @param[inout]
    x1      device strided_batched array or host strided_batched array of input scalars that is overwritten.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       strided1: [hipblasStride]
          specifies the increment between the beginning of d1_i and d1_(i+1)
       strided2: [hipblasStride]
          specifies the increment between the beginning of d2_i and d2_(i+1)
       stridex1: [hipblasStride]
          specifies the increment between the beginning of x1_i and x1_(i+1)
       y1: device strided_batched array or host strided_batched array of input scalars.
       stridey1: [hipblasStride]
          specifies the increment between the beginning of y1_i and y1_(i+1)
       param: device stridedBatched array or host stridedBatched array of vectors of 5 elements defining the rotation.
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
       strideParam: [hipblasStride]
          specifies the increment between the beginning of param_i and param_(i + 1)
       batchCount: [int]
          the number of instances in the batch.
          ******************************************************************
    """
    _hipblasSrotmgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSrotmgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(d1)._ptr,strided1,
        <float *>hip._util.types.DataHandle.from_pyobj(d2)._ptr,strided2,
        <float *>hip._util.types.DataHandle.from_pyobj(x1)._ptr,stridex1,
        <const float *>hip._util.types.DataHandle.from_pyobj(y1)._ptr,stridey1,
        <float *>hip._util.types.DataHandle.from_pyobj(param)._ptr,strideParam,batchCount))    # fully specified
    return (_hipblasSrotmgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDrotmgStridedBatched(object handle, object d1, long strided1, object d2, long strided2, object x1, long stridex1, object y1, long stridey1, object param, long strideParam, int batchCount):
    """(No brief)
    """
    _hipblasDrotmgStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDrotmgStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(d1)._ptr,strided1,
        <double *>hip._util.types.DataHandle.from_pyobj(d2)._ptr,strided2,
        <double *>hip._util.types.DataHandle.from_pyobj(x1)._ptr,stridex1,
        <const double *>hip._util.types.DataHandle.from_pyobj(y1)._ptr,stridey1,
        <double *>hip._util.types.DataHandle.from_pyobj(param)._ptr,strideParam,batchCount))    # fully specified
    return (_hipblasDrotmgStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSscal(object handle, int n, object alpha, object x, int incx):
    """BLAS Level 1 API

    scal  scales each element of vector x with scalar alpha.

        x := alpha * x

    - Supported precisions in rocBLAS : s,d,c,z,cs,zd
    - Supported precisions in cuBLAS  : s,d,c,z,cs,zd

    @param[inout]
    x         device pointer storing vector x.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       alpha: device pointer or host pointer for the scalar alpha.
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    _hipblasSscal__retval = hipblasStatus_t(chipblas.hipblasSscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasSscal__retval,)


@cython.embedsignature(True)
def hipblasDscal(object handle, int n, object alpha, object x, int incx):
    """(No brief)
    """
    _hipblasDscal__retval = hipblasStatus_t(chipblas.hipblasDscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDscal__retval,)


@cython.embedsignature(True)
def hipblasCscal(object handle, int n, object alpha, object x, int incx):
    """(No brief)
    """
    _hipblasCscal__retval = hipblasStatus_t(chipblas.hipblasCscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCscal__retval,)


@cython.embedsignature(True)
def hipblasCsscal(object handle, int n, object alpha, object x, int incx):
    """(No brief)
    """
    _hipblasCsscal__retval = hipblasStatus_t(chipblas.hipblasCsscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCsscal__retval,)


@cython.embedsignature(True)
def hipblasZscal(object handle, int n, object alpha, object x, int incx):
    """(No brief)
    """
    _hipblasZscal__retval = hipblasStatus_t(chipblas.hipblasZscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZscal__retval,)


@cython.embedsignature(True)
def hipblasZdscal(object handle, int n, object alpha, object x, int incx):
    """(No brief)
    """
    _hipblasZdscal__retval = hipblasStatus_t(chipblas.hipblasZdscal(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZdscal__retval,)


@cython.embedsignature(True)
def hipblasSscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """BLAS Level 1 API

    scalBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.

         x_i := alpha * x_i

     where (x_i) is the i-th instance of the batch.

    - Supported precisions in rocBLAS : s,d,c,z,cs,zd
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x           device array of device pointers storing each vector x_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i.
       alpha: host pointer or device pointer for the scalar alpha.
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          specifies the number of batches in x.
          ******************************************************************
    """
    _hipblasSscalBatched__retval = hipblasStatus_t(chipblas.hipblasSscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasSscalBatched__retval,)


@cython.embedsignature(True)
def hipblasDscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """(No brief)
    """
    _hipblasDscalBatched__retval = hipblasStatus_t(chipblas.hipblasDscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDscalBatched__retval,)


@cython.embedsignature(True)
def hipblasCscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """(No brief)
    """
    _hipblasCscalBatched__retval = hipblasStatus_t(chipblas.hipblasCscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCscalBatched__retval,)


@cython.embedsignature(True)
def hipblasZscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """(No brief)
    """
    _hipblasZscalBatched__retval = hipblasStatus_t(chipblas.hipblasZscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZscalBatched__retval,)


@cython.embedsignature(True)
def hipblasCsscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """(No brief)
    """
    _hipblasCsscalBatched__retval = hipblasStatus_t(chipblas.hipblasCsscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCsscalBatched__retval,)


@cython.embedsignature(True)
def hipblasZdscalBatched(object handle, int n, object alpha, object x, int incx, int batchCount):
    """(No brief)
    """
    _hipblasZdscalBatched__retval = hipblasStatus_t(chipblas.hipblasZdscalBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZdscalBatched__retval,)


@cython.embedsignature(True)
def hipblasSscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """BLAS Level 1 API

    scalStridedBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.

         x_i := alpha * x_i ,

     where (x_i) is the i-th instance of the batch.

    - Supported precisions in rocBLAS : s,d,c,z,cs,zd
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x           device pointer to the first vector (x_1) in the batch.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i.
       alpha: host pointer or device pointer for the scalar alpha.
       incx: [int]
          specifies the increment for the elements of x.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       batchCount: [int]
          specifies the number of batches in x.
          ******************************************************************
    """
    _hipblasSscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasSscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    _hipblasDscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    _hipblasCscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    _hipblasZscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    _hipblasCsscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCsscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZdscalStridedBatched(object handle, int n, object alpha, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    _hipblasZdscalStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZdscalStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZdscalStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSswap(object handle, int n, object x, int incx, object y, int incy):
    """BLAS Level 1 API

    swap  interchanges vectors x and y.

        y := x; x := y

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    x         device pointer storing vector x.

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       incx: [int]
          specifies the increment for the elements of x.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    _hipblasSswap__retval = hipblasStatus_t(chipblas.hipblasSswap(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSswap__retval,)


@cython.embedsignature(True)
def hipblasDswap(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasDswap__retval = hipblasStatus_t(chipblas.hipblasDswap(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDswap__retval,)


@cython.embedsignature(True)
def hipblasCswap(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasCswap__retval = hipblasStatus_t(chipblas.hipblasCswap(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCswap__retval,)


@cython.embedsignature(True)
def hipblasZswap(object handle, int n, object x, int incx, object y, int incy):
    """(No brief)
    """
    _hipblasZswap__retval = hipblasStatus_t(chipblas.hipblasZswap(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZswap__retval,)


@cython.embedsignature(True)
def hipblasSswapBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """BLAS Level 1 API

    swapBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount

        y_i := x_i; x_i := y_i

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device array of device pointers storing each vector x_i.

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    _hipblasSswapBatched__retval = hipblasStatus_t(chipblas.hipblasSswapBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float **>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSswapBatched__retval,)


@cython.embedsignature(True)
def hipblasDswapBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasDswapBatched__retval = hipblasStatus_t(chipblas.hipblasDswapBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double **>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDswapBatched__retval,)


@cython.embedsignature(True)
def hipblasCswapBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasCswapBatched__retval = hipblasStatus_t(chipblas.hipblasCswapBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex **>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex **>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCswapBatched__retval,)


@cython.embedsignature(True)
def hipblasZswapBatched(object handle, int n, object x, int incx, object y, int incy, int batchCount):
    """(No brief)
    """
    _hipblasZswapBatched__retval = hipblasStatus_t(chipblas.hipblasZswapBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex **>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex **>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZswapBatched__retval,)


@cython.embedsignature(True)
def hipblasSswapStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """BLAS Level 1 API

    swapStridedBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount

        y_i := x_i; x_i := y_i

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device pointer to the first vector x_1.

    @param[inout]
    y         device pointer to the first vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       incx: [int]
          specifies the increment for the elements of x.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       incy: [int]
          specifies the increment for the elements of y.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_y is of appropriate size, for a typical
          case this means stride_y >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    _hipblasSswapStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSswapStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSswapStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDswapStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasDswapStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDswapStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDswapStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCswapStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasCswapStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCswapStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCswapStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZswapStridedBatched(object handle, int n, object x, int incx, long stridex, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    _hipblasZswapStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZswapStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZswapStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    gbmv performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       trans: [hipblasOperation_t]
          indicates whether matrix A is tranposed (conjugated) or not
       m: [int]
          number of rows of matrix A
       n: [int]
          number of columns of matrix A
       kl: [int]
          number of sub-diagonals of A
       ku: [int]
          number of super-diagonals of A
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer storing banded matrix A.
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
       lda: [int]
          specifies the leading dimension of A. Must be >= (kl + ku + 1)
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgbmv__retval = hipblasStatus_t(chipblas.hipblasSgbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSgbmv__retval,)


@cython.embedsignature(True)
def hipblasDgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgbmv__retval = hipblasStatus_t(chipblas.hipblasDgbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDgbmv__retval,)


@cython.embedsignature(True)
def hipblasCgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgbmv__retval = hipblasStatus_t(chipblas.hipblasCgbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCgbmv__retval,)


@cython.embedsignature(True)
def hipblasZgbmv(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgbmv__retval = hipblasStatus_t(chipblas.hipblasZgbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZgbmv__retval,)


@cython.embedsignature(True)
def hipblasSgbmvBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    gbmvBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       trans: [hipblasOperation_t]
          indicates whether matrix A is tranposed (conjugated) or not
       m: [int]
          number of rows of each matrix A_i
       n: [int]
          number of columns of each matrix A_i
       kl: [int]
          number of sub-diagonals of each A_i
       ku: [int]
          number of super-diagonals of each A_i
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array of device pointers storing each banded matrix A_i.
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
       lda: [int]
          specifies the leading dimension of each A_i. Must be >= (kl + ku + 1)
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          specifies the number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgbmvBatched__retval = hipblasStatus_t(chipblas.hipblasSgbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSgbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDgbmvBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgbmvBatched__retval = hipblasStatus_t(chipblas.hipblasDgbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDgbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasCgbmvBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgbmvBatched__retval = hipblasStatus_t(chipblas.hipblasCgbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCgbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZgbmvBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgbmvBatched__retval = hipblasStatus_t(chipblas.hipblasZgbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZgbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasSgbmvStridedBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    gbmvStridedBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device pointer to first vector (y_1).

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       trans: [hipblasOperation_t]
          indicates whether matrix A is tranposed (conjugated) or not
       m: [int]
          number of rows of matrix A
       n: [int]
          number of columns of matrix A
       kl: [int]
          number of sub-diagonals of A
       ku: [int]
          number of super-diagonals of A
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer to first banded matrix (A_1).
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
       lda: [int]
          specifies the leading dimension of A. Must be >= (kl + ku + 1)
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: device pointer to first vector (x_1).
       incx: [int]
          specifies the increment for the elements of x.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1)
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (x_i+1)
       batchCount: [int]
          specifies the number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSgbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgbmvStridedBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDgbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgbmvStridedBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCgbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgbmvStridedBatched(object handle, object trans, int m, int n, int kl, int ku, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,kl,ku,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZgbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    gemv performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       trans: [hipblasOperation_t]
          indicates whether matrix A is tranposed (conjugated) or not
       m: [int]
          number of rows of matrix A
       n: [int]
          number of columns of matrix A
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer storing matrix A.
       lda: [int]
          specifies the leading dimension of A.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemv__retval = hipblasStatus_t(chipblas.hipblasSgemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSgemv__retval,)


@cython.embedsignature(True)
def hipblasDgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemv__retval = hipblasStatus_t(chipblas.hipblasDgemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDgemv__retval,)


@cython.embedsignature(True)
def hipblasCgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemv__retval = hipblasStatus_t(chipblas.hipblasCgemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCgemv__retval,)


@cython.embedsignature(True)
def hipblasZgemv(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemv__retval = hipblasStatus_t(chipblas.hipblasZgemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZgemv__retval,)


@cython.embedsignature(True)
def hipblasSgemvBatched(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    gemvBatched performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y           device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       trans: [hipblasOperation_t]
          indicates whether matrices A_i are tranposed (conjugated) or not
       m: [int]
          number of rows of each matrix A_i
       n: [int]
          number of columns of each matrix A_i
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array of device pointers storing each matrix A_i.
       lda: [int]
          specifies the leading dimension of each matrix A_i.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each vector x_i.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of each vector y_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemvBatched__retval = hipblasStatus_t(chipblas.hipblasSgemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSgemvBatched__retval,)


@cython.embedsignature(True)
def hipblasDgemvBatched(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemvBatched__retval = hipblasStatus_t(chipblas.hipblasDgemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDgemvBatched__retval,)


@cython.embedsignature(True)
def hipblasCgemvBatched(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemvBatched__retval = hipblasStatus_t(chipblas.hipblasCgemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCgemvBatched__retval,)


@cython.embedsignature(True)
def hipblasZgemvBatched(object handle, object trans, int m, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemvBatched__retval = hipblasStatus_t(chipblas.hipblasZgemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZgemvBatched__retval,)


@cython.embedsignature(True)
def hipblasSgemvStridedBatched(object handle, object transA, int m, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    gemvStridedBatched performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y           device pointer to the first vector (y_1) in the batch.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          indicates whether matrices A_i are tranposed (conjugated) or not
       m: [int]
          number of rows of matrices A_i
       n: [int]
          number of columns of matrices A_i
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer to the first matrix (A_1) in the batch.
       lda: [int]
          specifies the leading dimension of matrices A_i.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: device pointer to the first vector (x_1) in the batch.
       incx: [int]
          specifies the increment for the elements of vectors x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size. When trans equals HIPBLAS_OP_N
          this typically means stridex >= n * incx, otherwise stridex >= m * incx.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of vectors y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size. When trans equals HIPBLAS_OP_N
          this typically means stridey >= m * incy, otherwise stridey >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSgemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgemvStridedBatched(object handle, object transA, int m, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDgemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgemvStridedBatched(object handle, object transA, int m, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCgemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgemvStridedBatched(object handle, object transA, int m, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZgemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSger(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """BLAS Level 2 API

    ger,geru,gerc performs the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for gerc

    where alpha is a scalar, x and y are vectors, and A is an
    m by n matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    AP         device pointer storing matrix A.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       m: [int]
          the number of rows of the matrix A.
       n: [int]
          the number of columns of the matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
       lda: [int]
          specifies the leading dimension of A.
          ******************************************************************
    """
    _hipblasSger__retval = hipblasStatus_t(chipblas.hipblasSger(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSger__retval,)


@cython.embedsignature(True)
def hipblasDger(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    _hipblasDger__retval = hipblasStatus_t(chipblas.hipblasDger(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDger__retval,)


@cython.embedsignature(True)
def hipblasCgeru(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    _hipblasCgeru__retval = hipblasStatus_t(chipblas.hipblasCgeru(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCgeru__retval,)


@cython.embedsignature(True)
def hipblasCgerc(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    _hipblasCgerc__retval = hipblasStatus_t(chipblas.hipblasCgerc(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCgerc__retval,)


@cython.embedsignature(True)
def hipblasZgeru(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    _hipblasZgeru__retval = hipblasStatus_t(chipblas.hipblasZgeru(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZgeru__retval,)


@cython.embedsignature(True)
def hipblasZgerc(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    _hipblasZgerc__retval = hipblasStatus_t(chipblas.hipblasZgerc(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZgerc__retval,)


@cython.embedsignature(True)
def hipblasSgerBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """BLAS Level 2 API

    gerBatched,geruBatched,gercBatched performs a batch of the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for gerc

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing each matrix A_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       m: [int]
          the number of rows of each matrix A_i.
       n: [int]
          the number of columns of eaceh matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each vector x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each vector y_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasSgerBatched__retval = hipblasStatus_t(chipblas.hipblasSgerBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasSgerBatched__retval,)


@cython.embedsignature(True)
def hipblasDgerBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    _hipblasDgerBatched__retval = hipblasStatus_t(chipblas.hipblasDgerBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasDgerBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeruBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    _hipblasCgeruBatched__retval = hipblasStatus_t(chipblas.hipblasCgeruBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCgeruBatched__retval,)


@cython.embedsignature(True)
def hipblasCgercBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    _hipblasCgercBatched__retval = hipblasStatus_t(chipblas.hipblasCgercBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCgercBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeruBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    _hipblasZgeruBatched__retval = hipblasStatus_t(chipblas.hipblasZgeruBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZgeruBatched__retval,)


@cython.embedsignature(True)
def hipblasZgercBatched(object handle, int m, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    _hipblasZgercBatched__retval = hipblasStatus_t(chipblas.hipblasZgercBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZgercBatched__retval,)


@cython.embedsignature(True)
def hipblasSgerStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """BLAS Level 2 API

    gerStridedBatched,geruStridedBatched,gercStridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**T, OR
        A_i := A_i + alpha*x_i*y_i**H  for gerc

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device pointer to the first vector (y_1) in the batch.

    @param[inout]
    AP        device pointer to the first matrix (A_1) in the batch.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       m: [int]
          the number of rows of each matrix A_i.
       n: [int]
          the number of columns of each matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer to the first vector (x_1) in the batch.
       incx: [int]
          specifies the increments for the elements of each vector x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size, for a typical
          case this means stridex >= m * incx.
       incy: [int]
          specifies the increment for the elements of each vector y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size, for a typical
          case this means stridey >= n * incy.
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    _hipblasSgerStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgerStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasSgerStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgerStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    _hipblasDgerStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgerStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasDgerStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeruStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    _hipblasCgeruStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgeruStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCgeruStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgercStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    _hipblasCgercStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgercStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCgercStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeruStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    _hipblasZgeruStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgeruStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZgeruStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgercStridedBatched(object handle, int m, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    _hipblasZgercStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgercStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZgercStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    hbmv performs the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian band matrix, with k super-diagonals.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
       n: [int]
          the order of the matrix A.
       k: [int]
          the number of super-diagonals of the matrix A. Must be >= 0.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer storing matrix A. Of dimension (lda, n).
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
       lda: [int]
          specifies the leading dimension of A. must be >= k + 1
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChbmv__retval = hipblasStatus_t(chipblas.hipblasChbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChbmv__retval,)


@cython.embedsignature(True)
def hipblasZhbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhbmv__retval = hipblasStatus_t(chipblas.hipblasZhbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhbmv__retval,)


@cython.embedsignature(True)
def hipblasChbmvBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    hbmvBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
       n: [int]
          the order of each matrix A_i.
       k: [int]
          the number of super-diagonals of each matrix A_i. Must be >= 0.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, n).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The leading (k + 1) by n part of each A_i must contain the upper
          triangular band part of the Hermitian matrix, with the leading
          diagonal in row (k + 1), the first super-diagonal on the RHS
          of row k, etc.
          The top left k by x triangle of each A_i will not be referenced.
          Ex (upper, lda = n = 4, k = 1):
          A                             Represented matrix
          (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
          (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
          (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
          (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The leading (k + 1) by n part of each A_i must contain the lower
          triangular band part of the Hermitian matrix, with the leading
          diagonal in row (1), the first sub-diagonal on the LHS of
          row 2, etc.
          The bottom right k by k triangle of each A_i will not be referenced.
          Ex (lower, lda = 2, n = 4, k = 1):
          A                               Represented matrix
          (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
          (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
          (0, 0) (6, 8) (3, 0) (7,-7)
          (0, 0) (0, 0) (7, 7) (4, 0)
          As a Hermitian matrix, the imaginary part of the main diagonal
          of each A_i will not be referenced and is assumed to be == 0.
       lda: [int]
          specifies the leading dimension of each A_i. must be >= max(1, n)
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChbmvBatched__retval = hipblasStatus_t(chipblas.hipblasChbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasChbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZhbmvBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhbmvBatched__retval = hipblasStatus_t(chipblas.hipblasZhbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZhbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasChbmvStridedBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    hbmvStridedBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array pointing to the first vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
       n: [int]
          the order of each matrix A_i.
       k: [int]
          the number of super-diagonals of each matrix A_i. Must be >= 0.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array pointing to the first matrix A_1. Each A_i is of dimension (lda, n).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The leading (k + 1) by n part of each A_i must contain the upper
          triangular band part of the Hermitian matrix, with the leading
          diagonal in row (k + 1), the first super-diagonal on the RHS
          of row k, etc.
          The top left k by x triangle of each A_i will not be referenced.
          Ex (upper, lda = n = 4, k = 1):
          A                             Represented matrix
          (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
          (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
          (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
          (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The leading (k + 1) by n part of each A_i must contain the lower
          triangular band part of the Hermitian matrix, with the leading
          diagonal in row (1), the first sub-diagonal on the LHS of
          row 2, etc.
          The bottom right k by k triangle of each A_i will not be referenced.
          Ex (lower, lda = 2, n = 4, k = 1):
          A                               Represented matrix
          (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
          (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
          (0, 0) (6, 8) (3, 0) (7,-7)
          (0, 0) (0, 0) (7, 7) (4, 0)
          As a Hermitian matrix, the imaginary part of the main diagonal
          of each A_i will not be referenced and is assumed to be == 0.
       lda: [int]
          specifies the leading dimension of each A_i. must be >= max(1, n)
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: device array pointing to the first vector y_1.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1)
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasChbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasChbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhbmvStridedBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZhbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChemv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    hemv performs one of the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian matrix.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
       n: [int]
          the order of the matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer storing matrix A. Of dimension (lda, n).
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
       lda: [int]
          specifies the leading dimension of A. must be >= max(1, n)
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemv__retval = hipblasStatus_t(chipblas.hipblasChemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChemv__retval,)


@cython.embedsignature(True)
def hipblasZhemv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemv__retval = hipblasStatus_t(chipblas.hipblasZhemv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhemv__retval,)


@cython.embedsignature(True)
def hipblasChemvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    hemvBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
       n: [int]
          the order of each matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array of device pointers storing each matrix A_i of dimension (lda, n).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The upper triangular part of each A_i must contain
          the upper triangular part of a Hermitian matrix. The lower
          triangular part of each A_i will not be referenced.
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The lower triangular part of each A_i must contain
          the lower triangular part of a Hermitian matrix. The upper
          triangular part of each A_i will not be referenced.
          As a Hermitian matrix, the imaginary part of the main diagonal
          of each A_i will not be referenced and is assumed to be == 0.
       lda: [int]
          specifies the leading dimension of each A_i. must be >= max(1, n)
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemvBatched__retval = hipblasStatus_t(chipblas.hipblasChemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasChemvBatched__retval,)


@cython.embedsignature(True)
def hipblasZhemvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemvBatched__retval = hipblasStatus_t(chipblas.hipblasZhemvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZhemvBatched__retval,)


@cython.embedsignature(True)
def hipblasChemvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    hemvStridedBatched performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
       n: [int]
          the order of each matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device array of device pointers storing each matrix A_i of dimension (lda, n).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The upper triangular part of each A_i must contain
          the upper triangular part of a Hermitian matrix. The lower
          triangular part of each A_i will not be referenced.
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The lower triangular part of each A_i must contain
          the lower triangular part of a Hermitian matrix. The upper
          triangular part of each A_i will not be referenced.
          As a Hermitian matrix, the imaginary part of the main diagonal
          of each A_i will not be referenced and is assumed to be == 0.
       lda: [int]
          specifies the leading dimension of each A_i. must be >= max(1, n)
       strideA: [hipblasStride]
          stride from the start of one (A_i) to the next (A_i+1)
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasChemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasChemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhemvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhemvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZhemvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCher(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """BLAS Level 2 API

    her performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n Hermitian matrix.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       lda: [int]
          specifies the leading dimension of A. Must be at least max(1, n).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCher__retval = hipblasStatus_t(chipblas.hipblasCher(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCher__retval,)


@cython.embedsignature(True)
def hipblasZher(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZher__retval = hipblasStatus_t(chipblas.hipblasZher(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZher__retval,)


@cython.embedsignature(True)
def hipblasCherBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """BLAS Level 2 API

    herBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP       device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular portion
                of each A_i will not be touched.
            if uplo == HIPBLAS_FILL_MODE_LOWER:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular portion
                of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       lda: [int]
          specifies the leading dimension of each A_i. Must be at least max(1, n).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCherBatched__retval = hipblasStatus_t(chipblas.hipblasCherBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCherBatched__retval,)


@cython.embedsignature(True)
def hipblasZherBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZherBatched__retval = hipblasStatus_t(chipblas.hipblasZherBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZherBatched__retval,)


@cython.embedsignature(True)
def hipblasCherStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """BLAS Level 2 API

    herStridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n Hermitian matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == HIPBLAS_FILL_MODE_LOWER:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          stride from the start of one (A_i) and the next (A_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCherStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCherStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCherStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZherStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZherStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZherStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZherStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCher2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """BLAS Level 2 API

    her2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n Hermitian matrix.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
       lda: [int]
          specifies the leading dimension of A. Must be at least max(lda, 1).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCher2__retval = hipblasStatus_t(chipblas.hipblasCher2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCher2__retval,)


@cython.embedsignature(True)
def hipblasZher2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZher2__retval = hipblasStatus_t(chipblas.hipblasZher2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZher2__retval,)


@cython.embedsignature(True)
def hipblasCher2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """BLAS Level 2 API

    her2Batched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n Hermitian matrix for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i of size (lda, n).
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == HIPBLAS_FILL_MODE_LOWER:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of x.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       lda: [int]
          specifies the leading dimension of each A_i. Must be at least max(lda, 1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCher2Batched__retval = hipblasStatus_t(chipblas.hipblasCher2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCher2Batched__retval,)


@cython.embedsignature(True)
def hipblasZher2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZher2Batched__retval = hipblasStatus_t(chipblas.hipblasZher2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZher2Batched__retval,)


@cython.embedsignature(True)
def hipblasCher2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """BLAS Level 2 API

    her2StridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n Hermitian matrix for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device pointer pointing to the first matrix (A_1). Stores the specified triangular portion of
              each Hermitian matrix A_i.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == HIPBLAS_FILL_MODE_LOWER:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          specifies the stride between the beginning of one vector (x_i) and the next (x_i+1).
       y: device pointer pointing to the first vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          specifies the stride between the beginning of one vector (y_i) and the next (y_i+1).
       lda: [int]
          specifies the leading dimension of each A_i. Must be at least max(lda, 1).
       strideA: [hipblasStride]
          specifies the stride between the beginning of one matrix (A_i) and the next (A_i+1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCher2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasCher2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCher2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZher2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZher2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasZher2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZher2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChpmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    hpmv performs the matrix-vector operation

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian matrix, supplied in packed form (see description below).

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
       n: [int]
          the order of the matrix A, must be >= 0.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer storing the packed version of the specified triangular portion of
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
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpmv__retval = hipblasStatus_t(chipblas.hipblasChpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasChpmv__retval,)


@cython.embedsignature(True)
def hipblasZhpmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpmv__retval = hipblasStatus_t(chipblas.hipblasZhpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZhpmv__retval,)


@cython.embedsignature(True)
def hipblasChpmvBatched(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    hpmvBatched performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
       n: [int]
          the order of each matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer of device pointers storing the packed version of the specified triangular
          portion of each Hermitian matrix A_i. Each A_i is of at least size ((n * (n + 1)) / 2).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The upper triangular portion of each Hermitian matrix A_i is supplied.
          The matrix is compacted so that each AP_i contains the triangular portion column-by-column
          so that:
          AP(0) = A(0,0)
          AP(1) = A(0,1)
          AP(2) = A(1,1), etc.
          Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
          (1, 0) (2, 1) (3, 2)
          (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
          (3,-2) (5, 1) (6, 0)
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The lower triangular portion of each Hermitian matrix A_i is supplied.
          The matrix is compacted so that each AP_i contains the triangular portion column-by-column
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
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpmvBatched__retval = hipblasStatus_t(chipblas.hipblasChpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasChpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZhpmvBatched(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpmvBatched__retval = hipblasStatus_t(chipblas.hipblasZhpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZhpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasChpmvStridedBatched(object handle, object uplo, int n, object alpha, object AP, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    hpmvStridedBatched performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batchCount].

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    y         device array pointing to the beginning of the first vector (y_1).

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
       n: [int]
          the order of each matrix A_i.
       alpha: device pointer or host pointer to scalar alpha.
       AP: device pointer pointing to the beginning of the first matrix (AP_1). Stores the packed
          version of the specified triangular portion of each Hermitian matrix AP_i of size ((n * (n + 1)) / 2).
          if uplo == HIPBLAS_FILL_MODE_UPPER:
          The upper triangular portion of each Hermitian matrix A_i is supplied.
          The matrix is compacted so that each AP_i contains the triangular portion column-by-column
          so that:
          AP(0) = A(0,0)
          AP(1) = A(0,1)
          AP(2) = A(1,1), etc.
          Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
          (1, 0) (2, 1) (3, 2)
          (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
          (3,-2) (5, 1) (6, 0)
          if uplo == HIPBLAS_FILL_MODE_LOWER:
          The lower triangular portion of each Hermitian matrix A_i is supplied.
          The matrix is compacted so that each AP_i contains the triangular portion column-by-column
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
       strideA: [hipblasStride]
          stride from the start of one matrix (AP_i) and the next one (AP_i+1).
       x: device array pointing to the beginning of the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       beta: device pointer or host pointer to scalar beta.
       incy: [int]
          specifies the increment for the elements of y.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasChpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasChpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhpmvStridedBatched(object handle, object uplo, int n, object alpha, object AP, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZhpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChpr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """BLAS Level 2 API

    hpr performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n Hermitian matrix, supplied in packed form.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

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
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpr__retval = hipblasStatus_t(chipblas.hipblasChpr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasChpr__retval,)


@cython.embedsignature(True)
def hipblasZhpr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpr__retval = hipblasStatus_t(chipblas.hipblasZhpr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZhpr__retval,)


@cython.embedsignature(True)
def hipblasChprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """BLAS Level 2 API

    hprBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
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
                The lower triangular portion of each Hermitian matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChprBatched__retval = hipblasStatus_t(chipblas.hipblasChprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasChprBatched__retval,)


@cython.embedsignature(True)
def hipblasZhprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhprBatched__retval = hipblasStatus_t(chipblas.hipblasZhprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasZhprBatched__retval,)


@cython.embedsignature(True)
def hipblasChprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """BLAS Level 2 API

    hprStridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
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
                The lower triangular portion of each Hermitian matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       strideA: [hipblasStride]
          stride from the start of one (A_i) and the next (A_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasChprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasChprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasZhprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChpr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """BLAS Level 2 API

    hpr2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n Hermitian matrix, supplied in packed form.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

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
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpr2__retval = hipblasStatus_t(chipblas.hipblasChpr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasChpr2__retval,)


@cython.embedsignature(True)
def hipblasZhpr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpr2__retval = hipblasStatus_t(chipblas.hipblasZhpr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZhpr2__retval,)


@cython.embedsignature(True)
def hipblasChpr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int batchCount):
    """BLAS Level 2 API

    hpr2Batched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
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
                The lower triangular portion of each Hermitian matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpr2Batched__retval = hipblasStatus_t(chipblas.hipblasChpr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasChpr2Batched__retval,)


@cython.embedsignature(True)
def hipblasZhpr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpr2Batched__retval = hipblasStatus_t(chipblas.hipblasZhpr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasZhpr2Batched__retval,)


@cython.embedsignature(True)
def hipblasChpr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, long strideA, int batchCount):
    """BLAS Level 2 API

    hpr2StridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
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
                The lower triangular portion of each Hermitian matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       y: device pointer pointing to the first vector (y_1).
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
       strideA: [hipblasStride]
          stride from the start of one (A_i) and the next (A_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChpr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasChpr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasChpr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhpr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhpr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhpr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasZhpr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    sbmv performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : s,d

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
       k: [int]
          specifies the number of sub- and super-diagonals
       alpha: specifies the scalar alpha
       AP: pointer storing matrix A on the GPU
       lda: [int]
          specifies the leading dimension of matrix A
       x: pointer storing vector x on the GPU
       incx: [int]
          specifies the increment for the elements of x
       beta: specifies the scalar beta
       y: pointer storing vector y on the GPU
       incy: [int]
          specifies the increment for the elements of y
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsbmv__retval = hipblasStatus_t(chipblas.hipblasSsbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSsbmv__retval,)


@cython.embedsignature(True)
def hipblasDsbmv(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsbmv__retval = hipblasStatus_t(chipblas.hipblasDsbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDsbmv__retval,)


@cython.embedsignature(True)
def hipblasSsbmvBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    sbmvBatched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batchCount.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       k: [int]
          specifies the number of sub- and super-diagonals
       alpha: device pointer or host pointer to scalar alpha
       AP: device array of device pointers storing each matrix A_i
       lda: [int]
          specifies the leading dimension of each matrix A_i
       x: device array of device pointers storing each vector x_i
       incx: [int]
          specifies the increment for the elements of each vector x_i
       beta: device pointer or host pointer to scalar beta
       y: device array of device pointers storing each vector y_i
       incy: [int]
          specifies the increment for the elements of each vector y_i
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsbmvBatched__retval = hipblasStatus_t(chipblas.hipblasSsbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSsbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDsbmvBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsbmvBatched__retval = hipblasStatus_t(chipblas.hipblasDsbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDsbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasSsbmvStridedBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    sbmvStridedBatched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batchCount.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       k: [int]
          specifies the number of sub- and super-diagonals
       alpha: device pointer or host pointer to scalar alpha
       AP: Device pointer to the first matrix A_1 on the GPU
       lda: [int]
          specifies the leading dimension of each matrix A_i
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: Device pointer to the first vector x_1 on the GPU
       incx: [int]
          specifies the increment for the elements of each vector x_i
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size.
          This typically means stridex >= n * incx. stridex should be non zero.
       beta: device pointer or host pointer to scalar beta
       y: Device pointer to the first vector y_1 on the GPU
       incy: [int]
          specifies the increment for the elements of each vector y_i
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size.
          This typically means stridey >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSsbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsbmvStridedBatched(object handle, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDsbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSspmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    spmv performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : s,d

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
       alpha: specifies the scalar alpha
       AP: pointer storing matrix A on the GPU
       x: pointer storing vector x on the GPU
       incx: [int]
          specifies the increment for the elements of x
       beta: specifies the scalar beta
       y: pointer storing vector y on the GPU
       incy: [int]
          specifies the increment for the elements of y
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspmv__retval = hipblasStatus_t(chipblas.hipblasSspmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSspmv__retval,)


@cython.embedsignature(True)
def hipblasDspmv(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspmv__retval = hipblasStatus_t(chipblas.hipblasDspmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDspmv__retval,)


@cython.embedsignature(True)
def hipblasSspmvBatched(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    spmvBatched performs the matrix-vector operation:

        y_i := alpha*AP_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batchCount.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       alpha: device pointer or host pointer to scalar alpha
       AP: device array of device pointers storing each matrix A_i
       x: device array of device pointers storing each vector x_i
       incx: [int]
          specifies the increment for the elements of each vector x_i
       beta: device pointer or host pointer to scalar beta
       y: device array of device pointers storing each vector y_i
       incy: [int]
          specifies the increment for the elements of each vector y_i
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspmvBatched__retval = hipblasStatus_t(chipblas.hipblasSspmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSspmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDspmvBatched(object handle, object uplo, int n, object alpha, object AP, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspmvBatched__retval = hipblasStatus_t(chipblas.hipblasDspmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDspmvBatched__retval,)


@cython.embedsignature(True)
def hipblasSspmvStridedBatched(object handle, object uplo, int n, object alpha, object AP, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    spmvStridedBatched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batchCount.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       alpha: device pointer or host pointer to scalar alpha
       AP: Device pointer to the first matrix A_1 on the GPU
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: Device pointer to the first vector x_1 on the GPU
       incx: [int]
          specifies the increment for the elements of each vector x_i
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size.
          This typically means stridex >= n * incx. stridex should be non zero.
       beta: device pointer or host pointer to scalar beta
       y: Device pointer to the first vector y_1 on the GPU
       incy: [int]
          specifies the increment for the elements of each vector y_i
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size.
          This typically means stridey >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSspmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSspmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDspmvStridedBatched(object handle, object uplo, int n, object alpha, object AP, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDspmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDspmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """BLAS Level 2 API

    spr performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix, supplied in packed form.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

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
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspr__retval = hipblasStatus_t(chipblas.hipblasSspr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasSspr__retval,)


@cython.embedsignature(True)
def hipblasDspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspr__retval = hipblasStatus_t(chipblas.hipblasDspr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasDspr__retval,)


@cython.embedsignature(True)
def hipblasCspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCspr__retval = hipblasStatus_t(chipblas.hipblasCspr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasCspr__retval,)


@cython.embedsignature(True)
def hipblasZspr(object handle, object uplo, int n, object alpha, object x, int incx, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZspr__retval = hipblasStatus_t(chipblas.hipblasZspr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasZspr__retval,)


@cython.embedsignature(True)
def hipblasSsprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """BLAS Level 2 API

    sprBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each symmetric matrix A_i is supplied.
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
                The lower triangular portion of each symmetric matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsprBatched__retval = hipblasStatus_t(chipblas.hipblasSsprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasSsprBatched__retval,)


@cython.embedsignature(True)
def hipblasDsprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsprBatched__retval = hipblasStatus_t(chipblas.hipblasDsprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasDsprBatched__retval,)


@cython.embedsignature(True)
def hipblasCsprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsprBatched__retval = hipblasStatus_t(chipblas.hipblasCsprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasCsprBatched__retval,)


@cython.embedsignature(True)
def hipblasZsprBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsprBatched__retval = hipblasStatus_t(chipblas.hipblasZsprBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasZsprBatched__retval,)


@cython.embedsignature(True)
def hipblasSsprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """BLAS Level 2 API

    sprStridedBatched performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each symmetric matrix A_i is supplied.
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
                The lower triangular portion of each symmetric matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       strideA: [hipblasStride]
          stride from the start of one (A_i) and the next (A_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasSsprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasDsprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasCsprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsprStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsprStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsprStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasZsprStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSspr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """BLAS Level 2 API

    spr2 performs the matrix-vector operation

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix, supplied in packed form.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : s,d

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
     ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
       n: [int]
          the number of rows and columns of matrix A, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspr2__retval = hipblasStatus_t(chipblas.hipblasSspr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasSspr2__retval,)


@cython.embedsignature(True)
def hipblasDspr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspr2__retval = hipblasStatus_t(chipblas.hipblasDspr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr))    # fully specified
    return (_hipblasDspr2__retval,)


@cython.embedsignature(True)
def hipblasSspr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int batchCount):
    """BLAS Level 2 API

    spr2Batched performs the matrix-vector operation

        A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T

    where alpha is a scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each symmetric matrix A_i is supplied.
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
                The lower triangular portion of each symmetric matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspr2Batched__retval = hipblasStatus_t(chipblas.hipblasSspr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasSspr2Batched__retval,)


@cython.embedsignature(True)
def hipblasDspr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspr2Batched__retval = hipblasStatus_t(chipblas.hipblasDspr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,batchCount))    # fully specified
    return (_hipblasDspr2Batched__retval,)


@cython.embedsignature(True)
def hipblasSspr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, long strideA, int batchCount):
    """BLAS Level 2 API

    spr2StridedBatched performs the matrix-vector operation

        A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T

    where alpha is a scalar, x_i amd y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == HIPBLAS_FILL_MODE_UPPER:
                The upper triangular portion of each symmetric matrix A_i is supplied.
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
                The lower triangular portion of each symmetric matrix A_i is supplied.
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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
          HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
       n: [int]
          the number of rows and columns of each matrix A_i, must be at least 0.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer pointing to the first vector (x_1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
       y: device pointer pointing to the first vector (y_1).
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
       strideA: [hipblasStride]
          stride from the start of one (A_i) and the next (A_i+1)
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSspr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasSspr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasSspr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDspr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDspr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasDspr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,batchCount))    # fully specified
    return (_hipblasDspr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """BLAS Level 2 API

    symv performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
       alpha: specifies the scalar alpha
       AP: pointer storing matrix A on the GPU
       lda: [int]
          specifies the leading dimension of A
       x: pointer storing vector x on the GPU
       incx: [int]
          specifies the increment for the elements of x
       beta: specifies the scalar beta
       y: pointer storing vector y on the GPU
       incy: [int]
          specifies the increment for the elements of y
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymv__retval = hipblasStatus_t(chipblas.hipblasSsymv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasSsymv__retval,)


@cython.embedsignature(True)
def hipblasDsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymv__retval = hipblasStatus_t(chipblas.hipblasDsymv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasDsymv__retval,)


@cython.embedsignature(True)
def hipblasCsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymv__retval = hipblasStatus_t(chipblas.hipblasCsymv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasCsymv__retval,)


@cython.embedsignature(True)
def hipblasZsymv(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymv__retval = hipblasStatus_t(chipblas.hipblasZsymv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy))    # fully specified
    return (_hipblasZsymv__retval,)


@cython.embedsignature(True)
def hipblasSsymvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """BLAS Level 2 API

    symvBatched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batchCount.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       alpha: device pointer or host pointer to scalar alpha
       AP: device array of device pointers storing each matrix A_i
       lda: [int]
          specifies the leading dimension of each matrix A_i
       x: device array of device pointers storing each vector x_i
       incx: [int]
          specifies the increment for the elements of each vector x_i
       beta: device pointer or host pointer to scalar beta
       y: device array of device pointers storing each vector y_i
       incy: [int]
          specifies the increment for the elements of each vector y_i
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymvBatched__retval = hipblasStatus_t(chipblas.hipblasSsymvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasSsymvBatched__retval,)


@cython.embedsignature(True)
def hipblasDsymvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymvBatched__retval = hipblasStatus_t(chipblas.hipblasDsymvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double **>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasDsymvBatched__retval,)


@cython.embedsignature(True)
def hipblasCsymvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymvBatched__retval = hipblasStatus_t(chipblas.hipblasCsymvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex **>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasCsymvBatched__retval,)


@cython.embedsignature(True)
def hipblasZsymvBatched(object handle, object uplo, int n, object alpha, object AP, int lda, object x, int incx, object beta, object y, int incy, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymvBatched__retval = hipblasStatus_t(chipblas.hipblasZsymvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex **>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,batchCount))    # fully specified
    return (_hipblasZsymvBatched__retval,)


@cython.embedsignature(True)
def hipblasSsymvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """BLAS Level 2 API

    symvStridedBatched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batchCount.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          number of rows and columns of each matrix A_i
       alpha: device pointer or host pointer to scalar alpha
       AP: Device pointer to the first matrix A_1 on the GPU
       lda: [int]
          specifies the leading dimension of each matrix A_i
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: Device pointer to the first vector x_1 on the GPU
       incx: [int]
          specifies the increment for the elements of each vector x_i
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size.
          This typically means stridex >= n * incx. stridex should be non zero.
       beta: device pointer or host pointer to scalar beta
       y: Device pointer to the first vector y_1 on the GPU
       incy: [int]
          specifies the increment for the elements of each vector y_i
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size.
          This typically means stridey >= n * incy. stridey should be non zero.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsymvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasSsymvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsymvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsymvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasDsymvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsymvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsymvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasCsymvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsymvStridedBatched(object handle, object uplo, int n, object alpha, object AP, int lda, long strideA, object x, int incx, long stridex, object beta, object y, int incy, long stridey, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsymvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,batchCount))    # fully specified
    return (_hipblasZsymvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """BLAS Level 2 API

    syr performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    AP         device pointer storing matrix A.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       lda: [int]
          specifies the leading dimension of A.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyr__retval = hipblasStatus_t(chipblas.hipblasSsyr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSsyr__retval,)


@cython.embedsignature(True)
def hipblasDsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyr__retval = hipblasStatus_t(chipblas.hipblasDsyr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDsyr__retval,)


@cython.embedsignature(True)
def hipblasCsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyr__retval = hipblasStatus_t(chipblas.hipblasCsyr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCsyr__retval,)


@cython.embedsignature(True)
def hipblasZsyr(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyr__retval = hipblasStatus_t(chipblas.hipblasZsyr(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZsyr__retval,)


@cython.embedsignature(True)
def hipblasSsyrBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """BLAS Level 2 API

    syrBatched performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, x is an array of vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device array of device pointers storing each matrix A_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyrBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasSsyrBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyrBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasDsyrBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyrBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCsyrBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrBatched(object handle, object uplo, int n, object alpha, object x, int incx, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyrBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZsyrBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyrStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """BLAS Level 2 API

    syrStridedBatched performs the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device pointer to the first matrix A_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of each matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          specifies the pointer increment between vectors (x_i) and (x_i+1).
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyrStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasSsyrStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyrStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasDsyrStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyrStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCsyrStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrStridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyrStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZsyrStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """BLAS Level 2 API

    syr2 performs the matrix-vector operations

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device pointer storing matrix A.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
       y: device pointer storing vector y.
       incy: [int]
          specifies the increment for the elements of y.
       lda: [int]
          specifies the leading dimension of A.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyr2__retval = hipblasStatus_t(chipblas.hipblasSsyr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasSsyr2__retval,)


@cython.embedsignature(True)
def hipblasDsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyr2__retval = hipblasStatus_t(chipblas.hipblasDsyr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasDsyr2__retval,)


@cython.embedsignature(True)
def hipblasCsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyr2__retval = hipblasStatus_t(chipblas.hipblasCsyr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(y)._ptr,incy,
        hipblasComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasCsyr2__retval,)


@cython.embedsignature(True)
def hipblasZsyr2(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyr2__retval = hipblasStatus_t(chipblas.hipblasZsyr2(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda))    # fully specified
    return (_hipblasZsyr2__retval,)


@cython.embedsignature(True)
def hipblasSsyr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """BLAS Level 2 API

    syr2Batched performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrix, for i = 1 , ... , batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device array of device pointers storing each matrix A_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       y: device array of device pointers storing each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyr2Batched__retval = hipblasStatus_t(chipblas.hipblasSsyr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasSsyr2Batched__retval,)


@cython.embedsignature(True)
def hipblasDsyr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyr2Batched__retval = hipblasStatus_t(chipblas.hipblasDsyr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(y)._ptr,incy,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasDsyr2Batched__retval,)


@cython.embedsignature(True)
def hipblasCsyr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyr2Batched__retval = hipblasStatus_t(chipblas.hipblasCsyr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasCsyr2Batched__retval,)


@cython.embedsignature(True)
def hipblasZsyr2Batched(object handle, object uplo, int n, object alpha, object x, int incx, object y, int incy, object AP, int lda, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyr2Batched__retval = hipblasStatus_t(chipblas.hipblasZsyr2Batched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,batchCount))    # fully specified
    return (_hipblasZsyr2Batched__retval,)


@cython.embedsignature(True)
def hipblasSsyr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """BLAS Level 2 API

    syr2StridedBatched the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrices, for i = 1 , ... , batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    AP         device pointer to the first matrix A_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       n: [int]
          the number of rows and columns of each matrix A.
       alpha: device pointer or host pointer to scalar alpha.
       x: device pointer to the first vector x_1.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          specifies the pointer increment between vectors (x_i) and (x_i+1).
       y: device pointer to the first vector y_1.
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          specifies the pointer increment between vectors (y_i) and (y_i+1).
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsyr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsyr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const float *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasSsyr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsyr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsyr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <const double *>hip._util.types.DataHandle.from_pyobj(y)._ptr,incy,stridey,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasDsyr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsyr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsyr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasCsyr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyr2StridedBatched(object handle, object uplo, int n, object alpha, object x, int incx, long stridex, object y, int incy, long stridey, object AP, int lda, long strideA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsyr2StridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsyr2StridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(y)._ptr,incy,stridey,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,batchCount))    # fully specified
    return (_hipblasZsyr2StridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """BLAS Level 2 API

    tbmv performs one of the matrix-vector operations

        x := A*x      or
        x := A**T*x   or
        x := A**H*x,

    x is a vectors and A is a banded m by m matrix (see description below).

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    x         device pointer storing vector x.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
          HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
       transA: [hipblasOperation_t]
          indicates whether matrix A is tranposed (conjugated) or not.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
          1's and is not referenced.
          HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
       m: [int]
          the number of rows and columns of the matrix represented by A.
       k: [int]
          if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
          of the matrix A.
          if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
          of the matrix A.
          k must satisfy k > 0 && k < lda.
       AP: device pointer storing banded triangular matrix A.
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
       lda: [int]
          specifies the leading dimension of A. lda must satisfy lda > k.
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbmv__retval = hipblasStatus_t(chipblas.hipblasStbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStbmv__retval,)


@cython.embedsignature(True)
def hipblasDtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbmv__retval = hipblasStatus_t(chipblas.hipblasDtbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtbmv__retval,)


@cython.embedsignature(True)
def hipblasCtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbmv__retval = hipblasStatus_t(chipblas.hipblasCtbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtbmv__retval,)


@cython.embedsignature(True)
def hipblasZtbmv(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbmv__retval = hipblasStatus_t(chipblas.hipblasZtbmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtbmv__retval,)


@cython.embedsignature(True)
def hipblasStbmvBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx, int batchCount):
    """BLAS Level 2 API

    tbmvBatched performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device array of device pointer storing each vector x_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
          HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
       transA: [hipblasOperation_t]
          indicates whether each matrix A_i is tranposed (conjugated) or not.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
          1's and is not referenced.
          HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
       m: [int]
          the number of rows and columns of the matrix represented by each A_i.
       k: [int]
          if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
          of each matrix A_i.
          if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
          of each matrix A_i.
          k must satisfy k > 0 && k < lda.
       AP: device array of device pointers storing each banded triangular matrix A_i.
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
       lda: [int]
          specifies the leading dimension of each A_i. lda must satisfy lda > k.
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbmvBatched__retval = hipblasStatus_t(chipblas.hipblasStbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtbmvBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbmvBatched__retval = hipblasStatus_t(chipblas.hipblasDtbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtbmvBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbmvBatched__retval = hipblasStatus_t(chipblas.hipblasCtbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtbmvBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbmvBatched__retval = hipblasStatus_t(chipblas.hipblasZtbmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtbmvBatched__retval,)


@cython.embedsignature(True)
def hipblasStbmvStridedBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    tbmvStridedBatched performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device array to the first vector x_i of the batch.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
          HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
       transA: [hipblasOperation_t]
          indicates whether each matrix A_i is tranposed (conjugated) or not.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
          1's and is not referenced.
          HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
       m: [int]
          the number of rows and columns of the matrix represented by each A_i.
       k: [int]
          if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
          of each matrix A_i.
          if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
          of each matrix A_i.
          k must satisfy k > 0 && k < lda.
       AP: device array to the first matrix A_i of the batch. Stores each banded triangular matrix A_i.
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
       lda: [int]
          specifies the leading dimension of each A_i. lda must satisfy lda > k.
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_(i + 1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one x_i matrix to the next x_(i + 1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtbmvStridedBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtbmvStridedBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtbmvStridedBatched(object handle, object uplo, object transA, object diag, int m, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtbmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtbmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """BLAS Level 2 API

    tbsv solves

         A*x = b or A**T*x = b or A**H*x = b,

    where x and b are vectors and A is a banded triangular matrix.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    x         device pointer storing input vector b. Overwritten by the output vector x.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A*x = b
          HIPBLAS_OP_T: Solves A**T*x = b
          HIPBLAS_OP_C: Solves A**H*x = b
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
          of A are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
       n: [int]
          n specifies the number of rows of b. n >= 0.
       k: [int]
          if(uplo == HIPBLAS_FILL_MODE_UPPER)
          k specifies the number of super-diagonals of A.
          if(uplo == HIPBLAS_FILL_MODE_LOWER)
          k specifies the number of sub-diagonals of A.
          k >= 0.
       AP: device pointer storing the matrix A in banded format.
       lda: [int]
          specifies the leading dimension of A.
          lda >= (k + 1).
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbsv__retval = hipblasStatus_t(chipblas.hipblasStbsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStbsv__retval,)


@cython.embedsignature(True)
def hipblasDtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbsv__retval = hipblasStatus_t(chipblas.hipblasDtbsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtbsv__retval,)


@cython.embedsignature(True)
def hipblasCtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbsv__retval = hipblasStatus_t(chipblas.hipblasCtbsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtbsv__retval,)


@cython.embedsignature(True)
def hipblasZtbsv(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbsv__retval = hipblasStatus_t(chipblas.hipblasZtbsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtbsv__retval,)


@cython.embedsignature(True)
def hipblasStbsvBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx, int batchCount):
    """BLAS Level 2 API

    tbsvBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batchCount].

    The input vectors b_i are overwritten by the output vectors x_i.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device vector of device pointers storing each input vector b_i. Overwritten by each output
              vector x_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A_i*x_i = b_i
          HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
          HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
          of each A_i are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
       n: [int]
          n specifies the number of rows of each b_i. n >= 0.
       k: [int]
          if(uplo == HIPBLAS_FILL_MODE_UPPER)
          k specifies the number of super-diagonals of each A_i.
          if(uplo == HIPBLAS_FILL_MODE_LOWER)
          k specifies the number of sub-diagonals of each A_i.
          k >= 0.
       AP: device vector of device pointers storing each matrix A_i in banded format.
       lda: [int]
          specifies the leading dimension of each A_i.
          lda >= (k + 1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbsvBatched__retval = hipblasStatus_t(chipblas.hipblasStbsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStbsvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtbsvBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbsvBatched__retval = hipblasStatus_t(chipblas.hipblasDtbsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtbsvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtbsvBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbsvBatched__retval = hipblasStatus_t(chipblas.hipblasCtbsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtbsvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtbsvBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbsvBatched__retval = hipblasStatus_t(chipblas.hipblasZtbsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtbsvBatched__retval,)


@cython.embedsignature(True)
def hipblasStbsvStridedBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    tbsvStridedBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batchCount].

    The input vectors b_i are overwritten by the output vectors x_i.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by output vectors x.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A_i*x_i = b_i
          HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
          HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
          of each A_i are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
       n: [int]
          n specifies the number of rows of each b_i. n >= 0.
       k: [int]
          if(uplo == HIPBLAS_FILL_MODE_UPPER)
          k specifies the number of super-diagonals of each A_i.
          if(uplo == HIPBLAS_FILL_MODE_LOWER)
          k specifies the number of sub-diagonals of each A_i.
          k >= 0.
       AP: device pointer pointing to the first banded matrix A_1.
       lda: [int]
          specifies the leading dimension of each A_i.
          lda >= (k + 1).
       strideA: [hipblasStride]
          specifies the distance between the start of one matrix (A_i) and the next (A_i+1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          specifies the distance between the start of one vector (x_i) and the next (x_i+1).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStbsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStbsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStbsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtbsvStridedBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtbsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtbsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtbsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtbsvStridedBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtbsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtbsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtbsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtbsvStridedBatched(object handle, object uplo, object transA, object diag, int n, int k, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtbsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtbsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,n,k,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtbsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """BLAS Level 2 API

    tpmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.

    The vector x is overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of A. m >= 0.
       AP: device pointer storing matrix A,
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
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x. incx must not be zero.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpmv__retval = hipblasStatus_t(chipblas.hipblasStpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStpmv__retval,)


@cython.embedsignature(True)
def hipblasDtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpmv__retval = hipblasStatus_t(chipblas.hipblasDtpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtpmv__retval,)


@cython.embedsignature(True)
def hipblasCtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpmv__retval = hipblasStatus_t(chipblas.hipblasCtpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtpmv__retval,)


@cython.embedsignature(True)
def hipblasZtpmv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpmv__retval = hipblasStatus_t(chipblas.hipblasZtpmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtpmv__retval,)


@cython.embedsignature(True)
def hipblasStpmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """BLAS Level 2 API

    tpmvBatched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of matrices A_i. m >= 0.
       AP: device pointer storing pointer of matrices A_i,
          of dimension ( lda, m )
       x: device pointer storing vectors x_i.
       incx: [int]
          specifies the increment for the elements of vectors x_i.
       batchCount: [int]
          The number of batched matrices/vectors.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpmvBatched__retval = hipblasStatus_t(chipblas.hipblasStpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtpmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpmvBatched__retval = hipblasStatus_t(chipblas.hipblasDtpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtpmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpmvBatched__retval = hipblasStatus_t(chipblas.hipblasCtpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtpmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpmvBatched__retval = hipblasStatus_t(chipblas.hipblasZtpmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtpmvBatched__retval,)


@cython.embedsignature(True)
def hipblasStpmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    tpmvStridedBatched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of matrices A_i. m >= 0.
       AP: device pointer of the matrix A_0,
          of dimension ( lda, m )
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_{i + 1}
       x: device pointer storing the vector x_0.
       incx: [int]
          specifies the increment for the elements of one vector x.
       stridex: [hipblasStride]
          stride from the start of one x_i vector to the next x_{i + 1}
       batchCount: [int]
          The number of batched matrices/vectors.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtpmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtpmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtpmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtpmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtpmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """BLAS Level 2 API

    tpsv solves

         A*x = b or A**T*x = b, or A**H*x = b,

    where x and b are vectors and A is a triangular matrix stored in the packed format.

    The input vector b is overwritten by the output vector x.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    x         device pointer storing vector b on input, overwritten by x on output.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A*x = b
          HIPBLAS_OP_T: Solves A**T*x = b
          HIPBLAS_OP_C: Solves A**H*x = b
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
          of A are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of b. m >= 0.
       AP: device pointer storing the packed version of matrix A,
          of dimension >= (n * (n + 1) / 2)
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpsv__retval = hipblasStatus_t(chipblas.hipblasStpsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStpsv__retval,)


@cython.embedsignature(True)
def hipblasDtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpsv__retval = hipblasStatus_t(chipblas.hipblasDtpsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtpsv__retval,)


@cython.embedsignature(True)
def hipblasCtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpsv__retval = hipblasStatus_t(chipblas.hipblasCtpsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtpsv__retval,)


@cython.embedsignature(True)
def hipblasZtpsv(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpsv__retval = hipblasStatus_t(chipblas.hipblasZtpsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtpsv__retval,)


@cython.embedsignature(True)
def hipblasStpsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """BLAS Level 2 API

    tpsvBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batchCount].

    The input vectors b_i are overwritten by the output vectors x_i.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device array of device pointers storing each input vector b_i, overwritten by x_i on output.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A*x = b
          HIPBLAS_OP_T: Solves A**T*x = b
          HIPBLAS_OP_C: Solves A**H*x = b
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
          of each A_i are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each b_i. m >= 0.
       AP: device array of device pointers storing the packed versions of each matrix A_i,
          of dimension >= (n * (n + 1) / 2)
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          specifies the number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpsvBatched__retval = hipblasStatus_t(chipblas.hipblasStpsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStpsvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtpsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpsvBatched__retval = hipblasStatus_t(chipblas.hipblasDtpsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtpsvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtpsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpsvBatched__retval = hipblasStatus_t(chipblas.hipblasCtpsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtpsvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtpsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpsvBatched__retval = hipblasStatus_t(chipblas.hipblasZtpsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtpsvBatched__retval,)


@cython.embedsignature(True)
def hipblasStpsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    tpsvStridedBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batchCount].

    The input vectors b_i are overwritten by the output vectors x_i.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by each x_i on output.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: Solves A*x = b
          HIPBLAS_OP_T: Solves A**T*x = b
          HIPBLAS_OP_C: Solves A**H*x = b
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
          of each A_i are not used in computations).
          HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each b_i. m >= 0.
       AP: device pointer pointing to the first packed matrix A_1,
          of dimension >= (n * (n + 1) / 2)
       strideA: [hipblasStride]
          stride from the beginning of one packed matrix (AP_i) and the next (AP_i+1).
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the beginning of one vector (x_i) and the next (x_i+1).
       batchCount: [int]
          specifies the number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStpsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStpsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStpsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtpsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtpsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtpsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtpsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtpsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtpsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtpsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtpsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtpsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtpsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtpsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtpsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """BLAS Level 2 API

    trmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.

    The vector x is overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of A. m >= 0.
       AP: device pointer storing matrix A,
          of dimension ( lda, m )
       lda: [int]
          specifies the leading dimension of A.
          lda = max( 1, m ).
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmv__retval = hipblasStatus_t(chipblas.hipblasStrmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStrmv__retval,)


@cython.embedsignature(True)
def hipblasDtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmv__retval = hipblasStatus_t(chipblas.hipblasDtrmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtrmv__retval,)


@cython.embedsignature(True)
def hipblasCtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmv__retval = hipblasStatus_t(chipblas.hipblasCtrmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtrmv__retval,)


@cython.embedsignature(True)
def hipblasZtrmv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmv__retval = hipblasStatus_t(chipblas.hipblasZtrmv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtrmv__retval,)


@cython.embedsignature(True)
def hipblasStrmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """BLAS Level 2 API

    trmvBatched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of matrices A_i. m >= 0.
       AP: device pointer storing pointer of matrices A_i,
          of dimension ( lda, m )
       lda: [int]
          specifies the leading dimension of A_i.
          lda >= max( 1, m ).
       x: device pointer storing vectors x_i.
       incx: [int]
          specifies the increment for the elements of vectors x_i.
       batchCount: [int]
          The number of batched matrices/vectors.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmvBatched__retval = hipblasStatus_t(chipblas.hipblasStrmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStrmvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmvBatched__retval = hipblasStatus_t(chipblas.hipblasDtrmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtrmvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmvBatched__retval = hipblasStatus_t(chipblas.hipblasCtrmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtrmvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrmvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmvBatched__retval = hipblasStatus_t(chipblas.hipblasZtrmvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtrmvBatched__retval,)


@cython.embedsignature(True)
def hipblasStrmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    trmvStridedBatched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of matrices A_i. m >= 0.
       AP: device pointer of the matrix A_0,
          of dimension ( lda, m )
       lda: [int]
          specifies the leading dimension of A_i.
          lda >= max( 1, m ).
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_{i + 1}
       x: device pointer storing the vector x_0.
       incx: [int]
          specifies the increment for the elements of one vector x.
       stridex: [hipblasStride]
          stride from the start of one x_i vector to the next x_{i + 1}
       batchCount: [int]
          The number of batched matrices/vectors.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStrmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStrmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtrmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtrmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtrmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtrmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrmvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtrmvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtrmvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """BLAS Level 2 API

    trsv solves

         A*x = b or A**T*x = b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of b. m >= 0.
       AP: device pointer storing matrix A,
          of dimension ( lda, m )
       lda: [int]
          specifies the leading dimension of A.
          lda = max( 1, m ).
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment for the elements of x.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsv__retval = hipblasStatus_t(chipblas.hipblasStrsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasStrsv__retval,)


@cython.embedsignature(True)
def hipblasDtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsv__retval = hipblasStatus_t(chipblas.hipblasDtrsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasDtrsv__retval,)


@cython.embedsignature(True)
def hipblasCtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsv__retval = hipblasStatus_t(chipblas.hipblasCtrsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasCtrsv__retval,)


@cython.embedsignature(True)
def hipblasZtrsv(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsv__retval = hipblasStatus_t(chipblas.hipblasZtrsv(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx))    # fully specified
    return (_hipblasZtrsv__retval,)


@cython.embedsignature(True)
def hipblasStrsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """BLAS Level 2 API

    trsvBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an
    m by m triangular matrix.

    The vector x is overwritten on b.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of b. m >= 0.
       AP: device array of device pointers storing each matrix A_i.
       lda: [int]
          specifies the leading dimension of each A_i.
          lda = max(1, m)
       x: device array of device pointers storing each vector x_i.
       incx: [int]
          specifies the increment for the elements of x.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsvBatched__retval = hipblasStatus_t(chipblas.hipblasStrsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasStrsvBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsvBatched__retval = hipblasStatus_t(chipblas.hipblasDtrsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasDtrsvBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsvBatched__retval = hipblasStatus_t(chipblas.hipblasCtrsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasCtrsvBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrsvBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, object x, int incx, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsvBatched__retval = hipblasStatus_t(chipblas.hipblasZtrsvBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,batchCount))    # fully specified
    return (_hipblasZtrsvBatched__retval,)


@cython.embedsignature(True)
def hipblasStrsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """BLAS Level 2 API

    trsvStridedBatched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an m by m triangular matrix, for i = 1, ..., batchCount.

    The vector x is overwritten on b.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each b_i. m >= 0.
       AP: device pointer to the first matrix (A_1) in the batch, of dimension ( lda, m )
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_(i + 1)
       lda: [int]
          specifies the leading dimension of each A_i.
          lda = max( 1, m ).
       x: device pointer to the first vector (x_1) in the batch.
       stridex: [hipblasStride]
          stride from the start of one x_i vector to the next x_(i + 1)
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStrsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasStrsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtrsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasDtrsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtrsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasCtrsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrsvStridedBatched(object handle, object uplo, object transA, object diag, int m, object AP, int lda, long strideA, object x, int incx, long stridex, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsvStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtrsvStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,diag.value,m,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,batchCount))    # fully specified
    return (_hipblasZtrsvStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasHgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

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

    Args:
       handle: [hipblasHandle_t]
          .
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          number or rows of matrices op( A ) and C
       n: [int]
          number of columns of matrices op( B ) and C
       k: [int]
          number of columns of matrix op( A ) and number of rows of matrix op( B )
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device pointer storing matrix A.
       lda: [int]
          specifies the leading dimension of A.
       BP: device pointer storing matrix B.
       ldb: [int]
          specifies the leading dimension of B.
       beta: device pointer or host pointer specifying the scalar beta.
       CP: device pointer storing matrix C on the GPU.
       ldc: [int]
          specifies the leading dimension of C.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasHgemm__retval = hipblasStatus_t(chipblas.hipblasHgemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasHgemm__retval,)


@cython.embedsignature(True)
def hipblasSgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemm__retval = hipblasStatus_t(chipblas.hipblasSgemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSgemm__retval,)


@cython.embedsignature(True)
def hipblasDgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemm__retval = hipblasStatus_t(chipblas.hipblasDgemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDgemm__retval,)


@cython.embedsignature(True)
def hipblasCgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemm__retval = hipblasStatus_t(chipblas.hipblasCgemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCgemm__retval,)


@cython.embedsignature(True)
def hipblasZgemm(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemm__retval = hipblasStatus_t(chipblas.hipblasZgemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZgemm__retval,)


@cython.embedsignature(True)
def hipblasHgemmBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    gemmBatched performs one of the batched matrix-matrix operations
         C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.
     where op( X ) is one of
         op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
     alpha and beta are scalars, and A, B and C are strided batched matrices, with
    op( A ) an m by k by batchCount strided_batched matrix,
    op( B ) an k by n by batchCount strided_batched matrix and
    C an m by n by batchCount strided_batched matrix.

    - Supported precisions in rocBLAS : h,s,d,c,z
    - Supported precisions in cuBLAS  : h,s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          matrix dimention m.
       n: [int]
          matrix dimention n.
       k: [int]
          matrix dimention k.
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device array of device pointers storing each matrix A_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       BP: device array of device pointers storing each matrix B_i.
       ldb: [int]
          specifies the leading dimension of each B_i.
       beta: device pointer or host pointer specifying the scalar beta.
       CP: device array of device pointers storing each matrix C_i.
       ldc: [int]
          specifies the leading dimension of each C_i.
       batchCount: [int]
          number of gemm operations in the batch
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasHgemmBatched__retval = hipblasStatus_t(chipblas.hipblasHgemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const unsigned short *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <unsigned short *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasHgemmBatched__retval,)


@cython.embedsignature(True)
def hipblasSgemmBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemmBatched__retval = hipblasStatus_t(chipblas.hipblasSgemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSgemmBatched__retval,)


@cython.embedsignature(True)
def hipblasDgemmBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemmBatched__retval = hipblasStatus_t(chipblas.hipblasDgemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDgemmBatched__retval,)


@cython.embedsignature(True)
def hipblasCgemmBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemmBatched__retval = hipblasStatus_t(chipblas.hipblasCgemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCgemmBatched__retval,)


@cython.embedsignature(True)
def hipblasZgemmBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemmBatched__retval = hipblasStatus_t(chipblas.hipblasZgemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZgemmBatched__retval,)


@cython.embedsignature(True)
def hipblasHgemmStridedBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, long long strideA, object BP, int ldb, long long strideB, object beta, object CP, int ldc, long long strideC, int batchCount):
    """BLAS Level 3 API

    gemmStridedBatched performs one of the strided batched matrix-matrix operations

        C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are strided batched matrices, with
    op( A ) an m by k by batchCount strided_batched matrix,
    op( B ) an k by n by batchCount strided_batched matrix and
    C an m by n by batchCount strided_batched matrix.

    - Supported precisions in rocBLAS : h,s,d,c,z
    - Supported precisions in cuBLAS  : h,s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          matrix dimention m.
       n: [int]
          matrix dimention n.
       k: [int]
          matrix dimention k.
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device pointer pointing to the first matrix A_1.
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_(i + 1).
       BP: device pointer pointing to the first matrix B_1.
       ldb: [int]
          specifies the leading dimension of each B_i.
       strideB: [hipblasStride]
          stride from the start of one B_i matrix to the next B_(i + 1).
       beta: device pointer or host pointer specifying the scalar beta.
       CP: device pointer pointing to the first matrix C_1.
       ldc: [int]
          specifies the leading dimension of each C_i.
       strideC: [hipblasStride]
          stride from the start of one C_i matrix to the next C_(i + 1).
       batchCount: [int]
          number of gemm operatons in the batch
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasHgemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasHgemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const unsigned short *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasHgemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgemmStridedBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, long long strideA, object BP, int ldb, long long strideB, object beta, object CP, int ldc, long long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSgemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgemmStridedBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, long long strideA, object BP, int ldb, long long strideB, object beta, object CP, int ldc, long long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDgemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgemmStridedBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, long long strideA, object BP, int ldb, long long strideB, object beta, object CP, int ldc, long long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCgemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgemmStridedBatched(object handle, object transA, object transB, int m, int n, int k, object alpha, object AP, int lda, long long strideA, object BP, int ldb, long long strideB, object beta, object CP, int ldc, long long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZgemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCherk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """BLAS Level 3 API

    herk performs one of the matrix-matrix operations for a Hermitian rank-k update

    C := alpha*op( A )*op( A )^H + beta*C

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C is a n x n Hermitian matrix stored as either upper or lower.

        op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
        op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C:  op(A) = A^H
          HIPBLAS_ON_N:  op(A) = A
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherk__retval = hipblasStatus_t(chipblas.hipblasCherk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCherk__retval,)


@cython.embedsignature(True)
def hipblasZherk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherk__retval = hipblasStatus_t(chipblas.hipblasZherk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZherk__retval,)


@cython.embedsignature(True)
def hipblasCherkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    herkBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
        op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op(A) = A^H
          HIPBLAS_OP_N: op(A) = A
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherkBatched__retval = hipblasStatus_t(chipblas.hipblasCherkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCherkBatched__retval,)


@cython.embedsignature(True)
def hipblasZherkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherkBatched__retval = hipblasStatus_t(chipblas.hipblasZherkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZherkBatched__retval,)


@cython.embedsignature(True)
def hipblasCherkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    herkStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
        op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op(A) = A^H
          HIPBLAS_OP_N: op(A) = A
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCherkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCherkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZherkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZherkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZherkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCherkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    herkx performs one of the matrix-matrix operations for a Hermitian rank-k update

    C := alpha*op( A )*op( B )^H + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
    C is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
        op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
          HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: pointer storing matrix B on the GPU.
          Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
          only the upper/lower triangular part is accessed.
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherkx__retval = hipblasStatus_t(chipblas.hipblasCherkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCherkx__retval,)


@cython.embedsignature(True)
def hipblasZherkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherkx__retval = hipblasStatus_t(chipblas.hipblasZherkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZherkx__retval,)


@cython.embedsignature(True)
def hipblasCherkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    herkxBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op(A) = A^H
          HIPBLAS_OP_N: op(A) = A
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: device array of device pointers storing each matrix_i B of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherkxBatched__retval = hipblasStatus_t(chipblas.hipblasCherkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCherkxBatched__retval,)


@cython.embedsignature(True)
def hipblasZherkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherkxBatched__retval = hipblasStatus_t(chipblas.hipblasZherkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZherkxBatched__retval,)


@cython.embedsignature(True)
def hipblasCherkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    herkxStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update

    C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
          HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCherkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCherkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCherkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZherkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZherkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZherkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZherkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCher2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update

    C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
    C is a n x n Hermitian matrix stored as either upper or lower.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
        op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
          HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: pointer storing matrix B on the GPU.
          Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
          only the upper/lower triangular part is accessed.
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCher2k__retval = hipblasStatus_t(chipblas.hipblasCher2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCher2k__retval,)


@cython.embedsignature(True)
def hipblasZher2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZher2k__retval = hipblasStatus_t(chipblas.hipblasZher2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZher2k__retval,)


@cython.embedsignature(True)
def hipblasCher2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    her2kBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update

    C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op(A) = A^H
          HIPBLAS_OP_N: op(A) = A
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: device array of device pointers storing each matrix_i B of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCher2kBatched__retval = hipblasStatus_t(chipblas.hipblasCher2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCher2kBatched__retval,)


@cython.embedsignature(True)
def hipblasZher2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZher2kBatched__retval = hipblasStatus_t(chipblas.hipblasZher2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZher2kBatched__retval,)


@cython.embedsignature(True)
def hipblasCher2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    her2kStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update

    C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
    C_i is a n x n Hermitian matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
          HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU.
          The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCher2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCher2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCher2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZher2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZher2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZher2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZher2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    symm performs one of the matrix-matrix operations:

    C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
    C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B and C are m by n matrices, and
    A is a symmetric matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
          HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
       m: [int]
          m specifies the number of rows of B and C. m >= 0.
       n: [int]
          n specifies the number of columns of B and C. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A and B are not referenced.
       AP: pointer storing matrix A on the GPU.
          A is m by m if side == HIPBLAS_SIDE_LEFT
          A is n by n if side == HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       BP: pointer storing matrix B on the GPU.
          Matrix dimension is m by n
       ldb: [int]
          ldb specifies the first dimension of B. ldb >= max( 1, m )
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
          Matrix dimension is m by n
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, m )
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymm__retval = hipblasStatus_t(chipblas.hipblasSsymm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsymm__retval,)


@cython.embedsignature(True)
def hipblasDsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymm__retval = hipblasStatus_t(chipblas.hipblasDsymm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsymm__retval,)


@cython.embedsignature(True)
def hipblasCsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymm__retval = hipblasStatus_t(chipblas.hipblasCsymm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsymm__retval,)


@cython.embedsignature(True)
def hipblasZsymm(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymm__retval = hipblasStatus_t(chipblas.hipblasZsymm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsymm__retval,)


@cython.embedsignature(True)
def hipblasSsymmBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    symmBatched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
    C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a symmetric matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
          HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
       m: [int]
          m specifies the number of rows of B_i and C_i. m >= 0.
       n: [int]
          n specifies the number of columns of B_i and C_i. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i and B_i are not referenced.
       AP: device array of device pointers storing each matrix A_i on the GPU.
          A_i is m by m if side == HIPBLAS_SIDE_LEFT
          A_i is n by n if side == HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       BP: device array of device pointers storing each matrix B_i on the GPU.
          Matrix dimension is m by n
       ldb: [int]
          ldb specifies the first dimension of B_i. ldb >= max( 1, m )
       beta: beta specifies the scalar beta. When beta is
          zero then C_i need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          Matrix dimension is m by n
       ldc: [int]
          ldc specifies the first dimension of C_i. ldc >= max( 1, m )
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymmBatched__retval = hipblasStatus_t(chipblas.hipblasSsymmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSsymmBatched__retval,)


@cython.embedsignature(True)
def hipblasDsymmBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymmBatched__retval = hipblasStatus_t(chipblas.hipblasDsymmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDsymmBatched__retval,)


@cython.embedsignature(True)
def hipblasCsymmBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymmBatched__retval = hipblasStatus_t(chipblas.hipblasCsymmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCsymmBatched__retval,)


@cython.embedsignature(True)
def hipblasZsymmBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymmBatched__retval = hipblasStatus_t(chipblas.hipblasZsymmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZsymmBatched__retval,)


@cython.embedsignature(True)
def hipblasSsymmStridedBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    symmStridedBatched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
    C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a symmetric matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
          HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
       m: [int]
          m specifies the number of rows of B_i and C_i. m >= 0.
       n: [int]
          n specifies the number of columns of B_i and C_i. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i and B_i are not referenced.
       AP: device pointer to first matrix A_1
          A_i is m by m if side == HIPBLAS_SIDE_LEFT
          A_i is n by n if side == HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: device pointer to first matrix B_1 of dimension (ldb, n) on the GPU.
       ldb: [int]
          ldb specifies the first dimension of B_i. ldb >= max( 1, m )
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, m ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasSsymmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsymmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSsymmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsymmStridedBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasDsymmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsymmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDsymmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsymmStridedBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasCsymmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsymmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCsymmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsymmStridedBatched(object handle, object side, object uplo, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZsymmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsymmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZsymmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """BLAS Level 3 API

    syrk performs one of the matrix-matrix operations for a symmetric rank-k update

    C := alpha*op( A )*op( A )^T + beta*C

    where  alpha and beta are scalars, op(A) is an n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.

        op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
        op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T: op(A) = A^T
          HIPBLAS_OP_N: op(A) = A
          HIPBLAS_OP_C: op(A) = A^T
          HIPBLAS_OP_C is not supported for complex types, see cherk
          and zherk.
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrk__retval = hipblasStatus_t(chipblas.hipblasSsyrk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyrk__retval,)


@cython.embedsignature(True)
def hipblasDsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrk__retval = hipblasStatus_t(chipblas.hipblasDsyrk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyrk__retval,)


@cython.embedsignature(True)
def hipblasCsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrk__retval = hipblasStatus_t(chipblas.hipblasCsyrk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyrk__retval,)


@cython.embedsignature(True)
def hipblasZsyrk(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrk__retval = hipblasStatus_t(chipblas.hipblasZsyrk(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyrk__retval,)


@cython.embedsignature(True)
def hipblasSsyrkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    syrkBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
        op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T: op(A) = A^T
          HIPBLAS_OP_N: op(A) = A
          HIPBLAS_OP_C: op(A) = A^T
          HIPBLAS_OP_C is not supported for complex types, see cherk
          and zherk.
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrkBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSsyrkBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrkBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDsyrkBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrkBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCsyrkBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrkBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrkBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrkBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZsyrkBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyrkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    syrkStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
        op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T: op(A) = A^T
          HIPBLAS_OP_N: op(A) = A
          HIPBLAS_OP_C: op(A) = A^T
          HIPBLAS_OP_C is not supported for complex types, see cherk
          and zherk.
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU. on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSsyrkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDsyrkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCsyrkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrkStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrkStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrkStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZsyrkStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update

    C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
        op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
          HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A) and op(B). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: pointer storing matrix B on the GPU.
          Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
          only the upper/lower triangular part is accessed.
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyr2k__retval = hipblasStatus_t(chipblas.hipblasSsyr2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyr2k__retval,)


@cython.embedsignature(True)
def hipblasDsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyr2k__retval = hipblasStatus_t(chipblas.hipblasDsyr2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyr2k__retval,)


@cython.embedsignature(True)
def hipblasCsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyr2k__retval = hipblasStatus_t(chipblas.hipblasCsyr2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyr2k__retval,)


@cython.embedsignature(True)
def hipblasZsyr2k(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyr2k__retval = hipblasStatus_t(chipblas.hipblasZsyr2k(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyr2k__retval,)


@cython.embedsignature(True)
def hipblasSsyr2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    syr2kBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update

    C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
          HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: device array of device pointers storing each matrix_i B of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyr2kBatched__retval = hipblasStatus_t(chipblas.hipblasSsyr2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSsyr2kBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyr2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyr2kBatched__retval = hipblasStatus_t(chipblas.hipblasDsyr2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDsyr2kBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyr2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyr2kBatched__retval = hipblasStatus_t(chipblas.hipblasCsyr2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCsyr2kBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyr2kBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyr2kBatched__retval = hipblasStatus_t(chipblas.hipblasZsyr2kBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZsyr2kBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyr2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    syr2kStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update

    C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
          HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyr2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsyr2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSsyr2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyr2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyr2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsyr2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDsyr2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyr2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyr2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsyr2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCsyr2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyr2kStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyr2kStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsyr2kStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZsyr2kStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    syrkx performs one of the matrix-matrix operations for a symmetric rank-k update

    C := alpha*op( A )*op( B )^T + beta*C

    where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
    C is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.

        op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
        op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
          HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
       n: [int]
          n specifies the number of rows and columns of C. n >= 0.
       k: [int]
          k specifies the number of columns of op(A) and op(B). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: pointer storing matrix A on the GPU.
          Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: pointer storing matrix B on the GPU.
          Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
          only the upper/lower triangular part is accessed.
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrkx__retval = hipblasStatus_t(chipblas.hipblasSsyrkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSsyrkx__retval,)


@cython.embedsignature(True)
def hipblasDsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrkx__retval = hipblasStatus_t(chipblas.hipblasDsyrkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDsyrkx__retval,)


@cython.embedsignature(True)
def hipblasCsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrkx__retval = hipblasStatus_t(chipblas.hipblasCsyrkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCsyrkx__retval,)


@cython.embedsignature(True)
def hipblasZsyrkx(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrkx__retval = hipblasStatus_t(chipblas.hipblasZsyrkx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZsyrkx__retval,)


@cython.embedsignature(True)
def hipblasSsyrkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    syrkxBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
          HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: device array of device pointers storing each matrix_i A of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       BP: device array of device pointers storing each matrix_i B of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrkxBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSsyrkxBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrkxBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDsyrkxBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrkxBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCsyrkxBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrkxBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrkxBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrkxBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZsyrkxBatched__retval,)


@cython.embedsignature(True)
def hipblasSsyrkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    syrkxStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update

    C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i

    where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
    C_i is a symmetric n x n matrix stored as either upper or lower.
    This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.

        op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
        op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
       transA: [hipblasOperation_t]
          HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
          HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
       n: [int]
          n specifies the number of rows and columns of C_i. n >= 0.
       k: [int]
          k specifies the number of columns of op(A). k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and A need not be set before
          entry.
       AP: Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
       lda: [int]
          lda specifies the first dimension of A_i.
          if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
          otherwise lda >= max( 1, k ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
          when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
          otherwise ldb >= max( 1, k ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: Device pointer to the first matrix C_1 on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, n ).
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasSsyrkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSsyrkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSsyrkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDsyrkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasDsyrkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDsyrkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDsyrkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCsyrkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasCsyrkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCsyrkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCsyrkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZsyrkxStridedBatched(object handle, object uplo, object transA, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")
    _hipblasZsyrkxStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZsyrkxStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,transA.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZsyrkxStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """BLAS Level 3 API

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device pointer storing matrix A.
       lda: [int]
          specifies the leading dimension of A.
       beta: device pointer or host pointer specifying the scalar beta.
       BP: device pointer storing matrix B.
       ldb: [int]
          specifies the leading dimension of B.
       CP: device pointer storing matrix C.
       ldc: [int]
          specifies the leading dimension of C.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgeam__retval = hipblasStatus_t(chipblas.hipblasSgeam(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSgeam__retval,)


@cython.embedsignature(True)
def hipblasDgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgeam__retval = hipblasStatus_t(chipblas.hipblasDgeam(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDgeam__retval,)


@cython.embedsignature(True)
def hipblasCgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgeam__retval = hipblasStatus_t(chipblas.hipblasCgeam(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCgeam__retval,)


@cython.embedsignature(True)
def hipblasZgeam(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgeam__retval = hipblasStatus_t(chipblas.hipblasZgeam(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZgeam__retval,)


@cython.embedsignature(True)
def hipblasSgeamBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    geamBatched performs one of the batched matrix-matrix operations

        C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1

    where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
    and op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device array of device pointers storing each matrix A_i on the GPU.
          Each A_i is of dimension ( lda, k ), where k is m
          when  transA == HIPBLAS_OP_N and
          is  n  when  transA == HIPBLAS_OP_T.
       lda: [int]
          specifies the leading dimension of A.
       beta: device pointer or host pointer specifying the scalar beta.
       BP: device array of device pointers storing each matrix B_i on the GPU.
          Each B_i is of dimension ( ldb, k ), where k is m
          when  transB == HIPBLAS_OP_N and
          is  n  when  transB == HIPBLAS_OP_T.
       ldb: [int]
          specifies the leading dimension of B.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          Each C_i is of dimension ( ldc, n ).
       ldc: [int]
          specifies the leading dimension of C.
       batchCount: [int]
          number of instances i in the batch.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgeamBatched__retval = hipblasStatus_t(chipblas.hipblasSgeamBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSgeamBatched__retval,)


@cython.embedsignature(True)
def hipblasDgeamBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgeamBatched__retval = hipblasStatus_t(chipblas.hipblasDgeamBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDgeamBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeamBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgeamBatched__retval = hipblasStatus_t(chipblas.hipblasCgeamBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCgeamBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeamBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, object beta, object BP, int ldb, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgeamBatched__retval = hipblasStatus_t(chipblas.hipblasZgeamBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZgeamBatched__retval,)


@cython.embedsignature(True)
def hipblasSgeamStridedBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, long strideA, object beta, object BP, int ldb, long strideB, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    geamStridedBatched performs one of the batched matrix-matrix operations

        C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1

    where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
    and op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A )
       transB: [hipblasOperation_t]
          specifies the form of op( B )
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       alpha: device pointer or host pointer specifying the scalar alpha.
       AP: device pointer to the first matrix A_0 on the GPU.
          Each A_i is of dimension ( lda, k ), where k is m
          when  transA == HIPBLAS_OP_N and
          is  n  when  transA == HIPBLAS_OP_T.
       lda: [int]
          specifies the leading dimension of A.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       beta: device pointer or host pointer specifying the scalar beta.
       BP: pointer to the first matrix B_0 on the GPU.
          Each B_i is of dimension ( ldb, k ), where k is m
          when  transB == HIPBLAS_OP_N and
          is  n  when  transB == HIPBLAS_OP_T.
       ldb: [int]
          specifies the leading dimension of B.
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       CP: pointer to the first matrix C_0 on the GPU.
          Each C_i is of dimension ( ldc, n ).
       ldc: [int]
          specifies the leading dimension of C.
       strideC: [hipblasStride]
          stride from the start of one matrix (C_i) and the next one (C_i+1)
       batchCount: [int]
          number of instances i in the batch.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgeamStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgeamStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSgeamStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgeamStridedBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, long strideA, object beta, object BP, int ldb, long strideB, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgeamStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgeamStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDgeamStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeamStridedBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, long strideA, object beta, object BP, int ldb, long strideB, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgeamStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgeamStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCgeamStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeamStridedBatched(object handle, object transA, object transB, int m, int n, object alpha, object AP, int lda, long strideA, object beta, object BP, int ldb, long strideB, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgeamStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgeamStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZgeamStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasChemm(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """BLAS Level 3 API

    hemm performs one of the matrix-matrix operations:

    C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
    C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B and C are m by n matrices, and
    A is a Hermitian matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
          HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
       n: [int]
          n specifies the number of rows of B and C. n >= 0.
       k: [int]
          n specifies the number of columns of B and C. k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A and B are not referenced.
       AP: pointer storing matrix A on the GPU.
          A is m by m if side == HIPBLAS_SIDE_LEFT
          A is n by n if side == HIPBLAS_SIDE_RIGHT
          Only the upper/lower triangular part is accessed.
          The imaginary component of the diagonal elements is not used.
       lda: [int]
          lda specifies the first dimension of A.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       BP: pointer storing matrix B on the GPU.
          Matrix dimension is m by n
       ldb: [int]
          ldb specifies the first dimension of B. ldb >= max( 1, m )
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: pointer storing matrix C on the GPU.
          Matrix dimension is m by n
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, m )
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemm__retval = hipblasStatus_t(chipblas.hipblasChemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasChemm__retval,)


@cython.embedsignature(True)
def hipblasZhemm(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemm__retval = hipblasStatus_t(chipblas.hipblasZhemm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZhemm__retval,)


@cython.embedsignature(True)
def hipblasChemmBatched(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    hemmBatched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
    C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a Hermitian matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
          HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
       n: [int]
          n specifies the number of rows of B_i and C_i. n >= 0.
       k: [int]
          k specifies the number of columns of B_i and C_i. k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i and B_i are not referenced.
       AP: device array of device pointers storing each matrix A_i on the GPU.
          A_i is m by m if side == HIPBLAS_SIDE_LEFT
          A_i is n by n if side == HIPBLAS_SIDE_RIGHT
          Only the upper/lower triangular part is accessed.
          The imaginary component of the diagonal elements is not used.
       lda: [int]
          lda specifies the first dimension of A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       BP: device array of device pointers storing each matrix B_i on the GPU.
          Matrix dimension is m by n
       ldb: [int]
          ldb specifies the first dimension of B_i. ldb >= max( 1, m )
       beta: beta specifies the scalar beta. When beta is
          zero then C_i need not be set before entry.
       CP: device array of device pointers storing each matrix C_i on the GPU.
          Matrix dimension is m by n
       ldc: [int]
          ldc specifies the first dimension of C_i. ldc >= max( 1, m )
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemmBatched__retval = hipblasStatus_t(chipblas.hipblasChemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasChemmBatched__retval,)


@cython.embedsignature(True)
def hipblasZhemmBatched(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, object BP, int ldb, object beta, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemmBatched__retval = hipblasStatus_t(chipblas.hipblasZhemmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZhemmBatched__retval,)


@cython.embedsignature(True)
def hipblasChemmStridedBatched(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    hemmStridedBatched performs a batch of the matrix-matrix operations:

    C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
    C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,

    where alpha and beta are scalars, B_i and C_i are m by n matrices, and
    A_i is a Hermitian matrix stored as either upper or lower.

    - Supported precisions in rocBLAS : c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    strideC  [hipblasStride]
              stride from the start of one matrix (C_i) and the next one (C_i+1)

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
          HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
          HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
       n: [int]
          n specifies the number of rows of B_i and C_i. n >= 0.
       k: [int]
          k specifies the number of columns of B_i and C_i. k >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i and B_i are not referenced.
       AP: device pointer to first matrix A_1
          A_i is m by m if side == HIPBLAS_SIDE_LEFT
          A_i is n by n if side == HIPBLAS_SIDE_RIGHT
          Only the upper/lower triangular part is accessed.
          The imaginary component of the diagonal elements is not used.
       lda: [int]
          lda specifies the first dimension of A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          otherwise lda >= max( 1, n ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       BP: device pointer to first matrix B_1 of dimension (ldb, n) on the GPU
       ldb: [int]
          ldb specifies the first dimension of B_i.
          if side = HIPBLAS_OP_N,  ldb >= max( 1, m ),
          otherwise ldb >= max( 1, n ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       beta: beta specifies the scalar beta. When beta is
          zero then C need not be set before entry.
       CP: device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
       ldc: [int]
          ldc specifies the first dimension of C. ldc >= max( 1, m )
       batchCount: [int]
          number of instances in the batch
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasChemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasChemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasComplex.from_pyobj(beta)._ptr,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasChemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZhemmStridedBatched(object handle, object side, object uplo, int n, int k, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, object beta, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")
    _hipblasZhemmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZhemmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,n,k,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,
        hipblasDoubleComplex.from_pyobj(beta)._ptr,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZhemmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """BLAS Level 3 API

    trmm performs one of the matrix-matrix operations

    B := alpha*op( A )*B,   or   B := alpha*B*op( A )

    where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    @param[inout]
    BP       Device pointer to the first matrix B_0 on the GPU.
            On entry,  the leading  m by n part of the array  B must
           contain the matrix  B,  and  on exit  is overwritten  by the
           transformed matrix.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          Specifies whether op(A) multiplies B from the left or right as follows:
          HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
          HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).
       uplo: [hipblasFillMode_t]
          Specifies whether the matrix A is an upper or lower triangular matrix as follows:
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          Specifies the form of op(A) to be used in the matrix multiplication as follows:
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C:  op(A) = A^H.
       diag: [hipblasDiagType_t]
          Specifies whether or not A is unit triangular as follows:
          HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of B. m >= 0.
       n: [int]
          n specifies the number of columns of B. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A is not referenced and B need not be set before
          entry.
       AP: Device pointer to matrix A on the GPU.
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
       lda: [int]
          lda specifies the first dimension of A.
          if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       ldb: [int]
          ldb specifies the first dimension of B. ldb >= max( 1, m ).
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmm__retval = hipblasStatus_t(chipblas.hipblasStrmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasStrmm__retval,)


@cython.embedsignature(True)
def hipblasDtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmm__retval = hipblasStatus_t(chipblas.hipblasDtrmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasDtrmm__retval,)


@cython.embedsignature(True)
def hipblasCtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmm__retval = hipblasStatus_t(chipblas.hipblasCtrmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasCtrmm__retval,)


@cython.embedsignature(True)
def hipblasZtrmm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmm__retval = hipblasStatus_t(chipblas.hipblasZtrmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasZtrmm__retval,)


@cython.embedsignature(True)
def hipblasStrmmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """BLAS Level 3 API

    trmmBatched performs one of the batched matrix-matrix operations

    B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1

    where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    BP       device array of device pointers storing each matrix B_i on the GPU.
            On entry,  the leading  m by n part of the array  B_i must
           contain the matrix  B_i,  and  on exit  is overwritten  by the
           transformed matrix.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          Specifies whether op(A_i) multiplies B_i from the left or right as follows:
          HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
          HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
       uplo: [hipblasFillMode_t]
          Specifies whether the matrix A is an upper or lower triangular matrix as follows:
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
          HIPBLAS_OP_N:    op(A_i) = A_i.
          HIPBLAS_OP_T:      op(A_i) = A_i^T.
          HIPBLAS_OP_C:  op(A_i) = A_i^H.
       diag: [hipblasDiagType_t]
          Specifies whether or not A_i is unit triangular as follows:
          HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of B_i. m >= 0.
       n: [int]
          n specifies the number of columns of B_i. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i is not referenced and B_i need not be set before
          entry.
       AP: Device array of device pointers storing each matrix A_i on the GPU.
          Each A_i is of dimension ( lda, k ), where k is m
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
          A_i  are not referenced either,  but are assumed to be  unity.
       lda: [int]
          lda specifies the first dimension of A.
          if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       ldb: [int]
          ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
       batchCount: [int]
          number of instances i in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmmBatched__retval = hipblasStatus_t(chipblas.hipblasStrmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasStrmmBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrmmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmmBatched__retval = hipblasStatus_t(chipblas.hipblasDtrmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasDtrmmBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrmmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmmBatched__retval = hipblasStatus_t(chipblas.hipblasCtrmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasCtrmmBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrmmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmmBatched__retval = hipblasStatus_t(chipblas.hipblasZtrmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasZtrmmBatched__retval,)


@cython.embedsignature(True)
def hipblasStrmmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """BLAS Level 3 API

    trmmStridedBatched performs one of the strided_batched matrix-matrix operations

    B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1

    where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    @param[inout]
    BP       Device pointer to the first matrix B_0 on the GPU.
            On entry,  the leading  m by n part of the array  B_i must
           contain the matrix  B_i,  and  on exit  is overwritten  by the
           transformed matrix.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          Specifies whether op(A_i) multiplies B_i from the left or right as follows:
          HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
          HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
       uplo: [hipblasFillMode_t]
          Specifies whether the matrix A is an upper or lower triangular matrix as follows:
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
          HIPBLAS_OP_N:    op(A_i) = A_i.
          HIPBLAS_OP_T:      op(A_i) = A_i^T.
          HIPBLAS_OP_C:  op(A_i) = A_i^H.
       diag: [hipblasDiagType_t]
          Specifies whether or not A_i is unit triangular as follows:
          HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of B_i. m >= 0.
       n: [int]
          n specifies the number of columns of B_i. n >= 0.
       alpha: alpha specifies the scalar alpha. When alpha is
          zero then A_i is not referenced and B_i need not be set before
          entry.
       AP: Device pointer to the first matrix A_0 on the GPU.
          Each A_i is of dimension ( lda, k ), where k is m
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
          A_i  are not referenced either,  but are assumed to be  unity.
       lda: [int]
          lda specifies the first dimension of A.
          if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       ldb: [int]
          ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
       strideB: [hipblasStride]
          stride from the start of one matrix (B_i) and the next one (B_i+1)
       batchCount: [int]
          number of instances i in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStrmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasStrmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrmmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtrmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasDtrmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrmmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtrmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasCtrmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrmmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtrmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasZtrmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """BLAS Level 3 API

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of B. m >= 0.
       n: [int]
          n specifies the number of columns of B. n >= 0.
       alpha: device pointer or host pointer specifying the scalar alpha. When alpha is
          &zero then A is not referenced and B need not be set before
          entry.
       AP: device pointer storing matrix A.
          of dimension ( lda, k ), where k is m
          when  HIPBLAS_SIDE_LEFT  and
          is  n  when  HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       BP: device pointer storing matrix B.
       ldb: [int]
          ldb specifies the first dimension of B. ldb >= max( 1, m ).
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsm__retval = hipblasStatus_t(chipblas.hipblasStrsm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasStrsm__retval,)


@cython.embedsignature(True)
def hipblasDtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsm__retval = hipblasStatus_t(chipblas.hipblasDtrsm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasDtrsm__retval,)


@cython.embedsignature(True)
def hipblasCtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsm__retval = hipblasStatus_t(chipblas.hipblasCtrsm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasCtrsm__retval,)


@cython.embedsignature(True)
def hipblasZtrsm(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsm__retval = hipblasStatus_t(chipblas.hipblasZtrsm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb))    # fully specified
    return (_hipblasZtrsm__retval,)


@cython.embedsignature(True)
def hipblasStrsmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """BLAS Level 3 API

    trsmBatched performs the following batched operation:

        op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.

    where alpha is a scalar, X and B are batched m by n matrices,
    A is triangular batched matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each B_i. m >= 0.
       n: [int]
          n specifies the number of columns of each B_i. n >= 0.
       alpha: device pointer or host pointer specifying the scalar alpha. When alpha is
          &zero then A is not referenced and B need not be set before
          entry.
       AP: device array of device pointers storing each matrix A_i on the GPU.
          Matricies are of dimension ( lda, k ), where k is m
          when  HIPBLAS_SIDE_LEFT  and is  n  when  HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of each A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       BP: device array of device pointers storing each matrix B_i on the GPU.
       ldb: [int]
          ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
       batchCount: [int]
          number of trsm operatons in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsmBatched__retval = hipblasStatus_t(chipblas.hipblasStrsmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float **>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasStrsmBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrsmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsmBatched__retval = hipblasStatus_t(chipblas.hipblasDtrsmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double **>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasDtrsmBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrsmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsmBatched__retval = hipblasStatus_t(chipblas.hipblasCtrsmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex **>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasCtrsmBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrsmBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, object BP, int ldb, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsmBatched__retval = hipblasStatus_t(chipblas.hipblasZtrsmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex **>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,batchCount))    # fully specified
    return (_hipblasZtrsmBatched__retval,)


@cython.embedsignature(True)
def hipblasStrsmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """BLAS Level 3 API

    trsmSridedBatched performs the following strided batched operation:

        op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.

    where alpha is a scalar, X and B are strided batched m by n matrices,
    A is triangular strided batched matrix and op(A) is one of

        op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.

    Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.

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
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each B_i. m >= 0.
       n: [int]
          n specifies the number of columns of each B_i. n >= 0.
       alpha: device pointer or host pointer specifying the scalar alpha. When alpha is
          &zero then A is not referenced and B need not be set before
          entry.
       AP: device pointer pointing to the first matrix A_1.
          of dimension ( lda, k ), where k is m
          when  HIPBLAS_SIDE_LEFT  and
          is  n  when  HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of each A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       strideA: [hipblasStride]
          stride from the start of one A_i matrix to the next A_(i + 1).
       BP: device pointer pointing to the first matrix B_1.
       ldb: [int]
          ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
       strideB: [hipblasStride]
          stride from the start of one B_i matrix to the next B_(i + 1).
       batchCount: [int]
          number of trsm operatons in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrsmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStrsmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasStrsmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrsmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrsmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtrsmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasDtrsmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrsmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrsmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtrsmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasComplex.from_pyobj(alpha)._ptr,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasCtrsmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrsmStridedBatched(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object AP, int lda, long strideA, object BP, int ldb, long strideB, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrsmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtrsmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        hipblasDoubleComplex.from_pyobj(alpha)._ptr,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(BP)._ptr,ldb,strideB,batchCount))    # fully specified
    return (_hipblasZtrsmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasStrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """BLAS Level 3 API

    trtri  compute the inverse of a matrix A, namely, invA

    and write the result into invA;

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    ******************************************************************
    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
          if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
          if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
       diag: [hipblasDiagType_t]
          = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
          = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
       n: [int]
          size of matrix A and invA
       AP: device pointer storing matrix A.
       lda: [int]
          specifies the leading dimension of A.
       invA: device pointer storing matrix invA.
       ldinvA: [int]
          specifies the leading dimension of invA.
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrtri__retval = hipblasStatus_t(chipblas.hipblasStrtri(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasStrtri__retval,)


@cython.embedsignature(True)
def hipblasDtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrtri__retval = hipblasStatus_t(chipblas.hipblasDtrtri(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasDtrtri__retval,)


@cython.embedsignature(True)
def hipblasCtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrtri__retval = hipblasStatus_t(chipblas.hipblasCtrtri(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasCtrtri__retval,)


@cython.embedsignature(True)
def hipblasZtrtri(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrtri__retval = hipblasStatus_t(chipblas.hipblasZtrtri(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(invA)._ptr,ldinvA))    # fully specified
    return (_hipblasZtrtri__retval,)


@cython.embedsignature(True)
def hipblasStrtriBatched(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA, int batchCount):
    """BLAS Level 3 API

    trtriBatched  compute the inverse of A_i and write into invA_i where
                   A_i and invA_i are the i-th matrices in the batch,
                   for i = 1, ..., batchCount.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
       diag: [hipblasDiagType_t]
          = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
          = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
       n: [int]
       AP: device array of device pointers storing each matrix A_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       invA: device array of device pointers storing the inverse of each matrix A_i.
          Partial inplace operation is supported, see below.
          If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
          the inverse of the upper triangular matrix, and the strictly lower
          triangular part of invA is cleared.
          If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
          the inverse of the lower triangular matrix, and the strictly upper
          triangular part of invA is cleared.
       ldinvA: [int]
          specifies the leading dimension of each invA_i.
       batchCount: [int]
          numbers of matrices in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrtriBatched__retval = hipblasStatus_t(chipblas.hipblasStrtriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <float **>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,batchCount))    # fully specified
    return (_hipblasStrtriBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrtriBatched(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrtriBatched__retval = hipblasStatus_t(chipblas.hipblasDtrtriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <double **>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,batchCount))    # fully specified
    return (_hipblasDtrtriBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrtriBatched(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrtriBatched__retval = hipblasStatus_t(chipblas.hipblasCtrtriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex **>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,batchCount))    # fully specified
    return (_hipblasCtrtriBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrtriBatched(object handle, object uplo, object diag, int n, object AP, int lda, object invA, int ldinvA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrtriBatched__retval = hipblasStatus_t(chipblas.hipblasZtrtriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex **>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,batchCount))    # fully specified
    return (_hipblasZtrtriBatched__retval,)


@cython.embedsignature(True)
def hipblasStrtriStridedBatched(object handle, object uplo, object diag, int n, object AP, int lda, long strideA, object invA, int ldinvA, long stride_invA, int batchCount):
    """BLAS Level 3 API

    trtriStridedBatched compute the inverse of A_i and write into invA_i where
                   A_i and invA_i are the i-th matrices in the batch,
                   for i = 1, ..., batchCount

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       uplo: [hipblasFillMode_t]
          specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
       diag: [hipblasDiagType_t]
          = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
          = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
       n: [int]
       AP: device pointer pointing to address of first matrix A_1.
       lda: [int]
          specifies the leading dimension of each A.
       strideA: [hipblasStride]
          "batch stride a": stride from the start of one A_i matrix to the next A_(i + 1).
       invA: device pointer storing the inverses of each matrix A_i.
          Partial inplace operation is supported, see below.
          If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
          the inverse of the upper triangular matrix, and the strictly lower
          triangular part of invA is cleared.
          If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
          the inverse of the lower triangular matrix, and the strictly upper
          triangular part of invA is cleared.
       ldinvA: [int]
          specifies the leading dimension of each invA_i.
       stride_invA: [hipblasStride]
          "batch stride invA": stride from the start of one invA_i matrix to the next invA_(i + 1).
       batchCount: [int]
          numbers of matrices in the batch
          ******************************************************************
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasStrtriStridedBatched__retval = hipblasStatus_t(chipblas.hipblasStrtriStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,stride_invA,batchCount))    # fully specified
    return (_hipblasStrtriStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDtrtriStridedBatched(object handle, object uplo, object diag, int n, object AP, int lda, long strideA, object invA, int ldinvA, long stride_invA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasDtrtriStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDtrtriStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,ldinvA,stride_invA,batchCount))    # fully specified
    return (_hipblasDtrtriStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCtrtriStridedBatched(object handle, object uplo, object diag, int n, object AP, int lda, long strideA, object invA, int ldinvA, long stride_invA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasCtrtriStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCtrtriStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(invA)._ptr,ldinvA,stride_invA,batchCount))    # fully specified
    return (_hipblasCtrtriStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZtrtriStridedBatched(object handle, object uplo, object diag, int n, object AP, int lda, long strideA, object invA, int ldinvA, long stride_invA, int batchCount):
    """(No brief)
    """
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")
    _hipblasZtrtriStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZtrtriStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,uplo.value,diag.value,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(invA)._ptr,ldinvA,stride_invA,batchCount))    # fully specified
    return (_hipblasZtrtriStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """BLAS Level 3 API

    dgmm performs one of the matrix-matrix operations

        C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
        C = diag(x) * A if side == HIPBLAS_SIDE_LEFT

    where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
    and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
    if side == HIPBLAS_SIDE_LEFT.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : s,d,c,z

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          specifies the side of diag(x)
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       AP: device pointer storing matrix A.
       lda: [int]
          specifies the leading dimension of A.
       x: device pointer storing vector x.
       incx: [int]
          specifies the increment between values of x
       CP: device pointer storing matrix C.
       ldc: [int]
          specifies the leading dimension of C.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasSdgmm__retval = hipblasStatus_t(chipblas.hipblasSdgmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasSdgmm__retval,)


@cython.embedsignature(True)
def hipblasDdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasDdgmm__retval = hipblasStatus_t(chipblas.hipblasDdgmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasDdgmm__retval,)


@cython.embedsignature(True)
def hipblasCdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasCdgmm__retval = hipblasStatus_t(chipblas.hipblasCdgmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,
        hipblasComplex.from_pyobj(x)._ptr,incx,
        hipblasComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasCdgmm__retval,)


@cython.embedsignature(True)
def hipblasZdgmm(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasZdgmm__retval = hipblasStatus_t(chipblas.hipblasZdgmm(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc))    # fully specified
    return (_hipblasZdgmm__retval,)


@cython.embedsignature(True)
def hipblasSdgmmBatched(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc, int batchCount):
    """BLAS Level 3 API

    dgmmBatched performs one of the batched matrix-matrix operations

        C_i = A_i * diag(x_i) for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_RIGHT
        C_i = diag(x_i) * A_i for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_LEFT

    where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
    and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
    if side == HIPBLAS_SIDE_LEFT.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          specifies the side of diag(x)
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       AP: device array of device pointers storing each matrix A_i on the GPU.
          Each A_i is of dimension ( lda, n )
       lda: [int]
          specifies the leading dimension of A_i.
       x: device array of device pointers storing each vector x_i on the GPU.
          Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
          m if side == HIPBLAS_SIDE_LEFT
       incx: [int]
          specifies the increment between values of x_i
       CP: device array of device pointers storing each matrix C_i on the GPU.
          Each C_i is of dimension ( ldc, n ).
       ldc: [int]
          specifies the leading dimension of C_i.
       batchCount: [int]
          number of instances in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasSdgmmBatched__retval = hipblasStatus_t(chipblas.hipblasSdgmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const float *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const float *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <float *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasSdgmmBatched__retval,)


@cython.embedsignature(True)
def hipblasDdgmmBatched(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasDdgmmBatched__retval = hipblasStatus_t(chipblas.hipblasDdgmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const double *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <const double *const*>hip._util.types.ListOfDataHandle.from_pyobj(x)._ptr,incx,
        <double *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasDdgmmBatched__retval,)


@cython.embedsignature(True)
def hipblasCdgmmBatched(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasCdgmmBatched__retval = hipblasStatus_t(chipblas.hipblasCdgmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasCdgmmBatched__retval,)


@cython.embedsignature(True)
def hipblasZdgmmBatched(object handle, object side, int m, int n, object AP, int lda, object x, int incx, object CP, int ldc, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasZdgmmBatched__retval = hipblasStatus_t(chipblas.hipblasZdgmmBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,batchCount))    # fully specified
    return (_hipblasZdgmmBatched__retval,)


@cython.embedsignature(True)
def hipblasSdgmmStridedBatched(object handle, object side, int m, int n, object AP, int lda, long strideA, object x, int incx, long stridex, object CP, int ldc, long strideC, int batchCount):
    """BLAS Level 3 API

    dgmmStridedBatched performs one of the batched matrix-matrix operations

        C_i = A_i * diag(x_i)   if side == HIPBLAS_SIDE_RIGHT   for i = 0, 1, ... batchCount-1
        C_i = diag(x_i) * A_i   if side == HIPBLAS_SIDE_LEFT    for i = 0, 1, ... batchCount-1

    where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
    and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
    if side == HIPBLAS_SIDE_LEFT.

    - Supported precisions in rocBLAS : s,d,c,z
    - Supported precisions in cuBLAS  : No support

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          specifies the side of diag(x)
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       AP: device pointer to the first matrix A_0 on the GPU.
          Each A_i is of dimension ( lda, n )
       lda: [int]
          specifies the leading dimension of A.
       strideA: [hipblasStride]
          stride from the start of one matrix (A_i) and the next one (A_i+1)
       x: pointer to the first vector x_0 on the GPU.
          Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
          m if side == HIPBLAS_SIDE_LEFT
       incx: [int]
          specifies the increment between values of x
       stridex: [hipblasStride]
          stride from the start of one vector(x_i) and the next one (x_i+1)
       CP: device pointer to the first matrix C_0 on the GPU.
          Each C_i is of dimension ( ldc, n ).
       ldc: [int]
          specifies the leading dimension of C.
       strideC: [hipblasStride]
          stride from the start of one matrix (C_i) and the next one (C_i+1)
       batchCount: [int]
          number of instances i in the batch.
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasSdgmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSdgmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const float *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const float *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <float *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasSdgmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDdgmmStridedBatched(object handle, object side, int m, int n, object AP, int lda, long strideA, object x, int incx, long stridex, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasDdgmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDdgmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        <const double *>hip._util.types.DataHandle.from_pyobj(AP)._ptr,lda,strideA,
        <const double *>hip._util.types.DataHandle.from_pyobj(x)._ptr,incx,stridex,
        <double *>hip._util.types.DataHandle.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasDdgmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCdgmmStridedBatched(object handle, object side, int m, int n, object AP, int lda, long strideA, object x, int incx, long stridex, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasCdgmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCdgmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasCdgmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZdgmmStridedBatched(object handle, object side, int m, int n, object AP, int lda, long strideA, object x, int incx, long stridex, object CP, int ldc, long strideC, int batchCount):
    """(No brief)
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")
    _hipblasZdgmmStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZdgmmStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,m,n,
        hipblasDoubleComplex.from_pyobj(AP)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(x)._ptr,incx,stridex,
        hipblasDoubleComplex.from_pyobj(CP)._ptr,ldc,strideC,batchCount))    # fully specified
    return (_hipblasZdgmmStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """SOLVER API

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

    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.

              On entry, the n-by-n matrix A to be factored.
              On exit, the factors L and U from the factorization.
              The unit diagonal elements of L are not stored.

    Args:
       handle: hipblasHandle_t.
       n: int. n >= 0.
          The number of columns and rows of the matrix A.
       lda: int. lda >= n.
          Specifies the leading dimension of A.
       ipiv: pointer to int. Array on the GPU of dimension n.
          The vector of pivot indices. Elements of ipiv are 1-based indices.
          For 1 <= i <= n, the row i of the
          matrix was interchanged with row ipiv[i].
          Matrix P of the factorization can be derived from ipiv.
          The factorization here can be done without pivoting if ipiv is passed
          in as a nullptr.
       info: pointer to a int on the GPU.
          If info = 0, successful exit.
          If info = j > 0, U is singular. U[j,j] is the first zero pivot.
          ******************************************************************
    """
    _hipblasSgetrf__retval = hipblasStatus_t(chipblas.hipblasSgetrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgetrf__retval,)


@cython.embedsignature(True)
def hipblasDgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasDgetrf__retval = hipblasStatus_t(chipblas.hipblasDgetrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgetrf__retval,)


@cython.embedsignature(True)
def hipblasCgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasCgetrf__retval = hipblasStatus_t(chipblas.hipblasCgetrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgetrf__retval,)


@cython.embedsignature(True)
def hipblasZgetrf(object handle, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasZgetrf__retval = hipblasStatus_t(chipblas.hipblasZgetrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgetrf__retval,)


@cython.embedsignature(True)
def hipblasSgetrfBatched(object handle, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """SOLVER API

    getrfBatched computes the LU factorization of a batch of general
    n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
    be done without pivoting if ipiv is passed as a nullptr.

    In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:

    \f[
        A_i = P_iL_iU_i
    \f]

    where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
    diagonal elements, and \f$U_i\f$ is upper triangular.

    In the case that ipiv is null, the factorization is done without pivoting:

    \f[
        A_i = L_iU_i
    \f]

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z

    @param[inout]
    A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.

              On entry, the n-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorizations.
              The unit diagonal elements of L_i are not stored.

    Args:
       handle: hipblasHandle_t.
       n: int. n >= 0.
          The number of columns and rows of all matrices A_i in the batch.
       lda: int. lda >= n.
          Specifies the leading dimension of matrices A_i.
       ipiv: pointer to int. Array on the GPU.
          Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
          Dimension of ipiv_i is n.
          Elements of ipiv_i are 1-based indices.
          For each instance A_i in the batch and for 1 <= j <= n, the row j of the
          matrix A_i was interchanged with row ipiv_i[j].
          Matrix P_i of the factorization can be derived from ipiv_i.
          The factorization here can be done without pivoting if ipiv is passed
          in as a nullptr.
       info: pointer to int. Array of batchCount integers on the GPU.
          If info[i] = 0, successful exit for factorization of A_i.
          If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    _hipblasSgetrfBatched__retval = hipblasStatus_t(chipblas.hipblasSgetrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgetrfBatched__retval,)


@cython.embedsignature(True)
def hipblasDgetrfBatched(object handle, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasDgetrfBatched__retval = hipblasStatus_t(chipblas.hipblasDgetrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgetrfBatched__retval,)


@cython.embedsignature(True)
def hipblasCgetrfBatched(object handle, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasCgetrfBatched__retval = hipblasStatus_t(chipblas.hipblasCgetrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgetrfBatched__retval,)


@cython.embedsignature(True)
def hipblasZgetrfBatched(object handle, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasZgetrfBatched__retval = hipblasStatus_t(chipblas.hipblasZgetrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgetrfBatched__retval,)


@cython.embedsignature(True)
def hipblasSgetrfStridedBatched(object handle, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """SOLVER API

    getrfStridedBatched computes the LU factorization of a batch of
    general n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
    be done without pivoting if ipiv is passed as a nullptr.

    In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:

    \f[
        A_i = P_iL_iU_i
    \f]

    where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
    diagonal elements, and \f$U_i\f$ is upper triangular.

    In the case that ipiv is null, the factorization is done without pivoting:

    \f[
        A_i = L_iU_i
    \f]

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z

    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).

              On entry, the n-by-n matrices A_i to be factored.
              On exit, the factors L_i and U_i from the factorization.
              The unit diagonal elements of L_i are not stored.

    Args:
       handle: hipblasHandle_t.
       n: int. n >= 0.
          The number of columns and rows of all matrices A_i in the batch.
       lda: int. lda >= n.
          Specifies the leading dimension of matrices A_i.
       strideA: hipblasStride.
          Stride from the start of one matrix A_i to the next one A_(i+1).
          There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
       ipiv: pointer to int. Array on the GPU (the size depends on the value of strideP).
          Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
          Dimension of ipiv_i is n.
          Elements of ipiv_i are 1-based indices.
          For each instance A_i in the batch and for 1 <= j <= n, the row j of the
          matrix A_i was interchanged with row ipiv_i[j].
          Matrix P_i of the factorization can be derived from ipiv_i.
          The factorization here can be done without pivoting if ipiv is passed
          in as a nullptr.
       strideP: hipblasStride.
          Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
          There is no restriction for the value of strideP. Normal use case is strideP >= n.
       info: pointer to int. Array of batchCount integers on the GPU.
          If info[i] = 0, successful exit for factorization of A_i.
          If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    _hipblasSgetrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgetrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgetrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgetrfStridedBatched(object handle, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasDgetrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgetrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgetrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgetrfStridedBatched(object handle, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasCgetrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgetrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,strideA,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgetrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgetrfStridedBatched(object handle, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasZgetrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgetrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,strideA,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgetrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """SOLVER API

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

     \ref hipblasSgetrf "getrf".

     \ref hipblasSgetrf "getrf".

    ******************************************************************
    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations.
       n: int. n >= 0.
          The order of the system, i.e. the number of columns and rows of A.
       nrhs: int. nrhs >= 0.
          The number of right hand sides, i.e., the number of columns
          of the matrix B.
       A: pointer to type. Array on the GPU of dimension lda*n.
          The factors L and U of the factorization A = P*L*U returned by
       lda: int. lda >= n.
          The leading dimension of A.
       ipiv: pointer to int. Array on the GPU of dimension n.
          The pivot indices returned by
       B: pointer to type. Array on the GPU of dimension ldb*nrhs.
          On entry, the right hand side matrix B.
          On exit, the solution matrix X.
       ldb: int. ldb >= n.
          The leading dimension of B.
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgetrs__retval = hipblasStatus_t(chipblas.hipblasSgetrs(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgetrs__retval,)


@cython.embedsignature(True)
def hipblasDgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgetrs__retval = hipblasStatus_t(chipblas.hipblasDgetrs(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgetrs__retval,)


@cython.embedsignature(True)
def hipblasCgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgetrs__retval = hipblasStatus_t(chipblas.hipblasCgetrs(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        hipblasComplex.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgetrs__retval,)


@cython.embedsignature(True)
def hipblasZgetrs(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgetrs__retval = hipblasStatus_t(chipblas.hipblasZgetrs(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgetrs__retval,)


@cython.embedsignature(True)
def hipblasSgetrsBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info, const int batchCount):
    """SOLVER API

    getrsBatched solves a batch of systems of n linear equations on n

    variables in its factorized forms.

    For each instance i in the batch, it solves one of the following systems, depending on the value of trans:

    \f[
        \begin{array}{cl}
        A_i X_i = B_i & \: \text{not transposed,}\\
        A_i^T X_i = B_i & \: \text{transposed, or}\\
        A_i^H X_i = B_i & \: \text{conjugate transposed.}
        \end{array}
    \f]

    Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfBatched "getrfBatched".

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z

     \ref hipblasSgetrfBatched "getrfBatched".

     \ref hipblasSgetrfBatched "getrfBatched".

    ******************************************************************
    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations of each instance in the batch.
       n: int. n >= 0.
          The order of the system, i.e. the number of columns and rows of all A_i matrices.
       nrhs: int. nrhs >= 0.
          The number of right hand sides, i.e., the number of columns
          of all the matrices B_i.
       A: Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.
          The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by
       lda: int. lda >= n.
          The leading dimension of matrices A_i.
       ipiv: pointer to int. Array on the GPU.
          Contains the vectors ipiv_i of pivot indices returned by
       B: Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.
          On entry, the right hand side matrices B_i.
          On exit, the solution matrix X_i of each system in the batch.
       ldb: int. ldb >= n.
          The leading dimension of matrices B_i.
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
       batchCount: int. batchCount >= 0.
          Number of instances (systems) in the batch.
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgetrsBatched__retval = hipblasStatus_t(chipblas.hipblasSgetrsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgetrsBatched__retval,)


@cython.embedsignature(True)
def hipblasDgetrsBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgetrsBatched__retval = hipblasStatus_t(chipblas.hipblasDgetrsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgetrsBatched__retval,)


@cython.embedsignature(True)
def hipblasCgetrsBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgetrsBatched__retval = hipblasStatus_t(chipblas.hipblasCgetrsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgetrsBatched__retval,)


@cython.embedsignature(True)
def hipblasZgetrsBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, object ipiv, object B, const int ldb, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgetrsBatched__retval = hipblasStatus_t(chipblas.hipblasZgetrsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgetrsBatched__retval,)


@cython.embedsignature(True)
def hipblasSgetrsStridedBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, const long strideA, object ipiv, const long strideP, object B, const int ldb, const long strideB, object info, const int batchCount):
    """SOLVER API

    getrsStridedBatched solves a batch of systems of n linear equations
    on n variables in its factorized forms.

    For each instance i in the batch, it solves one of the following systems, depending on the value of trans:

    \f[
        \begin{array}{cl}
        A_i X_i = B_i & \: \text{not transposed,}\\
        A_i^T X_i = B_i & \: \text{transposed, or}\\
        A_i^H X_i = B_i & \: \text{conjugate transposed.}
        \end{array}
    \f]

    Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : No support

     \ref hipblasSgetrfStridedBatched "getrfStridedBatched".

     \ref hipblasSgetrfStridedBatched "getrfStridedBatched".

    ******************************************************************
    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations of each instance in the batch.
       n: int. n >= 0.
          The order of the system, i.e. the number of columns and rows of all A_i matrices.
       nrhs: int. nrhs >= 0.
          The number of right hand sides, i.e., the number of columns
          of all the matrices B_i.
       A: pointer to type. Array on the GPU (the size depends on the value of strideA).
          The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by
       lda: int. lda >= n.
          The leading dimension of matrices A_i.
       strideA: hipblasStride.
          Stride from the start of one matrix A_i to the next one A_(i+1).
          There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
       ipiv: pointer to int. Array on the GPU (the size depends on the value of strideP).
          Contains the vectors ipiv_i of pivot indices returned by
       strideP: hipblasStride.
          Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
          There is no restriction for the value of strideP. Normal use case is strideP >= n.
       B: pointer to type. Array on the GPU (size depends on the value of strideB).
          On entry, the right hand side matrices B_i.
          On exit, the solution matrix X_i of each system in the batch.
       ldb: int. ldb >= n.
          The leading dimension of matrices B_i.
       strideB: hipblasStride.
          Stride from the start of one matrix B_i to the next one B_(i+1).
          There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
       batchCount: int. batchCount >= 0.
          Number of instances (systems) in the batch.
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgetrsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgetrsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <float *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgetrsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgetrsStridedBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, const long strideA, object ipiv, const long strideP, object B, const int ldb, const long strideB, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgetrsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgetrsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        <double *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgetrsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgetrsStridedBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, const long strideA, object ipiv, const long strideP, object B, const int ldb, const long strideB, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgetrsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgetrsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,strideA,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        hipblasComplex.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgetrsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgetrsStridedBatched(object handle, object trans, const int n, const int nrhs, object A, const int lda, const long strideA, object ipiv, const long strideP, object B, const int ldb, const long strideB, object info, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgetrsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgetrsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,strideA,
        <const int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,strideP,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgetrsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgetriBatched(object handle, const int n, object A, const int lda, object ipiv, object C, const int ldc, object info, const int batchCount):
    """SOLVER API

    getriBatched computes the inverse \f$C_i = A_i^{-1}\f$ of a batch of general n-by-n matrices \f$A_i\f$.

    The inverse is computed by solving the linear system

    \f[
        A_i C_i = I
    \f]

    where I is the identity matrix, and \f$A_i\f$ is factorized as \f$A_i = P_i  L_i  U_i\f$ as given by \ref hipblasSgetrfBatched "getrfBatched".

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z

     \ref hipblasSgetrfBatched "getrfBatched".

     \ref hipblasSgetrfBatched "getrfBatched".
              ipiv can be passed in as a nullptr, this will assume that getrfBatched was called without partial pivoting.

    Args:
       handle: hipblasHandle_t.
       n: int. n >= 0.
          The number of rows and columns of all matrices A_i in the batch.
       A: array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.
          The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by
       lda: int. lda >= n.
          Specifies the leading dimension of matrices A_i.
       ipiv: pointer to int. Array on the GPU (the size depends on the value of strideP).
          The pivot indices returned by
       C: array of pointers to type. Each pointer points to an array on the GPU of dimension ldc*n.
          If info[i] = 0, the inverse of matrices A_i. Otherwise, undefined.
       ldc: int. ldc >= n.
          Specifies the leading dimension of C_i.
       info: pointer to int. Array of batchCount integers on the GPU.
          If info[i] = 0, successful exit for inversion of A_i.
          If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    _hipblasSgetriBatched__retval = hipblasStatus_t(chipblas.hipblasSgetriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(C)._ptr,ldc,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgetriBatched__retval,)


@cython.embedsignature(True)
def hipblasDgetriBatched(object handle, const int n, object A, const int lda, object ipiv, object C, const int ldc, object info, const int batchCount):
    """(No brief)
    """
    _hipblasDgetriBatched__retval = hipblasStatus_t(chipblas.hipblasDgetriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(C)._ptr,ldc,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgetriBatched__retval,)


@cython.embedsignature(True)
def hipblasCgetriBatched(object handle, const int n, object A, const int lda, object ipiv, object C, const int ldc, object info, const int batchCount):
    """(No brief)
    """
    _hipblasCgetriBatched__retval = hipblasStatus_t(chipblas.hipblasCgetriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(C)._ptr,ldc,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgetriBatched__retval,)


@cython.embedsignature(True)
def hipblasZgetriBatched(object handle, const int n, object A, const int lda, object ipiv, object C, const int ldc, object info, const int batchCount):
    """(No brief)
    """
    _hipblasZgetriBatched__retval = hipblasStatus_t(chipblas.hipblasZgetriBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.ListOfInt.from_pyobj(ipiv)._ptr,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(C)._ptr,ldc,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgetriBatched__retval,)


@cython.embedsignature(True)
def hipblasSgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n

    matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
    factorization computed by "GELQF").

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

    @param[inout]
    A           pointer to type. Array on the GPU of dimension lda*n.

                On entry, the matrix A.
                On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").

    @param[inout]
    B           pointer to type. Array on the GPU of dimension ldb*nrhs.

                On entry, the matrix B.
                On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.

    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations.
       m: int. m >= 0.
          The number of rows of matrix A.
       n: int. n >= 0.
          The number of columns of matrix A.
       nrhs: int. nrhs >= 0.
          The number of columns of matrices B and X;
          i.e., the columns on the right hand side.
       lda: int. lda >= m.
          Specifies the leading dimension of matrix A.
       ldb: int. ldb >= max(m,n).
          Specifies the leading dimension of matrix B.
       info: pointer to an int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
       deviceInfo: pointer to int on the GPU.
          If info = 0, successful exit.
          If info = i > 0, the solution could not be computed because input matrix A is
          rank deficient; the i-th diagonal element of its triangular factor is zero.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgels__retval = hipblasStatus_t(chipblas.hipblasSgels(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasSgels__retval,)


@cython.embedsignature(True)
def hipblasDgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgels__retval = hipblasStatus_t(chipblas.hipblasDgels(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasDgels__retval,)


@cython.embedsignature(True)
def hipblasCgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgels__retval = hipblasStatus_t(chipblas.hipblasCgels(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        hipblasComplex.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasCgels__retval,)


@cython.embedsignature(True)
def hipblasZgels(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgels__retval = hipblasStatus_t(chipblas.hipblasZgels(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr))    # fully specified
    return (_hipblasZgels__retval,)


@cython.embedsignature(True)
def hipblasSgelsBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo, const int batchCount):
    """gelsBatched solves a batch of overdetermined (or underdetermined) linear systems

    defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$, using the
    QR factorizations computed by "GEQRF_BATCHED" (or the LQ factorizations computed by "GELQF_BATCHED").

    For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed, or}\\
        A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
        \end{array}
    \f]

    If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X_j is found by minimizing

    \f[
        || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
    \f]

    If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z
    Note that cuBLAS backend supports only the non-transpose operation and only solves over-determined systems (m >= n).

    @param[inout]
    A           array of pointer to type. Each pointer points to an array on the GPU of dimension lda*n.

                On entry, the matrices A_j.
                On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_BATCHED"
                (or "GELQF_BATCHED").

    @param[inout]
    B           array of pointer to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.

                On entry, the matrices B_j.
                On exit, when info[j] = 0, B_j is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.

    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations.
       m: int. m >= 0.
          The number of rows of all matrices A_j in the batch.
       n: int. n >= 0.
          The number of columns of all matrices A_j in the batch.
       nrhs: int. nrhs >= 0.
          The number of columns of all matrices B_j and X_j in the batch;
          i.e., the columns on the right hand side.
       lda: int. lda >= m.
          Specifies the leading dimension of matrices A_j.
       ldb: int. ldb >= max(m,n).
          Specifies the leading dimension of matrices B_j.
       info: pointer to an int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
       deviceInfo: pointer to int. Array of batchCount integers on the GPU.
          If deviceInfo[j] = 0, successful exit for solution of A_j.
          If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
          matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgelsBatched__retval = hipblasStatus_t(chipblas.hipblasSgelsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batchCount))    # fully specified
    return (_hipblasSgelsBatched__retval,)


@cython.embedsignature(True)
def hipblasDgelsBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgelsBatched__retval = hipblasStatus_t(chipblas.hipblasDgelsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batchCount))    # fully specified
    return (_hipblasDgelsBatched__retval,)


@cython.embedsignature(True)
def hipblasCgelsBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgelsBatched__retval = hipblasStatus_t(chipblas.hipblasCgelsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batchCount))    # fully specified
    return (_hipblasCgelsBatched__retval,)


@cython.embedsignature(True)
def hipblasZgelsBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, object B, const int ldb, object info, object deviceInfo, const int batchCount):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgelsBatched__retval = hipblasStatus_t(chipblas.hipblasZgelsBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batchCount))    # fully specified
    return (_hipblasZgelsBatched__retval,)


@cython.embedsignature(True)
def hipblasSgelsStridedBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, const long strideA, object B, const int ldb, const long strideB, object info, object deviceInfo, const int batch_count):
    """gelsStridedBatched solves a batch of overdetermined (or underdetermined) linear

    systems defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$,
    using the QR factorizations computed by "GEQRF_STRIDED_BATCHED"
    (or the LQ factorizations computed by "GELQF_STRIDED_BATCHED").

    For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form

    \f[
        \begin{array}{cl}
        A_j X_j = B_j & \: \text{not transposed, or}\\
        A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
        \end{array}
    \f]

    If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
    and a least-squares solution approximating X_j is found by minimizing

    \f[
        || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
    \f]

    If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
    and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : currently unsupported

    @param[inout]
    A           pointer to type. Array on the GPU (the size depends on the value of strideA).

                On entry, the matrices A_j.
                On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_STRIDED_BATCHED"
                (or "GELQF_STRIDED_BATCHED").

    @param[inout]
    B           pointer to type. Array on the GPU (the size depends on the value of strideB).

                On entry, the matrices B_j.
                On exit, when info[j] = 0, each B_j is overwritten by the solution vectors (and the residuals in
                the overdetermined cases) stored as columns.

    Args:
       handle: hipblasHandle_t.
       trans: hipblasOperation_t.
          Specifies the form of the system of equations.
       m: int. m >= 0.
          The number of rows of all matrices A_j in the batch.
       n: int. n >= 0.
          The number of columns of all matrices A_j in the batch.
       nrhs: int. nrhs >= 0.
          The number of columns of all matrices B_j and X_j in the batch;
          i.e., the columns on the right hand side.
       lda: int. lda >= m.
          Specifies the leading dimension of matrices A_j.
       strideA: hipblasStride.
          Stride from the start of one matrix A_j to the next one A_(j+1).
          There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
       ldb: int. ldb >= max(m,n).
          Specifies the leading dimension of matrices B_j.
       strideB: hipblasStride.
          Stride from the start of one matrix B_j to the next one B_(j+1).
          There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs
       info: pointer to an int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
       deviceInfo: pointer to int. Array of batchCount integers on the GPU.
          If deviceInfo[j] = 0, successful exit for solution of A_j.
          If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
          matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasSgelsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgelsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batch_count))    # fully specified
    return (_hipblasSgelsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgelsStridedBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, const long strideA, object B, const int ldb, const long strideB, object info, object deviceInfo, const int batch_count):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasDgelsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgelsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batch_count))    # fully specified
    return (_hipblasDgelsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgelsStridedBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, const long strideA, object B, const int ldb, const long strideB, object info, object deviceInfo, const int batch_count):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasCgelsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgelsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasComplex.from_pyobj(A)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batch_count))    # fully specified
    return (_hipblasCgelsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgelsStridedBatched(object handle, object trans, const int m, const int n, const int nrhs, object A, const int lda, const long strideA, object B, const int ldb, const long strideB, object info, object deviceInfo, const int batch_count):
    """(No brief)
    """
    if not isinstance(trans,_hipblasOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipblasOperation_t__Base'")
    _hipblasZgelsStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgelsStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,trans.value,m,n,nrhs,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(B)._ptr,ldb,strideB,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(deviceInfo)._ptr,batch_count))    # fully specified
    return (_hipblasZgelsStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasSgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """SOLVER API

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

    @param[inout]
    A         pointer to type. Array on the GPU of dimension lda*n.

              On entry, the m-by-n matrix to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R; the elements below the diagonal are the last m - i elements
              of Householder vector v_i.

    Args:
       handle: hipblasHandle_t.
       m: int. m >= 0.
          The number of rows of the matrix A.
       n: int. n >= 0.
          The number of columns of the matrix A.
       lda: int. lda >= m.
          Specifies the leading dimension of A.
       ipiv: pointer to type. Array on the GPU of dimension min(m,n).
          The Householder scalars.
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = j < 0, the j-th argument is invalid.
          ******************************************************************
    """
    _hipblasSgeqrf__retval = hipblasStatus_t(chipblas.hipblasSgeqrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <float *>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasSgeqrf__retval,)


@cython.embedsignature(True)
def hipblasDgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasDgeqrf__retval = hipblasStatus_t(chipblas.hipblasDgeqrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <double *>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasDgeqrf__retval,)


@cython.embedsignature(True)
def hipblasCgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasCgeqrf__retval = hipblasStatus_t(chipblas.hipblasCgeqrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,
        hipblasComplex.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasCgeqrf__retval,)


@cython.embedsignature(True)
def hipblasZgeqrf(object handle, const int m, const int n, object A, const int lda, object ipiv, object info):
    """(No brief)
    """
    _hipblasZgeqrf__retval = hipblasStatus_t(chipblas.hipblasZgeqrf(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,
        hipblasDoubleComplex.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr))    # fully specified
    return (_hipblasZgeqrf__retval,)


@cython.embedsignature(True)
def hipblasSgeqrfBatched(object handle, const int m, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """SOLVER API

    geqrfBatched computes the QR factorization of a batch of general
    m-by-n matrices.

    The factorization of matrix \f$A_i\f$ in the batch has the form

    \f[
        A_i = Q_i\left[\begin{array}{c}
        R_i\\
        0
        \end{array}\right]
    \f]

    where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{i_j}\f$ is given by

    \f[
        H_{i_j} = I - \text{ipiv}_i[j] \cdot v_{i_j} v_{i_j}'
    \f]

    where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : s,d,c,z

    @param[inout]
    A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.

              On entry, the m-by-n matrices A_i to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_i. The elements below the diagonal are the last m - j elements
              of Householder vector v_(i_j).

    Args:
       handle: hipblasHandle_t.
       m: int. m >= 0.
          The number of rows of all the matrices A_i in the batch.
       n: int. n >= 0.
          The number of columns of all the matrices A_i in the batch.
       lda: int. lda >= m.
          Specifies the leading dimension of matrices A_i.
       ipiv: array of pointers to type. Each pointer points to an array on the GPU
          of dimension min(m, n).
          Contains the vectors ipiv_i of corresponding Householder scalars.
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = k < 0, the k-th argument is invalid.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    _hipblasSgeqrfBatched__retval = hipblasStatus_t(chipblas.hipblasSgeqrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <float *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <float *const*>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgeqrfBatched__retval,)


@cython.embedsignature(True)
def hipblasDgeqrfBatched(object handle, const int m, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasDgeqrfBatched__retval = hipblasStatus_t(chipblas.hipblasDgeqrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <double *const*>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,lda,
        <double *const*>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgeqrfBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeqrfBatched(object handle, const int m, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasCgeqrfBatched__retval = hipblasStatus_t(chipblas.hipblasCgeqrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <chipblas.hipblasComplex *const*>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgeqrfBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeqrfBatched(object handle, const int m, const int n, object A, const int lda, object ipiv, object info, const int batchCount):
    """(No brief)
    """
    _hipblasZgeqrfBatched__retval = hipblasStatus_t(chipblas.hipblasZgeqrfBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <chipblas.hipblasDoubleComplex *const*>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgeqrfBatched__retval,)


@cython.embedsignature(True)
def hipblasSgeqrfStridedBatched(object handle, const int m, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """SOLVER API

    geqrfStridedBatched computes the QR factorization of a batch of
    general m-by-n matrices.

    The factorization of matrix \f$A_i\f$ in the batch has the form

    \f[
        A_i = Q_i\left[\begin{array}{c}
        R_i\\
        0
        \end{array}\right]
    \f]

    where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
    a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices

    \f[
        Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
    \f]

    Each Householder matrix \f$H_{i_j}\f$ is given by

    \f[
        H_{i_j} = I - \text{ipiv}_j[j] \cdot v_{i_j} v_{i_j}'
    \f]

    where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.

    - Supported precisions in rocSOLVER : s,d,c,z
    - Supported precisions in cuBLAS    : No support

    @param[inout]
    A         pointer to type. Array on the GPU (the size depends on the value of strideA).

              On entry, the m-by-n matrices A_i to be factored.
              On exit, the elements on and above the diagonal contain the
              factor R_i. The elements below the diagonal are the last m - j elements
              of Householder vector v_(i_j).

    Args:
       handle: hipblasHandle_t.
       m: int. m >= 0.
          The number of rows of all the matrices A_i in the batch.
       n: int. n >= 0.
          The number of columns of all the matrices A_i in the batch.
       lda: int. lda >= m.
          Specifies the leading dimension of matrices A_i.
       strideA: hipblasStride.
          Stride from the start of one matrix A_i to the next one A_(i+1).
          There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
       ipiv: pointer to type. Array on the GPU (the size depends on the value of strideP).
          Contains the vectors ipiv_i of corresponding Householder scalars.
       strideP: hipblasStride.
          Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
          There is no restriction for the value
          of strideP. Normal use is strideP >= min(m,n).
       info: pointer to a int on the host.
          If info = 0, successful exit.
          If info = k < 0, the k-th argument is invalid.
       batchCount: int. batchCount >= 0.
          Number of matrices in the batch.
          ******************************************************************
    """
    _hipblasSgeqrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasSgeqrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <float *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <float *>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasSgeqrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasDgeqrfStridedBatched(object handle, const int m, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasDgeqrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasDgeqrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        <double *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <double *>hip._util.types.DataHandle.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasDgeqrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasCgeqrfStridedBatched(object handle, const int m, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasCgeqrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasCgeqrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasComplex.from_pyobj(A)._ptr,lda,strideA,
        hipblasComplex.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasCgeqrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasZgeqrfStridedBatched(object handle, const int m, const int n, object A, const int lda, const long strideA, object ipiv, const long strideP, object info, const int batchCount):
    """(No brief)
    """
    _hipblasZgeqrfStridedBatched__retval = hipblasStatus_t(chipblas.hipblasZgeqrfStridedBatched(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,m,n,
        hipblasDoubleComplex.from_pyobj(A)._ptr,lda,strideA,
        hipblasDoubleComplex.from_pyobj(ipiv)._ptr,strideP,
        <int *>hip._util.types.ListOfInt.from_pyobj(info)._ptr,batchCount))    # fully specified
    return (_hipblasZgeqrfStridedBatched__retval,)


@cython.embedsignature(True)
def hipblasGemmEx(object handle, object transA, object transB, int m, int n, int k, object alpha, object A, object aType, int lda, object B, object bType, int ldb, object beta, object C, object cType, int ldc, object computeType, object algo):
    """BLAS EX API

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A ).
       transB: [hipblasOperation_t]
          specifies the form of op( B ).
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       k: [int]
          matrix dimension k.
       alpha: [const void *]
          device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
       A: [void *]
          device pointer storing matrix A.
       aType: [hipblasDatatype_t]
          specifies the datatype of matrix A.
       lda: [int]
          specifies the leading dimension of A.
       B: [void *]
          device pointer storing matrix B.
       bType: [hipblasDatatype_t]
          specifies the datatype of matrix B.
       ldb: [int]
          specifies the leading dimension of B.
       beta: [const void *]
          device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
       C: [void *]
          device pointer storing matrix C.
       cType: [hipblasDatatype_t]
          specifies the datatype of matrix C.
       ldc: [int]
          specifies the leading dimension of C.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation.
       algo: [hipblasGemmAlgo_t]
          enumerant specifying the algorithm type.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(aType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'aType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(bType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'bType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(cType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'cType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(algo,_hipblasGemmAlgo_t__Base):
        raise TypeError("argument 'algo' must be of type '_hipblasGemmAlgo_t__Base'")
    _hipblasGemmEx__retval = hipblasStatus_t(chipblas.hipblasGemmEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const void *>hip._util.types.DataHandle.from_pyobj(A)._ptr,aType.value,lda,
        <const void *>hip._util.types.DataHandle.from_pyobj(B)._ptr,bType.value,ldb,
        <const void *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(C)._ptr,cType.value,ldc,computeType.value,algo.value))    # fully specified
    return (_hipblasGemmEx__retval,)


@cython.embedsignature(True)
def hipblasGemmBatchedEx(object handle, object transA, object transB, int m, int n, int k, object alpha, object A, object aType, int lda, object B, object bType, int ldb, object beta, object C, object cType, int ldc, int batchCount, object computeType, object algo):
    """BLAS EX API

    gemmBatchedEx performs one of the batched matrix-matrix operations
        C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount.
    where op( X ) is one of
        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,
    alpha and beta are scalars, and A, B, and C are batched pointers to matrices, with
    op( A ) an m by k by batchCount batched matrix,
    op( B ) a k by n by batchCount batched matrix and
    C a m by n by batchCount batched matrix.
    The batched matrices are an array of pointers to matrices.
    The number of pointers to matrices is batchCount.

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    Note for int8 users - For rocBLAS backend, please read rocblas_gemm_batched_ex documentation on int8
    data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
    format for a given device as documented in rocBLAS.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A ).
       transB: [hipblasOperation_t]
          specifies the form of op( B ).
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       k: [int]
          matrix dimension k.
       alpha: [const void *]
          device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
       A: [void *]
          device pointer storing array of pointers to each matrix A_i.
       aType: [hipblasDatatype_t]
          specifies the datatype of each matrix A_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       B: [void *]
          device pointer storing array of pointers to each matrix B_i.
       bType: [hipblasDatatype_t]
          specifies the datatype of each matrix B_i.
       ldb: [int]
          specifies the leading dimension of each B_i.
       beta: [const void *]
          device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
       C: [void *]
          device array of device pointers to each matrix C_i.
       cType: [hipblasDatatype_t]
          specifies the datatype of each matrix C_i.
       ldc: [int]
          specifies the leading dimension of each C_i.
       batchCount: [int]
          number of gemm operations in the batch.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation.
       algo: [hipblasGemmAlgo_t]
          enumerant specifying the algorithm type.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(aType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'aType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(bType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'bType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(cType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'cType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(algo,_hipblasGemmAlgo_t__Base):
        raise TypeError("argument 'algo' must be of type '_hipblasGemmAlgo_t__Base'")
    _hipblasGemmBatchedEx__retval = hipblasStatus_t(chipblas.hipblasGemmBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const void **>hip._util.types.ListOfDataHandle.from_pyobj(A)._ptr,aType.value,lda,
        <const void **>hip._util.types.ListOfDataHandle.from_pyobj(B)._ptr,bType.value,ldb,
        <const void *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <void **>hip._util.types.ListOfDataHandle.from_pyobj(C)._ptr,cType.value,ldc,batchCount,computeType.value,algo.value))    # fully specified
    return (_hipblasGemmBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasGemmStridedBatchedEx(object handle, object transA, object transB, int m, int n, int k, object alpha, object A, object aType, int lda, long strideA, object B, object bType, int ldb, long strideB, object beta, object C, object cType, int ldc, long strideC, int batchCount, object computeType, object algo):
    """BLAS EX API

    gemmStridedBatchedEx performs one of the strided_batched matrix-matrix operations

        C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, and C are strided_batched matrices, with
    op( A ) an m by k by batchCount strided_batched matrix,
    op( B ) a k by n by batchCount strided_batched matrix and
    C a m by n by batchCount strided_batched matrix.

    The strided_batched matrices are multiple matrices separated by a constant stride.
    The number of matrices is batchCount.

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    Note for int8 users - For rocBLAS backend, please read rocblas_gemm_strided_batched_ex documentation on int8
    data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
    format for a given device as documented in rocBLAS.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       transA: [hipblasOperation_t]
          specifies the form of op( A ).
       transB: [hipblasOperation_t]
          specifies the form of op( B ).
       m: [int]
          matrix dimension m.
       n: [int]
          matrix dimension n.
       k: [int]
          matrix dimension k.
       alpha: [const void *]
          device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
       A: [void *]
          device pointer pointing to first matrix A_1.
       aType: [hipblasDatatype_t]
          specifies the datatype of each matrix A_i.
       lda: [int]
          specifies the leading dimension of each A_i.
       strideA: [hipblasStride]
          specifies stride from start of one A_i matrix to the next A_(i + 1).
       B: [void *]
          device pointer pointing to first matrix B_1.
       bType: [hipblasDatatype_t]
          specifies the datatype of each matrix B_i.
       ldb: [int]
          specifies the leading dimension of each B_i.
       strideB: [hipblasStride]
          specifies stride from start of one B_i matrix to the next B_(i + 1).
       beta: [const void *]
          device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
       C: [void *]
          device pointer pointing to first matrix C_1.
       cType: [hipblasDatatype_t]
          specifies the datatype of each matrix C_i.
       ldc: [int]
          specifies the leading dimension of each C_i.
       strideC: [hipblasStride]
          specifies stride from start of one C_i matrix to the next C_(i + 1).
       batchCount: [int]
          number of gemm operations in the batch.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation.
       algo: [hipblasGemmAlgo_t]
          enumerant specifying the algorithm type.
          ******************************************************************
    """
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(transB,_hipblasOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(aType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'aType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(bType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'bType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(cType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'cType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(algo,_hipblasGemmAlgo_t__Base):
        raise TypeError("argument 'algo' must be of type '_hipblasGemmAlgo_t__Base'")
    _hipblasGemmStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasGemmStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <const void *>hip._util.types.DataHandle.from_pyobj(A)._ptr,aType.value,lda,strideA,
        <const void *>hip._util.types.DataHandle.from_pyobj(B)._ptr,bType.value,ldb,strideB,
        <const void *>hip._util.types.DataHandle.from_pyobj(beta)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(C)._ptr,cType.value,ldc,strideC,batchCount,computeType.value,algo.value))    # fully specified
    return (_hipblasGemmStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasTrsmEx(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object A, int lda, object B, int ldb, object invA, int invAsize, object computeType):
    """(No brief)

    BLAS EX API

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

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_ON_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of B. m >= 0.
       n: [int]
          n specifies the number of columns of B. n >= 0.
       alpha: [void *]
          device pointer or host pointer specifying the scalar alpha. When alpha is
          &zero then A is not referenced, and B need not be set before
          entry.
       A: [void *]
          device pointer storing matrix A.
          of dimension ( lda, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       B: [void *]
          device pointer storing matrix B.
          B is of dimension ( ldb, n ).
          Before entry, the leading m by n part of the array B must
          contain the right-hand side matrix B, and on exit is
          overwritten by the solution matrix X.
       ldb: [int]
          ldb specifies the first dimension of B. ldb >= max( 1, m ).
       invA: [void *]
          device pointer storing the inverse diagonal blocks of A.
          invA is of dimension ( ld_invA, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT.
          ld_invA must be equal to 128.
       invAsize: [int]
          invAsize specifies the number of elements of device memory in invA.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasTrsmEx__retval = hipblasStatus_t(chipblas.hipblasTrsmEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,
        <const void *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,invAsize,computeType.value))    # fully specified
    return (_hipblasTrsmEx__retval,)


@cython.embedsignature(True)
def hipblasTrsmBatchedEx(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object A, int lda, object B, int ldb, int batchCount, object invA, int invAsize, object computeType):
    """(No brief)

    BLAS EX API

    trsmBatchedEx solves

        op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,

    for i = 1, ..., batchCount; and where alpha is a scalar, X and B are arrays of m by n matrices,
    A is an array of triangular matrix and each op(A_i) is one of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    Each matrix X_i is overwritten on B_i.

    This function gives the user the ability to reuse the invA matrix between runs.
    If invA == NULL, hipblasTrsmBatchedEx will automatically calculate each invA_i on every run.

    Setting up invA:
    Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A_i, followed by any smaller diagonal block that remains.
    To set up each invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
    invA is an array of pointers of batchCount length holding each invA_i.

    Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
    HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
    should be passed as invAsize.

    To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
    matrix A_i. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batchCount = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previousBatchCount
      - ldinvA = 128
      - batchCount = 1

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each B_i. m >= 0.
       n: [int]
          n specifies the number of columns of each B_i. n >= 0.
       alpha: [void *]
          device pointer or host pointer alpha specifying the scalar alpha. When alpha is
          &zero then A is not referenced, and B need not be set before
          entry.
       A: [void *]
          device array of device pointers storing each matrix A_i.
          each A_i is of dimension ( lda, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of each A_i.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       B: [void *]
          device array of device pointers storing each matrix B_i.
          each B_i is of dimension ( ldb, n ).
          Before entry, the leading m by n part of the array B_i must
          contain the right-hand side matrix B_i, and on exit is
          overwritten by the solution matrix X_i
       ldb: [int]
          ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
       batchCount: [int]
          specifies how many batches.
       invA: [void *]
          device array of device pointers storing the inverse diagonal blocks of each A_i.
          each invA_i is of dimension ( ld_invA, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT.
          ld_invA must be equal to 128.
       invAsize: [int]
          invAsize specifies the number of elements of device memory in each invA_i.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasTrsmBatchedEx__retval = hipblasStatus_t(chipblas.hipblasTrsmBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,
        <void *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,batchCount,
        <const void *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,invAsize,computeType.value))    # fully specified
    return (_hipblasTrsmBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasTrsmStridedBatchedEx(object handle, object side, object uplo, object transA, object diag, int m, int n, object alpha, object A, int lda, long strideA, object B, int ldb, long strideB, int batchCount, object invA, int invAsize, long strideInvA, object computeType):
    """(No brief)

    BLAS EX API

    trsmStridedBatchedEx solves

        op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,

    for i = 1, ..., batchCount; and where alpha is a scalar, X and B are strided batched m by n matrices,
    A is a strided batched triangular matrix and op(A_i) is one of

        op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.

    Each matrix X_i is overwritten on B_i.

    This function gives the user the ability to reuse each invA_i matrix between runs.
    If invA == NULL, hipblasTrsmStridedBatchedEx will automatically calculate each invA_i on every run.

    Setting up invA:
    Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
    matrix A_i, followed by any smaller diagonal block that remains.
    To set up invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
    invA is a contiguous piece of memory holding each invA_i.

    Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
    HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
    should be passed as invAsize.

    To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
    matrix A_i. Below are the restricted parameters:
      - n = 128
      - ldinvA = 128
      - stride_invA = 128x128
      - batchCount = k / 128,

    Then any remaining block may be added:
      - n = k % 128
      - invA = invA + stride_invA * previousBatchCount
      - ldinvA = 128
      - batchCount = 1

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       side: [hipblasSideMode_t]
          HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
          HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
       uplo: [hipblasFillMode_t]
          HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
          HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
       transA: [hipblasOperation_t]
          HIPBLAS_OP_N: op(A) = A.
          HIPBLAS_OP_T: op(A) = A^T.
          HIPBLAS_OP_C: op(A) = A^H.
       diag: [hipblasDiagType_t]
          HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
          HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
       m: [int]
          m specifies the number of rows of each B_i. m >= 0.
       n: [int]
          n specifies the number of columns of each B_i. n >= 0.
       alpha: [void *]
          device pointer or host pointer specifying the scalar alpha. When alpha is
          &zero then A is not referenced, and B need not be set before
          entry.
       A: [void *]
          device pointer storing matrix A.
          of dimension ( lda, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT
          only the upper/lower triangular part is accessed.
       lda: [int]
          lda specifies the first dimension of A.
          if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
          if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
       strideA: [hipblasStride]
          The stride between each A matrix.
       B: [void *]
          device pointer pointing to first matrix B_i.
          each B_i is of dimension ( ldb, n ).
          Before entry, the leading m by n part of each array B_i must
          contain the right-hand side of matrix B_i, and on exit is
          overwritten by the solution matrix X_i.
       ldb: [int]
          ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
       strideB: [hipblasStride]
          The stride between each B_i matrix.
       batchCount: [int]
          specifies how many batches.
       invA: [void *]
          device pointer storing the inverse diagonal blocks of each A_i.
          invA points to the first invA_1.
          each invA_i is of dimension ( ld_invA, k ), where k is m
          when HIPBLAS_SIDE_LEFT and
          is n when HIPBLAS_SIDE_RIGHT.
          ld_invA must be equal to 128.
       invAsize: [int]
          invAsize specifies the number of elements of device memory in each invA_i.
       strideInvA: [hipblasStride]
          The stride between each invA matrix.
       computeType: [hipblasDatatype_t]
          specifies the datatype of computation
          ******************************************************************
    """
    if not isinstance(side,_hipblasSideMode_t__Base):
        raise TypeError("argument 'side' must be of type '_hipblasSideMode_t__Base'")                    
    if not isinstance(uplo,_hipblasFillMode_t__Base):
        raise TypeError("argument 'uplo' must be of type '_hipblasFillMode_t__Base'")                    
    if not isinstance(transA,_hipblasOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipblasOperation_t__Base'")                    
    if not isinstance(diag,_hipblasDiagType_t__Base):
        raise TypeError("argument 'diag' must be of type '_hipblasDiagType_t__Base'")                    
    if not isinstance(computeType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'computeType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasTrsmStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasTrsmStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,side.value,uplo.value,transA.value,diag.value,m,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(A)._ptr,lda,strideA,
        <void *>hip._util.types.DataHandle.from_pyobj(B)._ptr,ldb,strideB,batchCount,
        <const void *>hip._util.types.DataHandle.from_pyobj(invA)._ptr,invAsize,strideInvA,computeType.value))    # fully specified
    return (_hipblasTrsmStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasAxpyEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, object y, object yType, int incy, object executionType):
    """BLAS EX API

    axpyEx computes constant alpha multiplied by vector x, plus vector y

        y := alpha * x + y

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    y         device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       alpha: device pointer or host pointer to specify the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       x: device pointer storing vector x.
       xType: [hipblasDatatype_t]
          specifies the datatype of vector x.
       incx: [int]
          specifies the increment for the elements of x.
       yType: [hipblasDatatype_t]
          specifies the datatype of vector y.
       incy: [int]
          specifies the increment for the elements of y.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasAxpyEx__retval = hipblasStatus_t(chipblas.hipblasAxpyEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,executionType.value))    # fully specified
    return (_hipblasAxpyEx__retval,)


@cython.embedsignature(True)
def hipblasAxpyBatchedEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, object y, object yType, int incy, int batchCount, object executionType):
    """BLAS EX API

    axpyBatchedEx computes constant alpha multiplied by vector x, plus vector y over
                      a set of batched vectors.

        y := alpha * x + y

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    y         device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       alpha: device pointer or host pointer to specify the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       x: device array of device pointers storing each vector x_i.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasAxpyBatchedEx__retval = hipblasStatus_t(chipblas.hipblasAxpyBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,batchCount,executionType.value))    # fully specified
    return (_hipblasAxpyBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasAxpyStridedBatchedEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, long stridex, object y, object yType, int incy, long stridey, int batchCount, object executionType):
    """BLAS EX API

    axpyStridedBatchedEx computes constant alpha multiplied by vector x, plus vector y over
                      a set of strided batched vectors.

        y := alpha * x + y

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    y         device pointer to the first vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       alpha: device pointer or host pointer to specify the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       x: device pointer to the first vector x_1.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) to the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size, for a typical
          case this means stridex >= n * incx.
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) to the next one (y_i+1).
          There are no restrictions placed on stridey, however the user should
          take care to ensure that stridey is of appropriate size, for a typical
          case this means stridey >= n * incy.
       batchCount: [int]
          number of instances in the batch.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasAxpyStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasAxpyStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,stridey,batchCount,executionType.value))    # fully specified
    return (_hipblasAxpyStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasDotEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object result, object resultType, object executionType):
    """BLAS EX API

    dotEx  performs the dot product of vectors x and y

        result = x * y;

    dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y

        result = conjugate (x) * y;

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    result
              device pointer or host pointer to store the dot product.
              return is 0.0 if n <= 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x and y.
       x: device pointer storing vector x.
       xType: [hipblasDatatype_t]
          specifies the datatype of vector x.
       incx: [int]
          specifies the increment for the elements of y.
       y: device pointer storing vector y.
       yType: [hipblasDatatype_t]
          specifies the datatype of vector y.
       incy: [int]
          specifies the increment for the elements of y.
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotEx__retval = hipblasStatus_t(chipblas.hipblasDotEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotEx__retval,)


@cython.embedsignature(True)
def hipblasDotcEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object result, object resultType, object executionType):
    """(No brief)
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotcEx__retval = hipblasStatus_t(chipblas.hipblasDotcEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotcEx__retval,)


@cython.embedsignature(True)
def hipblasDotBatchedEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, int batchCount, object result, object resultType, object executionType):
    """BLAS EX API

    dotBatchedEx performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotcBatchedEx  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batchCount

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    result
              device array or host array of batchCount size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       x: device array of device pointers storing each vector x_i.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       y: device array of device pointers storing each vector y_i.
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       batchCount: [int]
          number of instances in the batch
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotBatchedEx__retval = hipblasStatus_t(chipblas.hipblasDotBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasDotcBatchedEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, int batchCount, object result, object resultType, object executionType):
    """(No brief)
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotcBatchedEx__retval = hipblasStatus_t(chipblas.hipblasDotcBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotcBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasDotStridedBatchedEx(object handle, int n, object x, object xType, int incx, long stridex, object y, object yType, int incy, long stridey, int batchCount, object result, object resultType, object executionType):
    """BLAS EX API

    dotStridedBatchedEx  performs a batch of dot products of vectors x and y

        result_i = x_i * y_i;

    dotc_strided_batched_ex  performs a batch of dot products of the conjugate of complex vector x and complex vector y

        result_i = conjugate (x_i) * y_i;

    where (x_i, y_i) is the i-th instance of the batch.
    x_i and y_i are vectors, for i = 1, ..., batchCount

        - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    result
              device array or host array of batchCount size to store the dot products of each batch.
              return 0.0 for each element if n <= 0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in each x_i and y_i.
       x: device pointer to the first vector (x_1) in the batch.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1)
       y: device pointer to the first vector (y_1) in the batch.
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment for the elements of each y_i.
       stridey: [hipblasStride]
          stride from the start of one vector (y_i) and the next one (y_i+1)
       batchCount: [int]
          number of instances in the batch
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasDotStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,stridey,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasDotcStridedBatchedEx(object handle, int n, object x, object xType, int incx, long stridex, object y, object yType, int incy, long stridey, int batchCount, object result, object resultType, object executionType):
    """(No brief)
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasDotcStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasDotcStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,
        <const void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,stridey,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasDotcStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasNrm2Ex(object handle, int n, object x, object xType, int incx, object result, object resultType, object executionType):
    """BLAS_EX API

    nrm2Ex computes the euclidean norm of a real or complex vector

              result := sqrt( x'*x ) for real vectors
              result := sqrt( x**H*x ) for complex vectors

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    result
              device pointer or host pointer to store the nrm2 product.
              return is 0.0 if n, incx<=0.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       x: device pointer storing vector x.
       xType: [hipblasDatatype_t]
          specifies the datatype of the vector x.
       incx: [int]
          specifies the increment for the elements of y.
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasNrm2Ex__retval = hipblasStatus_t(chipblas.hipblasNrm2Ex(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasNrm2Ex__retval,)


@cython.embedsignature(True)
def hipblasNrm2BatchedEx(object handle, int n, object x, object xType, int incx, int batchCount, object result, object resultType, object executionType):
    """BLAS_EX API

    nrm2BatchedEx computes the euclidean norm over a batch of real or complex vectors

              result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
              result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i.
       x: device array of device pointers storing each vector x_i.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       batchCount: [int]
          number of instances in the batch
       result: device pointer or host pointer to array of batchCount size for nrm2 results.
          return is 0.0 for each element if n <= 0, incx<=0.
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasNrm2BatchedEx__retval = hipblasStatus_t(chipblas.hipblasNrm2BatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasNrm2BatchedEx__retval,)


@cython.embedsignature(True)
def hipblasNrm2StridedBatchedEx(object handle, int n, object x, object xType, int incx, long stridex, int batchCount, object result, object resultType, object executionType):
    """BLAS_EX API

    nrm2StridedBatchedEx computes the euclidean norm over a batch of real or complex vectors

              := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
              := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i.
       x: device pointer to the first vector x_1.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i. incx must be > 0.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) and the next one (x_i+1).
          There are no restrictions placed on stride_x, however the user should
          take care to ensure that stride_x is of appropriate size, for a typical
          case this means stride_x >= n * incx.
       batchCount: [int]
          number of instances in the batch
       result: device pointer or host pointer to array for storing contiguous batchCount results.
          return is 0.0 for each element if n <= 0, incx<=0.
       resultType: [hipblasDatatype_t]
          specifies the datatype of the result.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(resultType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'resultType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasNrm2StridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasNrm2StridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,batchCount,
        <void *>hip._util.types.DataHandle.from_pyobj(result)._ptr,resultType.value,executionType.value))    # fully specified
    return (_hipblasNrm2StridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasRotEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object c, object s, object csType, object executionType):
    """BLAS EX API

    rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    In the case where cs_type is real:
        x := c * x + s * y
            y := c * y - s * x

    In the case where cs_type is complex, the imaginary part of c is ignored:
        x := real(c) * x + s * y
            y := real(c) * y - conj(s) * x

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x       device pointer storing vector x.

    @param[inout]
    y       device pointer storing vector y.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in the x and y vectors.
       xType: [hipblasDatatype_t]
          specifies the datatype of vector x.
       incx: [int]
          specifies the increment between elements of x.
       yType: [hipblasDatatype_t]
          specifies the datatype of vector y.
       incy: [int]
          specifies the increment between elements of y.
       c: device pointer or host pointer storing scalar cosine component of the rotation matrix.
       s: device pointer or host pointer storing scalar sine component of the rotation matrix.
       csType: [hipblasDatatype_t]
          specifies the datatype of c and s.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(csType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'csType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasRotEx__retval = hipblasStatus_t(chipblas.hipblasRotEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <const void *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const void *>hip._util.types.DataHandle.from_pyobj(s)._ptr,csType.value,executionType.value))    # fully specified
    return (_hipblasRotEx__retval,)


@cython.embedsignature(True)
def hipblasRotBatchedEx(object handle, int n, object x, object xType, int incx, object y, object yType, int incy, object c, object s, object csType, int batchCount, object executionType):
    """BLAS EX API

    rotBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    In the case where cs_type is real:
            x := c * x + s * y
            y := c * y - s * x

        In the case where cs_type is complex, the imaginary part of c is ignored:
            x := real(c) * x + s * y
            y := real(c) * y - conj(s) * x

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x       device array of deivce pointers storing each vector x_i.

    @param[inout]
    y       device array of device pointers storing each vector y_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i and y_i vectors.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment between elements of each x_i.
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment between elements of each y_i.
       c: device pointer or host pointer to scalar cosine component of the rotation matrix.
       s: device pointer or host pointer to scalar sine component of the rotation matrix.
       csType: [hipblasDatatype_t]
          specifies the datatype of c and s.
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(csType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'csType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasRotBatchedEx__retval = hipblasStatus_t(chipblas.hipblasRotBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,
        <const void *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const void *>hip._util.types.DataHandle.from_pyobj(s)._ptr,csType.value,batchCount,executionType.value))    # fully specified
    return (_hipblasRotBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasRotStridedBatchedEx(object handle, int n, object x, object xType, int incx, long stridex, object y, object yType, int incy, long stridey, object c, object s, object csType, int batchCount, object executionType):
    """BLAS Level 1 API

    rotStridedBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
        Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.

    In the case where cs_type is real:
            x := c * x + s * y
            y := c * y - s * x

        In the case where cs_type is complex, the imaginary part of c is ignored:
            x := real(c) * x + s * y
            y := real(c) * y - conj(s) * x

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x       device pointer to the first vector x_1.

    @param[inout]
    y       device pointer to the first vector y_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          number of elements in each x_i and y_i vectors.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment between elements of each x_i.
       stridex: [hipblasStride]
          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
       yType: [hipblasDatatype_t]
          specifies the datatype of each vector y_i.
       incy: [int]
          specifies the increment between elements of each y_i.
       stridey: [hipblasStride]
          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
       c: device pointer or host pointer to scalar cosine component of the rotation matrix.
       s: device pointer or host pointer to scalar sine component of the rotation matrix.
       csType: [hipblasDatatype_t]
          specifies the datatype of c and s.
       batchCount: [int]
          the number of x and y arrays, i.e. the number of batches.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(yType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'yType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(csType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'csType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasRotStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasRotStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,
        <void *>hip._util.types.DataHandle.from_pyobj(y)._ptr,yType.value,incy,stridey,
        <const void *>hip._util.types.DataHandle.from_pyobj(c)._ptr,
        <const void *>hip._util.types.DataHandle.from_pyobj(s)._ptr,csType.value,batchCount,executionType.value))    # fully specified
    return (_hipblasRotStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasScalEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, object executionType):
    """BLAS EX API

    scalEx  scales each element of vector x with scalar alpha.

        x := alpha * x

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x         device pointer storing vector x.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       alpha: device pointer or host pointer for the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       xType: [hipblasDatatype_t]
          specifies the datatype of vector x.
       incx: [int]
          specifies the increment for the elements of x.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasScalEx__retval = hipblasStatus_t(chipblas.hipblasScalEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,executionType.value))    # fully specified
    return (_hipblasScalEx__retval,)


@cython.embedsignature(True)
def hipblasScalBatchedEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, int batchCount, object executionType):
    """BLAS EX API

    scalBatchedEx  scales each element of each vector x_i with scalar alpha.

        x_i := alpha * x_i

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x         device array of device pointers storing each vector x_i.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       alpha: device pointer or host pointer for the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       batchCount: [int]
          number of instances in the batch.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasScalBatchedEx__retval = hipblasStatus_t(chipblas.hipblasScalBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,batchCount,executionType.value))    # fully specified
    return (_hipblasScalBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasScalStridedBatchedEx(object handle, int n, object alpha, object alphaType, object x, object xType, int incx, long stridex, int batchCount, object executionType):
    """BLAS EX API

    scalStridedBatchedEx  scales each element of vector x with scalar alpha over a set
                             of strided batched vectors.

        x := alpha * x

    - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.

    @param[inout]
    x         device pointer to the first vector x_1.

    Args:
       handle: [hipblasHandle_t]
          handle to the hipblas library context queue.
       n: [int]
          the number of elements in x.
       alpha: device pointer or host pointer for the scalar alpha.
       alphaType: [hipblasDatatype_t]
          specifies the datatype of alpha.
       xType: [hipblasDatatype_t]
          specifies the datatype of each vector x_i.
       incx: [int]
          specifies the increment for the elements of each x_i.
       stridex: [hipblasStride]
          stride from the start of one vector (x_i) to the next one (x_i+1).
          There are no restrictions placed on stridex, however the user should
          take care to ensure that stridex is of appropriate size, for a typical
          case this means stridex >= n * incx.
       batchCount: [int]
          number of instances in the batch.
       executionType: [hipblasDatatype_t]
          specifies the datatype of computation.
          ******************************************************************
    """
    if not isinstance(alphaType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'alphaType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(xType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'xType' must be of type '_hipblasDatatype_t__Base'")                    
    if not isinstance(executionType,_hipblasDatatype_t__Base):
        raise TypeError("argument 'executionType' must be of type '_hipblasDatatype_t__Base'")
    _hipblasScalStridedBatchedEx__retval = hipblasStatus_t(chipblas.hipblasScalStridedBatchedEx(
        <void *>hip._util.types.DataHandle.from_pyobj(handle)._ptr,n,
        <const void *>hip._util.types.DataHandle.from_pyobj(alpha)._ptr,alphaType.value,
        <void *>hip._util.types.DataHandle.from_pyobj(x)._ptr,xType.value,incx,stridex,batchCount,executionType.value))    # fully specified
    return (_hipblasScalStridedBatchedEx__retval,)


@cython.embedsignature(True)
def hipblasStatusToString(object status):
    """(No brief)

    hipblasStatusToString

    Returns string representing hipblasStatus_t value   HIPBLAS Auxiliary API

    /
    Args:
       status: [hipblasStatus_t]
          hipBLAS status to convert to string
    """
    if not isinstance(status,_hipblasStatus_t__Base):
        raise TypeError("argument 'status' must be of type '_hipblasStatus_t__Base'")
    cdef const char * _hipblasStatusToString__retval = chipblas.hipblasStatusToString(status.value)    # fully specified
    return (_hipblasStatusToString__retval,)
