# AMD_COPYRIGHT
import cython
import ctypes
import enum
HIPRAND_VERSION = chiprand.HIPRAND_VERSION

HIPRAND_DEFAULT_MAX_BLOCK_SIZE = chiprand.HIPRAND_DEFAULT_MAX_BLOCK_SIZE

HIPRAND_DEFAULT_MIN_WARPS_PER_EU = chiprand.HIPRAND_DEFAULT_MIN_WARPS_PER_EU

cdef class uint4:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef uint4 from_ptr(chiprand.uint4* ptr, bint owner=False):
        """Factory function to create ``uint4`` objects from
        given ``chiprand.uint4`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef uint4 wrapper = uint4.__new__(uint4)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef uint4 from_pyobj(object pyobj):
        """Derives a uint4 from a Python object.

        Derives a uint4 from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``uint4`` reference, this method
        returns it directly. No new ``uint4`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``uint4``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of uint4!
        """
        cdef uint4 wrapper = uint4.__new__(uint4)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,uint4):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chiprand.uint4*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chiprand.uint4*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chiprand.uint4*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chiprand.uint4*>wrapper._py_buffer.buf
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
    cdef __allocate(chiprand.uint4** ptr):
        ptr[0] = <chiprand.uint4*>stdlib.malloc(sizeof(chiprand.uint4))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef uint4 new():
        """Factory function to create uint4 objects with
        newly allocated chiprand.uint4"""
        cdef chiprand.uint4* ptr
        uint4.__allocate(&ptr)
        return uint4.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        uint4.__allocate(&self._ptr)
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
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<uint4 object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_x(self, i):
        """Get value ``x`` of ``self._ptr[i]``.
        """
        return self._ptr[i].x
    def set_x(self, i, unsigned int value):
        """Set value ``x`` of ``self._ptr[i]``.
        """
        self._ptr[i].x = value
    @property
    def x(self):
        return self.get_x(0)
    @x.setter
    def x(self, unsigned int value):
        self.set_x(0,value)

    def get_y(self, i):
        """Get value ``y`` of ``self._ptr[i]``.
        """
        return self._ptr[i].y
    def set_y(self, i, unsigned int value):
        """Set value ``y`` of ``self._ptr[i]``.
        """
        self._ptr[i].y = value
    @property
    def y(self):
        return self.get_y(0)
    @y.setter
    def y(self, unsigned int value):
        self.set_y(0,value)

    def get_z(self, i):
        """Get value ``z`` of ``self._ptr[i]``.
        """
        return self._ptr[i].z
    def set_z(self, i, unsigned int value):
        """Set value ``z`` of ``self._ptr[i]``.
        """
        self._ptr[i].z = value
    @property
    def z(self):
        return self.get_z(0)
    @z.setter
    def z(self, unsigned int value):
        self.set_z(0,value)

    def get_w(self, i):
        """Get value ``w`` of ``self._ptr[i]``.
        """
        return self._ptr[i].w
    def set_w(self, i, unsigned int value):
        """Set value ``w`` of ``self._ptr[i]``.
        """
        self._ptr[i].w = value
    @property
    def w(self):
        return self.get_w(0)
    @w.setter
    def w(self, unsigned int value):
        self.set_w(0,value)

    @staticmethod
    def PROPERTIES():
        return ["x","y","z","w"]

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


cdef class rocrand_discrete_distribution_st:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef rocrand_discrete_distribution_st from_ptr(chiprand.rocrand_discrete_distribution_st* ptr, bint owner=False):
        """Factory function to create ``rocrand_discrete_distribution_st`` objects from
        given ``chiprand.rocrand_discrete_distribution_st`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef rocrand_discrete_distribution_st wrapper = rocrand_discrete_distribution_st.__new__(rocrand_discrete_distribution_st)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef rocrand_discrete_distribution_st from_pyobj(object pyobj):
        """Derives a rocrand_discrete_distribution_st from a Python object.

        Derives a rocrand_discrete_distribution_st from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``rocrand_discrete_distribution_st`` reference, this method
        returns it directly. No new ``rocrand_discrete_distribution_st`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``rocrand_discrete_distribution_st``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of rocrand_discrete_distribution_st!
        """
        cdef rocrand_discrete_distribution_st wrapper = rocrand_discrete_distribution_st.__new__(rocrand_discrete_distribution_st)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,rocrand_discrete_distribution_st):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chiprand.rocrand_discrete_distribution_st*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chiprand.rocrand_discrete_distribution_st*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chiprand.rocrand_discrete_distribution_st*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chiprand.rocrand_discrete_distribution_st*>wrapper._py_buffer.buf
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
    cdef __allocate(chiprand.rocrand_discrete_distribution_st** ptr):
        ptr[0] = <chiprand.rocrand_discrete_distribution_st*>stdlib.malloc(sizeof(chiprand.rocrand_discrete_distribution_st))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef rocrand_discrete_distribution_st new():
        """Factory function to create rocrand_discrete_distribution_st objects with
        newly allocated chiprand.rocrand_discrete_distribution_st"""
        cdef chiprand.rocrand_discrete_distribution_st* ptr
        rocrand_discrete_distribution_st.__allocate(&ptr)
        return rocrand_discrete_distribution_st.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        rocrand_discrete_distribution_st.__allocate(&self._ptr)
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
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<rocrand_discrete_distribution_st object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_size(self, i):
        """Get value ``size`` of ``self._ptr[i]``.
        """
        return self._ptr[i].size
    def set_size(self, i, unsigned int value):
        """Set value ``size`` of ``self._ptr[i]``.
        """
        self._ptr[i].size = value
    @property
    def size(self):
        return self.get_size(0)
    @size.setter
    def size(self, unsigned int value):
        self.set_size(0,value)

    def get_offset(self, i):
        """Get value ``offset`` of ``self._ptr[i]``.
        """
        return self._ptr[i].offset
    def set_offset(self, i, unsigned int value):
        """Set value ``offset`` of ``self._ptr[i]``.
        """
        self._ptr[i].offset = value
    @property
    def offset(self):
        return self.get_offset(0)
    @offset.setter
    def offset(self, unsigned int value):
        self.set_offset(0,value)

    def get_alias(self, i):
        """Get value ``alias`` of ``self._ptr[i]``.
        """
        return hip._util.types.ListOfUnsigned.from_ptr(self._ptr[i].alias)
    def set_alias(self, i, object value):
        """Set value ``alias`` of ``self._ptr[i]``.

        Note:
            This can be dangerous if the pointer is from a python object
            that is later on garbage collected.
        """
        self._ptr[i].alias = <unsigned int *>cpython.long.PyLong_AsVoidPtr(hip._util.types.ListOfUnsigned.from_pyobj(value).ptr)
    @property
    def alias(self):
        """
        Note:
            Setting this alias can be dangerous if the underlying pointer is from a python object that
            is later on garbage collected.
        """
        return self.get_alias(0)
    @alias.setter
    def alias(self, object value):
        self.set_alias(0,value)

    def get_probability(self, i):
        """Get value ``probability`` of ``self._ptr[i]``.
        """
        return hip._util.types.DataHandle.from_ptr(self._ptr[i].probability)
    def set_probability(self, i, object value):
        """Set value ``probability`` of ``self._ptr[i]``.

        Note:
            This can be dangerous if the pointer is from a python object
            that is later on garbage collected.
        """
        self._ptr[i].probability = <double *>cpython.long.PyLong_AsVoidPtr(hip._util.types.DataHandle.from_pyobj(value).ptr)
    @property
    def probability(self):
        """
        Note:
            Setting this probability can be dangerous if the underlying pointer is from a python object that
            is later on garbage collected.
        """
        return self.get_probability(0)
    @probability.setter
    def probability(self, object value):
        self.set_probability(0,value)

    def get_cdf(self, i):
        """Get value ``cdf`` of ``self._ptr[i]``.
        """
        return hip._util.types.DataHandle.from_ptr(self._ptr[i].cdf)
    def set_cdf(self, i, object value):
        """Set value ``cdf`` of ``self._ptr[i]``.

        Note:
            This can be dangerous if the pointer is from a python object
            that is later on garbage collected.
        """
        self._ptr[i].cdf = <double *>cpython.long.PyLong_AsVoidPtr(hip._util.types.DataHandle.from_pyobj(value).ptr)
    @property
    def cdf(self):
        """
        Note:
            Setting this cdf can be dangerous if the underlying pointer is from a python object that
            is later on garbage collected.
        """
        return self.get_cdf(0)
    @cdf.setter
    def cdf(self, object value):
        self.set_cdf(0,value)

    @staticmethod
    def PROPERTIES():
        return ["size","offset","alias","probability","cdf"]

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


rocrand_discrete_distribution = rocrand_discrete_distribution_st

cdef class rocrand_generator_base_type:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef rocrand_generator_base_type from_ptr(chiprand.rocrand_generator_base_type* ptr, bint owner=False):
        """Factory function to create ``rocrand_generator_base_type`` objects from
        given ``chiprand.rocrand_generator_base_type`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef rocrand_generator_base_type wrapper = rocrand_generator_base_type.__new__(rocrand_generator_base_type)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef rocrand_generator_base_type from_pyobj(object pyobj):
        """Derives a rocrand_generator_base_type from a Python object.

        Derives a rocrand_generator_base_type from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``rocrand_generator_base_type`` reference, this method
        returns it directly. No new ``rocrand_generator_base_type`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``rocrand_generator_base_type``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of rocrand_generator_base_type!
        """
        cdef rocrand_generator_base_type wrapper = rocrand_generator_base_type.__new__(rocrand_generator_base_type)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,rocrand_generator_base_type):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chiprand.rocrand_generator_base_type*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chiprand.rocrand_generator_base_type*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chiprand.rocrand_generator_base_type*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chiprand.rocrand_generator_base_type*>wrapper._py_buffer.buf
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
        return f"<rocrand_generator_base_type object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
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


rocrand_generator = rocrand_generator_base_type

class rocrand_status(enum.IntEnum):
    ROCRAND_STATUS_SUCCESS = chiprand.ROCRAND_STATUS_SUCCESS
    ROCRAND_STATUS_VERSION_MISMATCH = chiprand.ROCRAND_STATUS_VERSION_MISMATCH
    ROCRAND_STATUS_NOT_CREATED = chiprand.ROCRAND_STATUS_NOT_CREATED
    ROCRAND_STATUS_ALLOCATION_FAILED = chiprand.ROCRAND_STATUS_ALLOCATION_FAILED
    ROCRAND_STATUS_TYPE_ERROR = chiprand.ROCRAND_STATUS_TYPE_ERROR
    ROCRAND_STATUS_OUT_OF_RANGE = chiprand.ROCRAND_STATUS_OUT_OF_RANGE
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = chiprand.ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = chiprand.ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED
    ROCRAND_STATUS_LAUNCH_FAILURE = chiprand.ROCRAND_STATUS_LAUNCH_FAILURE
    ROCRAND_STATUS_INTERNAL_ERROR = chiprand.ROCRAND_STATUS_INTERNAL_ERROR
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class rocrand_rng_type(enum.IntEnum):
    ROCRAND_RNG_PSEUDO_DEFAULT = chiprand.ROCRAND_RNG_PSEUDO_DEFAULT
    ROCRAND_RNG_PSEUDO_XORWOW = chiprand.ROCRAND_RNG_PSEUDO_XORWOW
    ROCRAND_RNG_PSEUDO_MRG32K3A = chiprand.ROCRAND_RNG_PSEUDO_MRG32K3A
    ROCRAND_RNG_PSEUDO_MTGP32 = chiprand.ROCRAND_RNG_PSEUDO_MTGP32
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = chiprand.ROCRAND_RNG_PSEUDO_PHILOX4_32_10
    ROCRAND_RNG_PSEUDO_MRG31K3P = chiprand.ROCRAND_RNG_PSEUDO_MRG31K3P
    ROCRAND_RNG_PSEUDO_LFSR113 = chiprand.ROCRAND_RNG_PSEUDO_LFSR113
    ROCRAND_RNG_QUASI_DEFAULT = chiprand.ROCRAND_RNG_QUASI_DEFAULT
    ROCRAND_RNG_QUASI_SOBOL32 = chiprand.ROCRAND_RNG_QUASI_SOBOL32
    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = chiprand.ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
    ROCRAND_RNG_QUASI_SOBOL64 = chiprand.ROCRAND_RNG_QUASI_SOBOL64
    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = chiprand.ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hiprandGenerator_t = rocrand_generator_base_type

hiprandDiscreteDistribution_t = rocrand_discrete_distribution_st

class hiprandStatus(enum.IntEnum):
    HIPRAND_STATUS_SUCCESS = chiprand.HIPRAND_STATUS_SUCCESS
    HIPRAND_STATUS_VERSION_MISMATCH = chiprand.HIPRAND_STATUS_VERSION_MISMATCH
    HIPRAND_STATUS_NOT_INITIALIZED = chiprand.HIPRAND_STATUS_NOT_INITIALIZED
    HIPRAND_STATUS_ALLOCATION_FAILED = chiprand.HIPRAND_STATUS_ALLOCATION_FAILED
    HIPRAND_STATUS_TYPE_ERROR = chiprand.HIPRAND_STATUS_TYPE_ERROR
    HIPRAND_STATUS_OUT_OF_RANGE = chiprand.HIPRAND_STATUS_OUT_OF_RANGE
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = chiprand.HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
    HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = chiprand.HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
    HIPRAND_STATUS_LAUNCH_FAILURE = chiprand.HIPRAND_STATUS_LAUNCH_FAILURE
    HIPRAND_STATUS_PREEXISTING_FAILURE = chiprand.HIPRAND_STATUS_PREEXISTING_FAILURE
    HIPRAND_STATUS_INITIALIZATION_FAILED = chiprand.HIPRAND_STATUS_INITIALIZATION_FAILED
    HIPRAND_STATUS_ARCH_MISMATCH = chiprand.HIPRAND_STATUS_ARCH_MISMATCH
    HIPRAND_STATUS_INTERNAL_ERROR = chiprand.HIPRAND_STATUS_INTERNAL_ERROR
    HIPRAND_STATUS_NOT_IMPLEMENTED = chiprand.HIPRAND_STATUS_NOT_IMPLEMENTED
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class hiprandRngType(enum.IntEnum):
    HIPRAND_RNG_TEST = chiprand.HIPRAND_RNG_TEST
    HIPRAND_RNG_PSEUDO_DEFAULT = chiprand.HIPRAND_RNG_PSEUDO_DEFAULT
    HIPRAND_RNG_PSEUDO_XORWOW = chiprand.HIPRAND_RNG_PSEUDO_XORWOW
    HIPRAND_RNG_PSEUDO_MRG32K3A = chiprand.HIPRAND_RNG_PSEUDO_MRG32K3A
    HIPRAND_RNG_PSEUDO_MTGP32 = chiprand.HIPRAND_RNG_PSEUDO_MTGP32
    HIPRAND_RNG_PSEUDO_MT19937 = chiprand.HIPRAND_RNG_PSEUDO_MT19937
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = chiprand.HIPRAND_RNG_PSEUDO_PHILOX4_32_10
    HIPRAND_RNG_QUASI_DEFAULT = chiprand.HIPRAND_RNG_QUASI_DEFAULT
    HIPRAND_RNG_QUASI_SOBOL32 = chiprand.HIPRAND_RNG_QUASI_SOBOL32
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = chiprand.HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
    HIPRAND_RNG_QUASI_SOBOL64 = chiprand.HIPRAND_RNG_QUASI_SOBOL64
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = chiprand.HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def hiprandCreateGenerator(object rng_type):
    """\brief Creates a new random number generator.
    Creates a new random number generator of type \p rng_type,
    and returns it in \p generator. That generator will use
    GPU to create random numbers.
    Values for \p rng_type are:
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
    \param generator - Pointer to generator
    \param rng_type - Type of random number generator to create
    \return
    - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
    - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
    - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
    dynamically linked library version \n
    - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
    - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type \p rng_type is not implemented yet \n
    - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
    """
    generator = rocrand_generator_base_type.from_ptr(NULL)
    if not isinstance(rng_type,hiprandRngType):
        raise TypeError("argument 'rng_type' must be of type 'hiprandRngType'")
    _hiprandCreateGenerator__retval = hiprandStatus(chiprand.hiprandCreateGenerator(&generator._ptr,rng_type.value))    # fully specified
    return (_hiprandCreateGenerator__retval,generator)


@cython.embedsignature(True)
def hiprandCreateGeneratorHost(object rng_type):
    """\brief Creates a new random number generator on host.
    Creates a new host random number generator of type \p rng_type
    and returns it in \p generator. Created generator will use
    host CPU to generate random numbers.
    Values for \p rng_type are:
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
    \param generator - Pointer to generator
    \param rng_type - Type of random number generator to create
    \return
    - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
    - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
    dynamically linked library version \n
    - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
    - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type \p rng_type is not implemented yet \n
    - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
    """
    generator = rocrand_generator_base_type.from_ptr(NULL)
    if not isinstance(rng_type,hiprandRngType):
        raise TypeError("argument 'rng_type' must be of type 'hiprandRngType'")
    _hiprandCreateGeneratorHost__retval = hiprandStatus(chiprand.hiprandCreateGeneratorHost(&generator._ptr,rng_type.value))    # fully specified
    return (_hiprandCreateGeneratorHost__retval,generator)


@cython.embedsignature(True)
def hiprandDestroyGenerator(object generator):
    """\brief Destroys random number generator.
    Destroys random number generator and frees related memory.
    \param generator - Generator to be destroyed
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully \n
    """
    _hiprandDestroyGenerator__retval = hiprandStatus(chiprand.hiprandDestroyGenerator(
        rocrand_generator_base_type.from_pyobj(generator)._ptr))    # fully specified
    return (_hiprandDestroyGenerator__retval,)


@cython.embedsignature(True)
def hiprandGenerate(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed 32-bit unsigned integers.
    Generates \p n uniformly distributed 32-bit unsigned integers and
    saves them to \p output_data.
    Generated numbers are between \p 0 and \p 2^32, including \p 0 and
    excluding \p 2^32.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of 32-bit unsigned integers to generate
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerate__retval = hiprandStatus(chiprand.hiprandGenerate(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned int *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerate__retval,)


@cython.embedsignature(True)
def hiprandGenerateChar(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed 8-bit unsigned integers.
    Generates \p n uniformly distributed 8-bit unsigned integers and
    saves them to \p output_data.
    Generated numbers are between \p 0 and \p 2^8, including \p 0 and
    excluding \p 2^8.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of 8-bit unsigned integers to generate
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateChar__retval = hiprandStatus(chiprand.hiprandGenerateChar(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned char *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateChar__retval,)


@cython.embedsignature(True)
def hiprandGenerateShort(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed 16-bit unsigned integers.
    Generates \p n uniformly distributed 16-bit unsigned integers and
    saves them to \p output_data.
    Generated numbers are between \p 0 and \p 2^16, including \p 0 and
    excluding \p 2^16.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of 16-bit unsigned integers to generate
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateShort__retval = hiprandStatus(chiprand.hiprandGenerateShort(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned short *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateShort__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniform(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed floats.
    Generates \p n uniformly distributed 32-bit floating-point values
    and saves them to \p output_data.
    Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
    including \p 1.0f.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of floats to generate
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateUniform__retval = hiprandStatus(chiprand.hiprandGenerateUniform(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniform__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniformDouble(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed double-precision floating-point values.
    Generates \p n uniformly distributed 64-bit double-precision floating-point
    values and saves them to \p output_data.
    Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
    including \p 1.0.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of floats to generate
    Note: When \p generator is of type: \p HIPRAND_RNG_PSEUDO_MRG32K3A,
    \p HIPRAND_RNG_PSEUDO_MTGP32, or \p HIPRAND_RNG_QUASI_SOBOL32,
    then the returned \p double values are generated from only 32 random bits
    each (one <tt>unsigned int</tt> value per one generated \p double).
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateUniformDouble__retval = hiprandStatus(chiprand.hiprandGenerateUniformDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniformDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniformHalf(object generator, object output_data, unsigned long n):
    """\brief Generates uniformly distributed half-precision floating-point values.
    Generates \p n uniformly distributed 16-bit half-precision floating-point
    values and saves them to \p output_data.
    Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
    including \p 1.0.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of halfs to generate
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateUniformHalf__retval = hiprandStatus(chiprand.hiprandGenerateUniformHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniformHalf__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormal(object generator, object output_data, unsigned long n, float mean, float stddev):
    """\brief Generates normally distributed floats.
    Generates \p n normally distributed 32-bit floating-point
    values and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of floats to generate
    \param mean - Mean value of normal distribution
    \param stddev - Standard deviation value of normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateNormal__retval = hiprandStatus(chiprand.hiprandGenerateNormal(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormal__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormalDouble(object generator, object output_data, unsigned long n, double mean, double stddev):
    """\brief Generates normally distributed doubles.
    Generates \p n normally distributed 64-bit double-precision floating-point
    numbers and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of doubles to generate
    \param mean - Mean value of normal distribution
    \param stddev - Standard deviation value of normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateNormalDouble__retval = hiprandStatus(chiprand.hiprandGenerateNormalDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormalDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormalHalf(object generator, object output_data, unsigned long n, int mean, int stddev):
    """\brief Generates normally distributed halfs.
    Generates \p n normally distributed 16-bit half-precision floating-point
    numbers and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of halfs to generate
    \param mean - Mean value of normal distribution
    \param stddev - Standard deviation value of normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateNormalHalf__retval = hiprandStatus(chiprand.hiprandGenerateNormalHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormalHalf__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormal(object generator, object output_data, unsigned long n, float mean, float stddev):
    """\brief Generates log-normally distributed floats.
    Generates \p n log-normally distributed 32-bit floating-point values
    and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of floats to generate
    \param mean - Mean value of log normal distribution
    \param stddev - Standard deviation value of log normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateLogNormal__retval = hiprandStatus(chiprand.hiprandGenerateLogNormal(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormal__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormalDouble(object generator, object output_data, unsigned long n, double mean, double stddev):
    """\brief Generates log-normally distributed doubles.
    Generates \p n log-normally distributed 64-bit double-precision floating-point
    values and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of doubles to generate
    \param mean - Mean value of log normal distribution
    \param stddev - Standard deviation value of log normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateLogNormalDouble__retval = hiprandStatus(chiprand.hiprandGenerateLogNormalDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormalDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormalHalf(object generator, object output_data, unsigned long n, int mean, int stddev):
    """\brief Generates log-normally distributed halfs.
    Generates \p n log-normally distributed 16-bit half-precision floating-point
    values and saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of halfs to generate
    \param mean - Mean value of log normal distribution
    \param stddev - Standard deviation value of log normal distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
    aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGenerateLogNormalHalf__retval = hiprandStatus(chiprand.hiprandGenerateLogNormalHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormalHalf__retval,)


@cython.embedsignature(True)
def hiprandGeneratePoisson(object generator, object output_data, unsigned long n, double lambda_):
    """\brief Generates Poisson-distributed 32-bit unsigned integers.
    Generates \p n Poisson-distributed 32-bit unsigned integers and
    saves them to \p output_data.
    \param generator - Generator to use
    \param output_data - Pointer to memory to store generated numbers
    \param n - Number of 32-bit unsigned integers to generate
    \param lambda - lambda for the Poisson distribution
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
    - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
    - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
    of used quasi-random generator \n
    - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
    """
    _hiprandGeneratePoisson__retval = hiprandStatus(chiprand.hiprandGeneratePoisson(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned int *>hip._util.types.DataHandle.from_pyobj(output_data)._ptr,n,lambda_))    # fully specified
    return (_hiprandGeneratePoisson__retval,)


@cython.embedsignature(True)
def hiprandGenerateSeeds(object generator):
    """\brief Initializes the generator's state on GPU or host.
    Initializes the generator's state on GPU or host.
    If hiprandGenerateSeeds() was not called for a generator, it will be
    automatically called by functions which generates random numbers like
    hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.
    \param generator - Generator to initialize
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
    - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
    a previous kernel launch \n
    - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
    - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully \n
    """
    _hiprandGenerateSeeds__retval = hiprandStatus(chiprand.hiprandGenerateSeeds(
        rocrand_generator_base_type.from_pyobj(generator)._ptr))    # fully specified
    return (_hiprandGenerateSeeds__retval,)


@cython.embedsignature(True)
def hiprandSetStream(object generator, object stream):
    """\brief Sets the current stream for kernel launches.
    Sets the current stream for all kernel launches of the generator.
    All functions will use this stream.
    \param generator - Generator to modify
    \param stream - Stream to use or NULL for default stream
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_SUCCESS if stream was set successfully \n
    """
    _hiprandSetStream__retval = hiprandStatus(chiprand.hiprandSetStream(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hiprandSetStream__retval,)


@cython.embedsignature(True)
def hiprandSetPseudoRandomGeneratorSeed(object generator, unsigned long long seed):
    """\brief Sets the seed of a pseudo-random number generator.
    Sets the seed of the pseudo-random number generator.
    - This operation resets the generator's internal state.
    - This operation does not change the generator's offset.
    \param generator - Pseudo-random number generator
    \param seed - New seed value
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator \n
    - HIPRAND_STATUS_SUCCESS if seed was set successfully \n
    """
    _hiprandSetPseudoRandomGeneratorSeed__retval = hiprandStatus(chiprand.hiprandSetPseudoRandomGeneratorSeed(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,seed))    # fully specified
    return (_hiprandSetPseudoRandomGeneratorSeed__retval,)


@cython.embedsignature(True)
def hiprandSetGeneratorOffset(object generator, unsigned long long offset):
    """\brief Sets the offset of a random number generator.
    Sets the absolute offset of the random number generator.
    - This operation resets the generator's internal state.
    - This operation does not change the generator's seed.
    Absolute offset cannot be set if generator's type is
    HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.
    \param generator - Random number generator
    \param offset - New absolute offset
    \return
    - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
    - HIPRAND_STATUS_SUCCESS if offset was successfully set \n
    - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
    or HIPRAND_RNG_PSEUDO_MT19937 \n
    """
    _hiprandSetGeneratorOffset__retval = hiprandStatus(chiprand.hiprandSetGeneratorOffset(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,offset))    # fully specified
    return (_hiprandSetGeneratorOffset__retval,)


@cython.embedsignature(True)
def hiprandSetQuasiRandomGeneratorDimensions(object generator, unsigned int dimensions):
    """\brief Set the number of dimensions of a quasi-random number generator.
    Set the number of dimensions of a quasi-random number generator.
    Supported values of \p dimensions are 1 to 20000.
    - This operation resets the generator's internal state.
    - This operation does not change the generator's offset.
    \param generator - Quasi-random number generator
    \param dimensions - Number of dimensions
    \return
    - HIPRAND_STATUS_NOT_CREATED if the generator wasn't created \n
    - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
    - HIPRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
    - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
    """
    _hiprandSetQuasiRandomGeneratorDimensions__retval = hiprandStatus(chiprand.hiprandSetQuasiRandomGeneratorDimensions(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,dimensions))    # fully specified
    return (_hiprandSetQuasiRandomGeneratorDimensions__retval,)


@cython.embedsignature(True)
def hiprandGetVersion():
    """\brief Returns the version number of the cuRAND or rocRAND library.
    Returns in \p version the version number of the underlying cuRAND or
    rocRAND library.
    \param version - Version of the library
    \return
    - HIPRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
    - HIPRAND_STATUS_SUCCESS if the version number was successfully returned \n
    """
    cdef int version
    _hiprandGetVersion__retval = hiprandStatus(chiprand.hiprandGetVersion(&version))    # fully specified
    return (_hiprandGetVersion__retval,version)


@cython.embedsignature(True)
def hiprandCreatePoissonDistribution(double lambda_):
    """\brief Construct the histogram for a Poisson distribution.
    Construct the histogram for the Poisson distribution with lambda \p lambda.
    \param lambda - lambda for the Poisson distribution
    \param discrete_distribution - pointer to the histogram in device memory
    \return
    - HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
    - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
    - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
    - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
    """
    discrete_distribution = rocrand_discrete_distribution_st.from_ptr(NULL)
    _hiprandCreatePoissonDistribution__retval = hiprandStatus(chiprand.hiprandCreatePoissonDistribution(lambda_,&discrete_distribution._ptr))    # fully specified
    return (_hiprandCreatePoissonDistribution__retval,discrete_distribution)


@cython.embedsignature(True)
def hiprandDestroyDistribution(object discrete_distribution):
    """\brief Destroy the histogram array for a discrete distribution.
    Destroy the histogram array for a discrete distribution created by
    hiprandCreatePoissonDistribution.
    \param discrete_distribution - pointer to the histogram in device memory
    \return
    - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
    - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
    """
    _hiprandDestroyDistribution__retval = hiprandStatus(chiprand.hiprandDestroyDistribution(
        rocrand_discrete_distribution_st.from_pyobj(discrete_distribution)._ptr))    # fully specified
    return (_hiprandDestroyDistribution__retval,)
