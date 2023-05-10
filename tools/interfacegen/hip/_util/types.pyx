# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport cpython.long
cimport cpython.int
cimport cpython.buffer
cimport libc.stdlib
cimport libc.stdint

import ctypes
import math

cdef class DataHandle:
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False

    @staticmethod
    cdef DataHandle from_ptr(void* ptr):
        cdef DataHandle wrapper = DataHandle.__new__(DataHandle)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of DataHandle, only the pointer is copied.
            Releasing an acquired Py_buffer handles is still an obligation of the original object.
        """
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)
       
        self._py_buffer_acquired = False
        if pyobj is None:
            self._ptr = NULL
        elif isinstance(pyobj,DataHandle):
            self._ptr = (<DataHandle>pyobj)._ptr
        elif isinstance(pyobj,int):
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            self._ptr = cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj, 
                &self._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            self._py_buffer_acquired = True
            self._ptr = self._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")

    @staticmethod
    cdef DataHandle from_pyobj(object pyobj):
        """Derives a DataHandle from the given object.

        In case ``pyobj`` is itself an ``DataHandle`` instance, this method
        returns it directly. No new DataHandle is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``DataHandle``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of DataHandle.
        """
        cdef DataHandle wrapper = DataHandle.__new__(DataHandle)
        
        if isinstance(pyobj,DataHandle):
            return pyobj
        else:
            wrapper = DataHandle.__new__(DataHandle)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    @property
    def ptr(self):
        """"Data pointer as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    @property
    def is_ptr_null(self):
        """If data pointer is NULL."""
        return self._ptr == NULL
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<DataHandle object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """"Data pointer as ``ctypes.c_void_p``."""
        return ctypes.c_void_p(self.ptr)

    def __getitem__(self,offset):
        """Returns a new DataHandle whose pointer is this instance's pointer offsetted by ``offset``.

        Args:
            offset (int): Offset (in bytes) to add to this instance's pointer.
        """
        cdef DataHandle result
        if isinstance(offset,int):
            if offset < 0:
                raise ValueError("offset='{offset}' must be non-negative")
            return DataHandle.from_ptr(<void*>(<unsigned long>self._ptr + cpython.long.PyLong_AsUnsignedLong(offset)))
        raise NotImplementedError("'__getitem__': not implemented for other 'offset' types than 'int'")
    
    def __init__(self,object pyobj):
        DataHandle.init_from_pyobj(self,pyobj)

cdef class Array(DataHandle):
    # members declared in declaration part ``types.pxd``

    NUMPY_CHAR_CODES = (
        "?", "=?", "<?", ">?", "bool", "bool_", "bool8",
        "uint8", "u1", "=u1", "<u1", ">u1",
        "uint16", "u2", "=u2", "<u2", ">u2",
        "uint32", "u4", "=u4", "<u4", ">u4",
        "uint64", "u8", "=u8", "<u8", ">u8",

        "int8", "i1", "=i1", "<i1", ">i1",
        "int16", "i2", "=i2", "<i2", ">i2",
        "int32", "i4", "=i4", "<i4", ">i4",
        "int64", "i8", "=i8", "<i8", ">i8",

        "float16", "f2", "=f2", "<f2", ">f2",
        "float32", "f4", "=f4", "<f4", ">f4",
        "float64", "f8", "=f8", "<f8", ">f8",

        "complex64", "c8", "=c8", "<c8", ">c8",
        "complex128", "c16", "=c16", "<c16", ">c16",

        "byte", "b", "=b", "<b", ">b",
        "short", "h", "=h", "<h", ">h",
        "intc", "i", "=i", "<i", ">i",
        "intp", "int0", "p", "=p", "<p", ">p",
        "long", "int", "int_", "l", "=l", "<l", ">l",
        "longlong", "q", "=q", "<q", ">q",

        "ubyte", "B", "=B", "<B", ">B",
        "ushort", "H", "=H", "<H", ">H",
        "uintc", "I", "=I", "<I", ">I",
        "uintp", "uint0", "P", "=P", "<P", ">P",
        "ulong", "uint", "L", "=L", "<L", ">L",
        "ulonglong", "Q", "=Q", "<Q", ">Q",

        "half", "e", "=e", "<e", ">e",
        "single", "f", "=f", "<f", ">f",
        "double", "float", "float_", "d", "=d", "<d", ">d",
        "longdouble", "longfloat", "g", "=g", "<g", ">g",

        "csingle", "singlecomplex", "F", "=F", "<F", ">F",
        "cdouble", "complex", "complex_", "cfloat", "D", "=D", "<D", ">D",
        "clongdouble", "clongfloat", "longcomplex", "G", "=G", "<G", ">G",

        "str", "str_", "str0", "unicode", "unicode_", "U", "=U", "<U", ">U",
        "bytes", "bytes_", "bytes0", "S", "=S", "<S", ">S",
        "void", "void0", "V", "=V", "<V", ">V",
        "object", "object_", "O", "=O", "<O", ">O",

        "datetime64", "=datetime64", "<datetime64", ">datetime64",
        "datetime64[Y]", "=datetime64[Y]", "<datetime64[Y]", ">datetime64[Y]",
        "datetime64[M]", "=datetime64[M]", "<datetime64[M]", ">datetime64[M]",
        "datetime64[W]", "=datetime64[W]", "<datetime64[W]", ">datetime64[W]",
        "datetime64[D]", "=datetime64[D]", "<datetime64[D]", ">datetime64[D]",
        "datetime64[h]", "=datetime64[h]", "<datetime64[h]", ">datetime64[h]",
        "datetime64[m]", "=datetime64[m]", "<datetime64[m]", ">datetime64[m]",
        "datetime64[s]", "=datetime64[s]", "<datetime64[s]", ">datetime64[s]",
        "datetime64[ms]", "=datetime64[ms]", "<datetime64[ms]", ">datetime64[ms]",
        "datetime64[us]", "=datetime64[us]", "<datetime64[us]", ">datetime64[us]",
        "datetime64[ns]", "=datetime64[ns]", "<datetime64[ns]", ">datetime64[ns]",
        "datetime64[ps]", "=datetime64[ps]", "<datetime64[ps]", ">datetime64[ps]",
        "datetime64[fs]", "=datetime64[fs]", "<datetime64[fs]", ">datetime64[fs]",
        "datetime64[as]", "=datetime64[as]", "<datetime64[as]", ">datetime64[as]",
        "M", "=M", "<M", ">M",
        "M8", "=M8", "<M8", ">M8",
        "M8[Y]", "=M8[Y]", "<M8[Y]", ">M8[Y]",
        "M8[M]", "=M8[M]", "<M8[M]", ">M8[M]",
        "M8[W]", "=M8[W]", "<M8[W]", ">M8[W]",
        "M8[D]", "=M8[D]", "<M8[D]", ">M8[D]",
        "M8[h]", "=M8[h]", "<M8[h]", ">M8[h]",
        "M8[m]", "=M8[m]", "<M8[m]", ">M8[m]",
        "M8[s]", "=M8[s]", "<M8[s]", ">M8[s]",
        "M8[ms]", "=M8[ms]", "<M8[ms]", ">M8[ms]",
        "M8[us]", "=M8[us]", "<M8[us]", ">M8[us]",
        "M8[ns]", "=M8[ns]", "<M8[ns]", ">M8[ns]",
        "M8[ps]", "=M8[ps]", "<M8[ps]", ">M8[ps]",
        "M8[fs]", "=M8[fs]", "<M8[fs]", ">M8[fs]",
        "M8[as]", "=M8[as]", "<M8[as]", ">M8[as]",

        "timedelta64", "=timedelta64", "<timedelta64", ">timedelta64",
        "timedelta64[Y]", "=timedelta64[Y]", "<timedelta64[Y]", ">timedelta64[Y]",
        "timedelta64[M]", "=timedelta64[M]", "<timedelta64[M]", ">timedelta64[M]",
        "timedelta64[W]", "=timedelta64[W]", "<timedelta64[W]", ">timedelta64[W]",
        "timedelta64[D]", "=timedelta64[D]", "<timedelta64[D]", ">timedelta64[D]",
        "timedelta64[h]", "=timedelta64[h]", "<timedelta64[h]", ">timedelta64[h]",
        "timedelta64[m]", "=timedelta64[m]", "<timedelta64[m]", ">timedelta64[m]",
        "timedelta64[s]", "=timedelta64[s]", "<timedelta64[s]", ">timedelta64[s]",
        "timedelta64[ms]", "=timedelta64[ms]", "<timedelta64[ms]", ">timedelta64[ms]",
        "timedelta64[us]", "=timedelta64[us]", "<timedelta64[us]", ">timedelta64[us]",
        "timedelta64[ns]", "=timedelta64[ns]", "<timedelta64[ns]", ">timedelta64[ns]",
        "timedelta64[ps]", "=timedelta64[ps]", "<timedelta64[ps]", ">timedelta64[ps]",
        "timedelta64[fs]", "=timedelta64[fs]", "<timedelta64[fs]", ">timedelta64[fs]",
        "timedelta64[as]", "=timedelta64[as]", "<timedelta64[as]", ">timedelta64[as]",
        "m", "=m", "<m", ">m",
        "m8", "=m8", "<m8", ">m8",
        "m8[Y]", "=m8[Y]", "<m8[Y]", ">m8[Y]",
        "m8[M]", "=m8[M]", "<m8[M]", ">m8[M]",
        "m8[W]", "=m8[W]", "<m8[W]", ">m8[W]",
        "m8[D]", "=m8[D]", "<m8[D]", ">m8[D]",
        "m8[h]", "=m8[h]", "<m8[h]", ">m8[h]",
        "m8[m]", "=m8[m]", "<m8[m]", ">m8[m]",
        "m8[s]", "=m8[s]", "<m8[s]", ">m8[s]",
        "m8[ms]", "=m8[ms]", "<m8[ms]", ">m8[ms]",
        "m8[us]", "=m8[us]", "<m8[us]", ">m8[us]",
        "m8[ns]", "=m8[ns]", "<m8[ns]", ">m8[ns]",
        "m8[ps]", "=m8[ps]", "<m8[ps]", ">m8[ps]",
        "m8[fs]", "=m8[fs]", "<m8[fs]", ">m8[fs]",
        "m8[as]", "=m8[as]", "<m8[as]", ">m8[as]",
    )

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False
        self._itemsize = 1
        self.___cuda_array_interface__ = dict(
            shape=(1,),
            typestr='b', # See: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
            data=(None,False), # 1: data pointer as int (long int), 2: read-only?
            strides=None,
            offset=0,
            mask=None,
            version=3,
            # numba
            stream=None, # 
        )

    cdef _set_ptr(self,void* ptr):
        self._ptr = ptr
        self.___cuda_array_interface__["data"][0] = cpython.long.PyLong_FromVoidPtr(ptr)

    @staticmethod
    cdef Array from_ptr(void* ptr):
        cdef Array wrapper = Array.__new__(Array)
        wrapper._set_ptr(ptr)
        return wrapper
    
    @property
    def rank(self):
        """Rank of the underlying data.
        
        See:
            set_bounds
        """
        return len(self.___cuda_array_interface__["shape"])
        
    cdef int _numpy_typestr_to_bytes(self,str typestr):
        if typestr in ("?", "=?", "<?", ">?", "bool", "bool_", "bool8"):
            return <int>sizeof(bool)
        elif typestr in ("uint8", "u1", "=u1", "<u1", ">u1"):
            return <int>sizeof(libc.stdint.uint8_t)
        elif typestr in ("uint16", "u2", "=u2", "<u2", ">u2"):
            return <int>sizeof(libc.stdint.uint16_t)
        elif typestr in ("uint32", "u4", "=u4", "<u4", ">u4"):
            return <int>sizeof(libc.stdint.uint32_t)
        elif typestr in ("uint64", "u8", "=u8", "<u8", ">u8"):
            return <int>sizeof(libc.stdint.uint64_t)
        elif typestr in ("int8", "i1", "=i1", "<i1", ">i1"):
            return <int>sizeof(libc.stdint.int8_t)
        elif typestr in ("int16", "i2", "=i2", "<i2", ">i2"):
            return <int>sizeof(libc.stdint.int16_t)
        elif typestr in ("int32", "i4", "=i4", "<i4", ">i4"):
            return <int>sizeof(libc.stdint.int32_t)
        elif typestr in ("int64", "i8", "=i8", "<i8", ">i8"):
            return <int>sizeof(libc.stdint.int64_t)
        elif typestr in ("float16", "f2", "=f2", "<f2", ">f2"):
            return <int>sizeof(libc.stdint.uint16_t)
        elif typestr in ("float32", "f4", "=f4", "<f4", ">f4"):
            return <int>sizeof(libc.stdint.uint32_t)
        elif typestr in ("float64", "f8", "=f8", "<f8", ">f8"):
            return <int>sizeof(libc.stdint.uint64_t)
        elif typestr in ("complex64", "c8", "=c8", "<c8", ">c8"):
            return <int>sizeof(libc.stdint.uint64_t)
        elif typestr in ("complex128", "c16", "=c16", "<c16", ">c16"):
            return <int>sizeof(libc.stdint.uint64_t)*2
        elif typestr in ("byte", "b", "=b", "<b", ">b"):
            return 1
        elif typestr in ("short", "h", "=h", "<h", ">h"):
            return <int>sizeof(short)
        elif typestr in ("intc", "i", "=i", "<i", ">i"):
            return <int>sizeof(int)
        elif typestr in ("intp", "int0", "p", "=p", "<p", ">p"):
            return <int>sizeof(libc.stdint.intptr_t)
        elif typestr in ("long", "int", "int_", "l", "=l", "<l", ">l"):
            return <int>sizeof(long)
        elif typestr in ("longlong", "q", "=q", "<q", ">q"):
            return <int>sizeof(long long)
        elif typestr in ("ubyte", "B", "=B", "<B", ">B"):
            return 1
        elif typestr in ("ushort", "H", "=H", "<H", ">H"):
            return <int>sizeof(unsigned short)
        elif typestr in ("uintc", "I", "=I", "<I", ">I"):
            return <int>sizeof(unsigned int)
        elif typestr in ("uintp", "uint0", "P", "=P", "<P", ">P"):
            return <int>sizeof(libc.stdint.uintptr_t)
        elif typestr in ("ulong", "uint", "L", "=L", "<L", ">L"):
            return <int>sizeof(unsigned long)
        elif typestr in ("ulonglong", "Q", "=Q", "<Q", ">Q"):
            return <int>sizeof(unsigned long long)
        elif typestr in ("half", "e", "=e", "<e", ">e"):
            return <int>sizeof(libc.stdint.uint16_t)
        elif typestr in ("single", "f", "=f", "<f", ">f"):
            return <int>sizeof(float)
        elif typestr in ("double", "float", "float_", "d", "=d", "<d", ">d"):
            return <int>sizeof(double)
        elif typestr in ("longdouble", "longfloat", "g", "=g", "<g", ">g"):
            return <int>sizeof(long double)
        elif typestr in ("csingle", "singlecomplex", "F", "=F", "<F", ">F"):
            return <int>sizeof(float complex)
        elif typestr in ("cdouble", "complex", "complex_", "cfloat", "D", "=D", "<D", ">D"):
            return <int>sizeof(double complex)
        elif typestr in ("clongdouble", "clongfloat", "longcomplex", "G", "=G", "<G", ">G"):
            return <int>sizeof(long double complex)
        return -1

    def configure(self, **kwargs):
        """
        Args:
            \*\*kwargs: Keyword arguments.
            
        Keyword Arguments:

            shape (tuple): An int that describes the intent per dimension. The length of the tuple is the number of dimensions.
            typestr (str): A numpy typestr, see the first note for more details.
            stream (int or None): The stream to synchronize before consuming this array. See first note for more details.
            itemsize (int): Size in bytes of 
                            each item. Defaults to 1. See the notes.
            read_only (bool): Array is read_only. Second entry of the CAI 'data' tuple. Defaults to False.
            _force(bool): Ignore changes in the total number of bytes when override shape, typestr, and/or itemsize.

        Note:
            More details on the keyword arguments can be found here:
            https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html

        Note:

            This method does not automatically map numpy/numba typestr to appropriate number 
            of bytes, i.e. `itemsize`. Hence, you need to specify itemsize additionally
            when dealing with other datatypes than bytes (typestr: ``'b'``).
        """
        cdef list supported_keys = ["shape","typestr","stream"]
        cdef list extra_keys = ["itemsize","read_only","_force"]
        cdef list allowed_keys_str =", ".join([f"'{e}'" for e in supported_keys + extra_keys])
        cdef bint force_new_shape
        cdef tuple shape
        cdef tuple old_shape
        cdef bint read_only
        cdef int itemsize = -1
        cdef size_t old_num_bytes
        cdef size_t new_num_bytes
        cdef str typestr = None

        for k in kwargs:
            if k not in (supported_keys + extra_keys):
                raise KeyError(f"allowed keyword arguments are: {allowed_keys_str}")
        
        force_new_shape = kwargs.get("_force",False)
        shape = old_shape = self.___cuda_array_interface__["shape"]
        if "shape" in kwargs:
            shape = kwargs["shape"]
            if not len(shape):
                raise ValueError("'shape': must have at least one entry")
            for i in shape:
                if not isinstance(i,int):
                    raise TypeError("'shape': entries must be int")
            #self.___cuda_array_interface__["shape"] = shape
        if "typestr" in kwargs:
            typestr = kwargs["typestr"]
            self.___cuda_array_interface__["typestr"] = typestr
            itemsize = self._numpy_typestr_to_bytes(typestr)
            if itemsize < 0:
                if typestr not in self.NUMPY_CHAR_CODES:
                    raise ValueError(f"'typestr': value '{typestr}' is not a valid numpy char code. See class attributes 'NUMPY_CHAR_CODES' for valid expressions.")
                elif "itemsize" not in kwargs:
                    raise ValueError(f"'typestr': value '{typestr}' could not be mapped to a number of bytes. Please additionally specify 'itemsize'.")
        if "itemsize" in kwargs:
            itemsize = kwargs["itemsize"]
            if not isinstance(itemsize,int):
                raise TypeError("'itemsize': must be int")
            if itemsize <= 0:
                raise ValueError("'itemsize': must be positive int")
        #
        if "stream" in kwargs:
            stream = kwargs["stream"]
            if isinstance(stream,int):
                if stream == 0:
                    return ValueError("'stream': value '0' is disallowed as it would be ambiguous between None and the default stream, more details: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html")
                elif stream < 0:
                    return ValueError("'stream': expected positive integer")
                self.___cuda_array_interface__["stream"] = stream
            else:
                self.___cuda_array_interface__["stream"] = DataHandle.from_pyobj(stream).ptr()
        if "read_only" in kwargs:
            read_only = kwargs["read_only"]
            if not isinstance(shape,bool):
                raise ValueError("'read_only:' expected bool")
            self.___cuda_array_interface__["data"][1] = read_only

        if itemsize > 0 or shape != old_shape:
            old_num_bytes = self._itemsize * math.prod(old_shape)
            new_num_bytes = itemsize * math.prod(shape)
            if old_num_bytes == new_num_bytes or force_new_shape:
                self._itemsize = itemsize
                self.___cuda_array_interface__["shape"] = shape
            else:
                raise ValueError("new shape would change buffer size information: {old_num_bytes} B -> {new_num_bytes} B. Additionaly specify `_force=True` if this is intended.")

        return self

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of Array, only the pointer is copied.
            Releasing an acquired Py_buffer handles is still an obligation of the original object.
        """
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)
       
        self._py_buffer_acquired = False
        if pyobj is None:
            self._set_ptr(NULL)
        elif isinstance(pyobj,int):
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(pyobj))
        elif isinstance(pyobj,ctypes.c_void_p):
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            if cuda_array_interface["strides"] != None:
                raise RuntimeError("CUDA array interface is not contiguous")
            ptr_as_int = cuda_array_interface["data"][0]
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(ptr_as_int))
            self.configure(cuda_array_interface)
            if isinstance(pyobj,Array):
                self._itemsize = pyobj._itemsize
        elif isinstance(pyobj,DataHandle):
            self._set_ptr((<DataHandle>pyobj)._ptr)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            raise NotImplementedError("Py_buffer is no ideal format for data that is not accessible from the host")
        else:
            raise NotImplementedError(f"no conversion implemented for instance of '{type(pyobj)}'")

    @staticmethod
    cdef Array from_pyobj(object pyobj):
        """Derives a Array from the given object.

        In case ``pyobj`` is itself an ``Array`` instance, this method
        returns it directly. No new Array is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``Array``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of Array.
        """
        cdef Array wrapper = Array.__new__(Array)
        
        if isinstance(pyobj,Array):
            return pyobj
        else:
            wrapper = Array.__new__(Array)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    @property
    def ptr(self):
        """"Data pointer as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    @property
    def is_ptr_null(self):
        """If data pointer is NULL."""
        return self._ptr == NULL
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<Array object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """"Data pointer as ``ctypes.c_void_p``."""
        return ctypes.c_void_p(self.ptr)


    cdef tuple _handle_int(self,size_t subscript, size_t shape_dim):
        if subscript < 0:
            raise ValueError(f"subscript='{subscript}' must be non-negative.")  
        if subscript >= shape_dim:
            raise ValueError(f"subscript='{subscript}' must be smaller than axis' exclusive upper bound ('{shape_dim}')")
        return (subscript,subscript+1,False)
        

    cdef tuple _handle_slice(self,slice subscript,size_t shape_dim):
        cdef size_t start = -1
        cdef size_t stop = -1
        cdef bint extract_full_dim = False

        if subscript.step not in (None,1):
            raise ValueError("subscript.step='{subscript.step}' must be 'None' or '1'.")  
        if subscript.stop != None:
            if subscript.stop <= 0:
                raise ValueError("subscript.stop='{subscript.stop}' must be greater than zero.")
            if subscript.stop > shape_dim:
                raise ValueError("subscript.stop='{subscript.stop}' must not be greater than axis' exclusive upper bound ({shape_dim}).")
            stop = subscript.stop
        else:
            stop = shape_dim
        if subscript.start != None:
            if subscript.start < 0:
                raise ValueError("subscript.start='{subscript.start}' must be non-negative.")
            if subscript.start >= shape_dim:
                raise ValueError("subscript.start='{subscript.start}' must be smaller than axis' exclusive upper bound ({shape_dim}).")
            start = subscript.start
        else:
            start = 0

        if start >= stop:
            raise ValueError("subscript.stop='{subscript.stop}' must be greater than subscript.start='{subscript.start}'")

        extract_full_dim = (
            start == 0
            and stop == shape_dim
        )
        return (start,stop,extract_full_dim)


    def __getitem__(self,subscript):
        """Returns a contiguous subarray according to the subscript expression.

        Args:
            Subscript: Either an integer, a slice, or a tuple of slices and integers.

        Note:
            If the subscript is a single integer, only the first axis ("axis 0") of the 
            array is accessed. A KeyError is raised if the extent
            of axis 0 is surpassed. This behavior is identical to that of numpy.
        
        Raise:
            TypeError: If the subscript types are not 'int', 'slice' or a 'tuple' thereof.
            ValueError: If the subscripts do not yield an contiguous subarray. A single array element
                        is regarded as contiguous array of size 1.
        """
        cdef size_t stride = 1
        cdef size_t offset = 0
        cdef bint contiguous = False
        cdef list shape = self.__cuda_array_interface__["shape"]
        cdef size_t rank = len(shape)
        cdef list result_shape = list()
        cdef tuple subscript_tuple
        
        if isinstance(subscript,tuple):
            subscript_tuple = subscript
        elif isinstance(subscript,(slice,int)):
            subscript_tuple = (subscript,)
        else:
            raise TypeError(f"subscript type='{type(subscript)}' is none of: 'slice', 'int', 'tuple'")
        #
        for _i,spec in enumerate(reversed(subscript_tuple)): # row major
            i = rank-_i-1
            if not contiguous:
                raise ValueError(f"subscript='{subscript_tuple}' yields no contiguous subarray")
            if isinstance(spec,int):
                (start,stop,contiguous) = self._handle_int(spec,shape[i])
            elif isinstance(spec,slice):
                (start,stop,contiguous) = self._handle_slice(spec,shape[i])
            else:
                raise TypeError(f"subscript tuple entry type='{type(spec)}' is none of: 'slice', 'int'")
            result_shape.append(stop-start)
            offset = start*stride
            stride *= <size_t>shape[i]
        return Array.from_ptr(<void*>(<unsigned long>self._ptr + offset)).configure({
            "typestr":  self.___cuda_array_interface__["typestr"],
            "itemsize":  self._itemsize,
            "shape": tuple(result_shape),
            "read_only":  self.___cuda_array_interface__["data"]["read_only"],
            "stream": self.___cuda_array_interface__["stream"],
        })

    @property
    def typestr(self):
        return self.___cuda_array_interface__["typestr"]
    @property
    def itemsize(self):
        return self._itemsize
    @property
    def is_read_only(self):
        return self.___cuda_array_interface__["data"]["read_only"]
    @property
    def stream_as_int(self):
        return self.___cuda_array_interface__["stream"]
    
    def __init__(self,object pyobj):
        Array.init_from_pyobj(self,pyobj)

cdef class ListOfBytes(DataHandle):
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._owner = False

    @staticmethod
    cdef ListOfBytes from_ptr(void* ptr):
        cdef ListOfBytes wrapper = ListOfBytes.__new__(ListOfBytes)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of ListOfBytes, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations 
            of the original object.
        """
        cdef const char* entry_as_cstr = NULL

        self._py_buffer_acquired = False
        self._owner = False
        if isinstance(pyobj,ListOfBytes):
            self._ptr = (<ListOfBytes>pyobj)._ptr
        elif isinstance(pyobj,(tuple,list)):
            self._owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(const char*))
            for i,entry in enumerate(pyobj):
                if not isinstance(entry,bytes):
                    raise ValueError("elements of list/tuple input must be of type 'bytes'")
                entry_as_cstr = entry # assumes pyobj/pyobj's entries won't be garbage collected
                # More details: https://cython.readthedocs.io/en/latest/src/tutorial/strings.html
                (<const char**>self._ptr)[i] = entry_as_cstr
        else:
            self._owner = False
            DataHandle.init_from_pyobj(self,pyobj)

    @staticmethod
    cdef ListOfBytes from_pyobj(object pyobj):
        """Derives a ListOfBytes from the given object.

        In case ``pyobj`` is itself an ``ListOfBytes`` instance, this method
        returns it directly. No new ListOfBytes is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ListOfBytes``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfBytes.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfBytes wrapper = ListOfBytes.__new__(ListOfBytes)
        
        if isinstance(pyobj,ListOfBytes):
            return pyobj
        else:
            wrapper = ListOfBytes.__new__(ListOfBytes)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        ListOfBytes.init_from_pyobj(self,pyobj)

cdef class ListOfDataHandle(DataHandle):
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._owner = False

    @staticmethod
    cdef ListOfDataHandle from_ptr(void* ptr):
        cdef ListOfDataHandle wrapper = ListOfDataHandle.__new__(ListOfDataHandle)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of ListOfDataHandle, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations 
            of the original object.
        """
        self._py_buffer_acquired = False
        self._owner = False
        if isinstance(pyobj,ListOfDataHandle):
            self._ptr = (<ListOfDataHandle>pyobj)._ptr
        
        elif isinstance(pyobj,(tuple,list)):
            self._owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(void *))
            for i,entry in enumerate(pyobj):
                (<void**>self._ptr)[i] = cpython.long.PyLong_AsVoidPtr(DataHandle.from_pyobj(entry).ptr())
        else:
            self._owner = False
            DataHandle.init_from_pyobj(self,pyobj)

    @staticmethod
    cdef ListOfDataHandle from_pyobj(object pyobj):
        """Derives a ListOfDataHandle from the given object.

        In case ``pyobj`` is itself an ``ListOfDataHandle`` instance, this method
        returns it directly. No new ListOfDataHandle is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ListOfDataHandle``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfDataHandle.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfDataHandle wrapper = ListOfDataHandle.__new__(ListOfDataHandle)
        
        if isinstance(pyobj,ListOfDataHandle):
            return pyobj
        else:
            wrapper = ListOfDataHandle.__new__(ListOfDataHandle)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        ListOfDataHandle.init_from_pyobj(self,pyobj)

cdef class ListOfInt(DataHandle):
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._owner = False

    @staticmethod
    cdef ListOfInt from_ptr(void* ptr):
        cdef ListOfInt wrapper = ListOfInt.__new__(ListOfInt)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of ListOfInt, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations 
            of the original object.
        """
        self._py_buffer_acquired = False
        self._owner = False
        if isinstance(pyobj,ListOfInt):
            self._ptr = (<ListOfInt>pyobj)._ptr
        
        elif isinstance(pyobj,(tuple,list)):
            self._owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(int))
            for i,entry in enumerate(pyobj):
                if isinstance(entry,int):
                    (<int*>self._ptr)[i] = <int>cpython.long.PyLong_AsLongLong(pyobj)
                elif isinstance(entry,(
                    ctypes.c_bool,
                    ctypes.c_short,
                    ctypes.c_ushort,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_long,
                    ctypes.c_ulong,
                    ctypes.c_longlong,
                    ctypes.c_ulonglong,
                    ctypes.c_size_t,
                    ctypes.c_ssize_t,
                )):
                    (<int*>self._ptr)[i] = <int>cpython.long.PyLong_AsLongLong(entry.value)
                else:
                    raise ValueError(f"element '{i}' of input cannot be converted to C int type")
        else:
            self._owner = False
            DataHandle.init_from_pyobj(self,pyobj)

    @staticmethod
    cdef ListOfInt from_pyobj(object pyobj):
        """Derives a ListOfInt from the given object.

        In case ``pyobj`` is itself an ``ListOfInt`` instance, this method
        returns it directly. No new ListOfInt is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ListOfInt``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfInt.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfInt wrapper = ListOfInt.__new__(ListOfInt)
        
        if isinstance(pyobj,ListOfInt):
            return pyobj
        else:
            wrapper = ListOfInt.__new__(ListOfInt)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        ListOfInt.init_from_pyobj(self,pyobj)

cdef class ListOfUnsigned(DataHandle):
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._owner = False

    @staticmethod
    cdef ListOfUnsigned from_ptr(void* ptr):
        cdef ListOfUnsigned wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of ListOfUnsigned, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations 
            of the original object.
        """
        self._py_buffer_acquired = False
        self._owner = False
        if isinstance(pyobj,ListOfUnsigned):
            self._ptr = (<ListOfUnsigned>pyobj)._ptr
        
        elif isinstance(pyobj,(tuple,list)):
            self._owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned int))
            for i,entry in enumerate(pyobj):
                if isinstance(entry,int):
                    (<unsigned int*>self._ptr)[i] = <unsigned int>cpython.long.PyLong_AsUnsignedLongLong(pyobj)
                elif isinstance(entry,(
                    ctypes.c_bool,
                    ctypes.c_short,
                    ctypes.c_ushort,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_long,
                    ctypes.c_ulong,
                    ctypes.c_longlong,
                    ctypes.c_ulonglong,
                    ctypes.c_size_t,
                    ctypes.c_ssize_t,
                )):
                    (<unsigned int*>self._ptr)[i] = <unsigned int>cpython.long.PyLong_AsUnsignedLongLong(entry.value)
                else:
                    raise ValueError(f"element '{i}' of input cannot be converted to C unsigned int type")
        else:
            self._owner = False
            DataHandle.init_from_pyobj(self,pyobj)

    @staticmethod
    cdef ListOfUnsigned from_pyobj(object pyobj):
        """Derives a ListOfUnsigned from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsigned`` instance, this method
        returns it directly. No new ListOfUnsigned is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ListOfUnsigned``, ``unsigned int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfUnsigned.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsigned wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
        
        if isinstance(pyobj,ListOfUnsigned):
            return pyobj
        else:
            wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        ListOfUnsigned.init_from_pyobj(self,pyobj)

cdef class ListOfUnsignedLong(DataHandle):
    # members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._owner = False

    @staticmethod
    cdef ListOfUnsignedLong from_ptr(void* ptr):
        cdef ListOfUnsignedLong wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        NOTE:
            If ``pyobj`` is an instance of ListOfUnsignedLong, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations 
            of the original object.
        """
        self._py_buffer_acquired = False
        self._owner = False
        if isinstance(pyobj,ListOfUnsignedLong):
            self._ptr = (<ListOfUnsignedLong>pyobj)._ptr
        
        elif isinstance(pyobj,(tuple,list)):
            self._owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned long))
            for i,entry in enumerate(pyobj):
                if isinstance(entry,int):
                    (<unsigned long*>self._ptr)[i] = <unsigned long>cpython.long.PyLong_AsUnsignedLongLong(pyobj)
                elif isinstance(entry,(
                    ctypes.c_bool,
                    ctypes.c_short,
                    ctypes.c_ushort,
                    ctypes.c_int,
                    ctypes.c_uint,
                    ctypes.c_long,
                    ctypes.c_ulong,
                    ctypes.c_longlong,
                    ctypes.c_ulonglong,
                    ctypes.c_size_t,
                    ctypes.c_ssize_t,
                )):
                    (<unsigned long*>self._ptr)[i] = <unsigned long>cpython.long.PyLong_AsUnsignedLongLong(entry.value)
                else:
                    raise ValueError(f"element '{i}' of input cannot be converted to C unsigned long type")
        else:
            self._owner = False
            DataHandle.init_from_pyobj(self,pyobj)

    @staticmethod
    cdef ListOfUnsignedLong from_pyobj(object pyobj):
        """Derives a ListOfUnsignedLong from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsignedLong`` instance, this method
        returns it directly. No new ListOfUnsignedLong is created.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ListOfUnsignedLong``, ``unsigned long``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfUnsignedLong.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsignedLong wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
        
        if isinstance(pyobj,ListOfUnsignedLong):
            return pyobj
        else:
            wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        ListOfUnsignedLong.init_from_pyobj(self,pyobj)