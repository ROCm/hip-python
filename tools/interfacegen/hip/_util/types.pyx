# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import ctypes

cimport cpython.long
cimport cpython.int
cimport cpython.buffer
cimport libc.stdlib

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
    
    def __init__(self,object pyobj):
        DataHandle.init_from_pyobj(self,pyobj)

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
                if isinstance(entry,unsigned int):
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