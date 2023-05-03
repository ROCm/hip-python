# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import ctypes

cimport cpython.long
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
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value)
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
            generic_handle = DataHandle.init_from_pyobj(self,pyobj)

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
