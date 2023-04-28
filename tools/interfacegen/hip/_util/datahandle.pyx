# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import ctypes

cimport cpython.long
cimport cpython.buffer

cdef class DataHandle:
    # members declared in declaration part ``datahandle.pxd``

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False
    
    @staticmethod
    cdef DataHandle from_ptr(void* ptr):
        cdef DataHandle wrapper = DataHandle.__new__(DataHandle)
        wrapper._ptr = ptr
        return wrapper

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
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)
        
        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,DataHandle):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value)
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj, 
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
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
