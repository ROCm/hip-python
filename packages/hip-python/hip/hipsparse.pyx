# AMD_COPYRIGHT

"""
Attributes:
    bsrsv2Info_t:
        alias of `~.bsrsv2Info`

    bsrsm2Info_t:
        alias of `~.bsrsm2Info`

    bsrilu02Info_t:
        alias of `~.bsrilu02Info`

    bsric02Info_t:
        alias of `~.bsric02Info`

    csrsv2Info_t:
        alias of `~.csrsv2Info`

    csrsm2Info_t:
        alias of `~.csrsm2Info`

    csrilu02Info_t:
        alias of `~.csrilu02Info`

    csric02Info_t:
        alias of `~.csric02Info`

    csrgemm2Info_t:
        alias of `~.csrgemm2Info`

    pruneInfo_t:
        alias of `~.pruneInfo`

    csru2csrInfo_t:
        alias of `~.csru2csrInfo`

    hipsparseSpGEMMDescr_t:
        alias of `~.hipsparseSpGEMMDescr`

    hipsparseSpSVDescr_t:
        alias of `~.hipsparseSpSVDescr`

    hipsparseSpSMDescr_t:
        alias of `~.hipsparseSpSMDescr`

"""

import cython
import ctypes
import enum
cdef class bsrsv2Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef bsrsv2Info from_ptr(chipsparse.bsrsv2Info* ptr, bint owner=False):
        """Factory function to create ``bsrsv2Info`` objects from
        given ``chipsparse.bsrsv2Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef bsrsv2Info wrapper = bsrsv2Info.__new__(bsrsv2Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef bsrsv2Info from_pyobj(object pyobj):
        """Derives a bsrsv2Info from a Python object.

        Derives a bsrsv2Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``bsrsv2Info`` reference, this method
        returns it directly. No new ``bsrsv2Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `bsrsv2Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of bsrsv2Info!
        """
        cdef bsrsv2Info wrapper = bsrsv2Info.__new__(bsrsv2Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,bsrsv2Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.bsrsv2Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.bsrsv2Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.bsrsv2Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.bsrsv2Info*>wrapper._py_buffer.buf
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
        return f"<bsrsv2Info object, self.ptr={int(self)}>"
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


bsrsv2Info_t = bsrsv2Info

cdef class bsrsm2Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef bsrsm2Info from_ptr(chipsparse.bsrsm2Info* ptr, bint owner=False):
        """Factory function to create ``bsrsm2Info`` objects from
        given ``chipsparse.bsrsm2Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef bsrsm2Info wrapper = bsrsm2Info.__new__(bsrsm2Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef bsrsm2Info from_pyobj(object pyobj):
        """Derives a bsrsm2Info from a Python object.

        Derives a bsrsm2Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``bsrsm2Info`` reference, this method
        returns it directly. No new ``bsrsm2Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `bsrsm2Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of bsrsm2Info!
        """
        cdef bsrsm2Info wrapper = bsrsm2Info.__new__(bsrsm2Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,bsrsm2Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.bsrsm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.bsrsm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.bsrsm2Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.bsrsm2Info*>wrapper._py_buffer.buf
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
        return f"<bsrsm2Info object, self.ptr={int(self)}>"
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


bsrsm2Info_t = bsrsm2Info

cdef class bsrilu02Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef bsrilu02Info from_ptr(chipsparse.bsrilu02Info* ptr, bint owner=False):
        """Factory function to create ``bsrilu02Info`` objects from
        given ``chipsparse.bsrilu02Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef bsrilu02Info wrapper = bsrilu02Info.__new__(bsrilu02Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef bsrilu02Info from_pyobj(object pyobj):
        """Derives a bsrilu02Info from a Python object.

        Derives a bsrilu02Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``bsrilu02Info`` reference, this method
        returns it directly. No new ``bsrilu02Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `bsrilu02Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of bsrilu02Info!
        """
        cdef bsrilu02Info wrapper = bsrilu02Info.__new__(bsrilu02Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,bsrilu02Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.bsrilu02Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.bsrilu02Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.bsrilu02Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.bsrilu02Info*>wrapper._py_buffer.buf
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
        return f"<bsrilu02Info object, self.ptr={int(self)}>"
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


bsrilu02Info_t = bsrilu02Info

cdef class bsric02Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef bsric02Info from_ptr(chipsparse.bsric02Info* ptr, bint owner=False):
        """Factory function to create ``bsric02Info`` objects from
        given ``chipsparse.bsric02Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef bsric02Info wrapper = bsric02Info.__new__(bsric02Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef bsric02Info from_pyobj(object pyobj):
        """Derives a bsric02Info from a Python object.

        Derives a bsric02Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``bsric02Info`` reference, this method
        returns it directly. No new ``bsric02Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `bsric02Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of bsric02Info!
        """
        cdef bsric02Info wrapper = bsric02Info.__new__(bsric02Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,bsric02Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.bsric02Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.bsric02Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.bsric02Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.bsric02Info*>wrapper._py_buffer.buf
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
        return f"<bsric02Info object, self.ptr={int(self)}>"
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


bsric02Info_t = bsric02Info

cdef class csrsv2Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csrsv2Info from_ptr(chipsparse.csrsv2Info* ptr, bint owner=False):
        """Factory function to create ``csrsv2Info`` objects from
        given ``chipsparse.csrsv2Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csrsv2Info wrapper = csrsv2Info.__new__(csrsv2Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csrsv2Info from_pyobj(object pyobj):
        """Derives a csrsv2Info from a Python object.

        Derives a csrsv2Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csrsv2Info`` reference, this method
        returns it directly. No new ``csrsv2Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csrsv2Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csrsv2Info!
        """
        cdef csrsv2Info wrapper = csrsv2Info.__new__(csrsv2Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csrsv2Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csrsv2Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csrsv2Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csrsv2Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csrsv2Info*>wrapper._py_buffer.buf
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
        return f"<csrsv2Info object, self.ptr={int(self)}>"
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


csrsv2Info_t = csrsv2Info

cdef class csrsm2Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csrsm2Info from_ptr(chipsparse.csrsm2Info* ptr, bint owner=False):
        """Factory function to create ``csrsm2Info`` objects from
        given ``chipsparse.csrsm2Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csrsm2Info wrapper = csrsm2Info.__new__(csrsm2Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csrsm2Info from_pyobj(object pyobj):
        """Derives a csrsm2Info from a Python object.

        Derives a csrsm2Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csrsm2Info`` reference, this method
        returns it directly. No new ``csrsm2Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csrsm2Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csrsm2Info!
        """
        cdef csrsm2Info wrapper = csrsm2Info.__new__(csrsm2Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csrsm2Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csrsm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csrsm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csrsm2Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csrsm2Info*>wrapper._py_buffer.buf
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
        return f"<csrsm2Info object, self.ptr={int(self)}>"
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


csrsm2Info_t = csrsm2Info

cdef class csrilu02Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csrilu02Info from_ptr(chipsparse.csrilu02Info* ptr, bint owner=False):
        """Factory function to create ``csrilu02Info`` objects from
        given ``chipsparse.csrilu02Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csrilu02Info wrapper = csrilu02Info.__new__(csrilu02Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csrilu02Info from_pyobj(object pyobj):
        """Derives a csrilu02Info from a Python object.

        Derives a csrilu02Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csrilu02Info`` reference, this method
        returns it directly. No new ``csrilu02Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csrilu02Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csrilu02Info!
        """
        cdef csrilu02Info wrapper = csrilu02Info.__new__(csrilu02Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csrilu02Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csrilu02Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csrilu02Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csrilu02Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csrilu02Info*>wrapper._py_buffer.buf
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
        return f"<csrilu02Info object, self.ptr={int(self)}>"
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


csrilu02Info_t = csrilu02Info

cdef class csric02Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csric02Info from_ptr(chipsparse.csric02Info* ptr, bint owner=False):
        """Factory function to create ``csric02Info`` objects from
        given ``chipsparse.csric02Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csric02Info wrapper = csric02Info.__new__(csric02Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csric02Info from_pyobj(object pyobj):
        """Derives a csric02Info from a Python object.

        Derives a csric02Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csric02Info`` reference, this method
        returns it directly. No new ``csric02Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csric02Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csric02Info!
        """
        cdef csric02Info wrapper = csric02Info.__new__(csric02Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csric02Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csric02Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csric02Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csric02Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csric02Info*>wrapper._py_buffer.buf
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
        return f"<csric02Info object, self.ptr={int(self)}>"
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


csric02Info_t = csric02Info

cdef class csrgemm2Info:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csrgemm2Info from_ptr(chipsparse.csrgemm2Info* ptr, bint owner=False):
        """Factory function to create ``csrgemm2Info`` objects from
        given ``chipsparse.csrgemm2Info`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csrgemm2Info wrapper = csrgemm2Info.__new__(csrgemm2Info)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csrgemm2Info from_pyobj(object pyobj):
        """Derives a csrgemm2Info from a Python object.

        Derives a csrgemm2Info from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csrgemm2Info`` reference, this method
        returns it directly. No new ``csrgemm2Info`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csrgemm2Info`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csrgemm2Info!
        """
        cdef csrgemm2Info wrapper = csrgemm2Info.__new__(csrgemm2Info)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csrgemm2Info):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csrgemm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csrgemm2Info*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csrgemm2Info*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csrgemm2Info*>wrapper._py_buffer.buf
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
        return f"<csrgemm2Info object, self.ptr={int(self)}>"
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


csrgemm2Info_t = csrgemm2Info

cdef class pruneInfo:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef pruneInfo from_ptr(chipsparse.pruneInfo* ptr, bint owner=False):
        """Factory function to create ``pruneInfo`` objects from
        given ``chipsparse.pruneInfo`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef pruneInfo wrapper = pruneInfo.__new__(pruneInfo)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef pruneInfo from_pyobj(object pyobj):
        """Derives a pruneInfo from a Python object.

        Derives a pruneInfo from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``pruneInfo`` reference, this method
        returns it directly. No new ``pruneInfo`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `pruneInfo`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of pruneInfo!
        """
        cdef pruneInfo wrapper = pruneInfo.__new__(pruneInfo)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,pruneInfo):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.pruneInfo*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.pruneInfo*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.pruneInfo*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.pruneInfo*>wrapper._py_buffer.buf
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
        return f"<pruneInfo object, self.ptr={int(self)}>"
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


pruneInfo_t = pruneInfo

cdef class csru2csrInfo:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef csru2csrInfo from_ptr(chipsparse.csru2csrInfo* ptr, bint owner=False):
        """Factory function to create ``csru2csrInfo`` objects from
        given ``chipsparse.csru2csrInfo`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef csru2csrInfo wrapper = csru2csrInfo.__new__(csru2csrInfo)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef csru2csrInfo from_pyobj(object pyobj):
        """Derives a csru2csrInfo from a Python object.

        Derives a csru2csrInfo from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``csru2csrInfo`` reference, this method
        returns it directly. No new ``csru2csrInfo`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `csru2csrInfo`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of csru2csrInfo!
        """
        cdef csru2csrInfo wrapper = csru2csrInfo.__new__(csru2csrInfo)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,csru2csrInfo):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.csru2csrInfo*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.csru2csrInfo*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.csru2csrInfo*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.csru2csrInfo*>wrapper._py_buffer.buf
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
        return f"<csru2csrInfo object, self.ptr={int(self)}>"
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


csru2csrInfo_t = csru2csrInfo

class _hipsparseStatus_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseStatus_t(_hipsparseStatus_t__Base):
    HIPSPARSE_STATUS_SUCCESS = chipsparse.HIPSPARSE_STATUS_SUCCESS
    HIPSPARSE_STATUS_NOT_INITIALIZED = chipsparse.HIPSPARSE_STATUS_NOT_INITIALIZED
    HIPSPARSE_STATUS_ALLOC_FAILED = chipsparse.HIPSPARSE_STATUS_ALLOC_FAILED
    HIPSPARSE_STATUS_INVALID_VALUE = chipsparse.HIPSPARSE_STATUS_INVALID_VALUE
    HIPSPARSE_STATUS_ARCH_MISMATCH = chipsparse.HIPSPARSE_STATUS_ARCH_MISMATCH
    HIPSPARSE_STATUS_MAPPING_ERROR = chipsparse.HIPSPARSE_STATUS_MAPPING_ERROR
    HIPSPARSE_STATUS_EXECUTION_FAILED = chipsparse.HIPSPARSE_STATUS_EXECUTION_FAILED
    HIPSPARSE_STATUS_INTERNAL_ERROR = chipsparse.HIPSPARSE_STATUS_INTERNAL_ERROR
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = chipsparse.HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
    HIPSPARSE_STATUS_ZERO_PIVOT = chipsparse.HIPSPARSE_STATUS_ZERO_PIVOT
    HIPSPARSE_STATUS_NOT_SUPPORTED = chipsparse.HIPSPARSE_STATUS_NOT_SUPPORTED
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES = chipsparse.HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparsePointerMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparsePointerMode_t(_hipsparsePointerMode_t__Base):
    HIPSPARSE_POINTER_MODE_HOST = chipsparse.HIPSPARSE_POINTER_MODE_HOST
    HIPSPARSE_POINTER_MODE_DEVICE = chipsparse.HIPSPARSE_POINTER_MODE_DEVICE
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseAction_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseAction_t(_hipsparseAction_t__Base):
    HIPSPARSE_ACTION_SYMBOLIC = chipsparse.HIPSPARSE_ACTION_SYMBOLIC
    HIPSPARSE_ACTION_NUMERIC = chipsparse.HIPSPARSE_ACTION_NUMERIC
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseMatrixType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseMatrixType_t(_hipsparseMatrixType_t__Base):
    HIPSPARSE_MATRIX_TYPE_GENERAL = chipsparse.HIPSPARSE_MATRIX_TYPE_GENERAL
    HIPSPARSE_MATRIX_TYPE_SYMMETRIC = chipsparse.HIPSPARSE_MATRIX_TYPE_SYMMETRIC
    HIPSPARSE_MATRIX_TYPE_HERMITIAN = chipsparse.HIPSPARSE_MATRIX_TYPE_HERMITIAN
    HIPSPARSE_MATRIX_TYPE_TRIANGULAR = chipsparse.HIPSPARSE_MATRIX_TYPE_TRIANGULAR
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseFillMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseFillMode_t(_hipsparseFillMode_t__Base):
    HIPSPARSE_FILL_MODE_LOWER = chipsparse.HIPSPARSE_FILL_MODE_LOWER
    HIPSPARSE_FILL_MODE_UPPER = chipsparse.HIPSPARSE_FILL_MODE_UPPER
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseDiagType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseDiagType_t(_hipsparseDiagType_t__Base):
    HIPSPARSE_DIAG_TYPE_NON_UNIT = chipsparse.HIPSPARSE_DIAG_TYPE_NON_UNIT
    HIPSPARSE_DIAG_TYPE_UNIT = chipsparse.HIPSPARSE_DIAG_TYPE_UNIT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseIndexBase_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseIndexBase_t(_hipsparseIndexBase_t__Base):
    HIPSPARSE_INDEX_BASE_ZERO = chipsparse.HIPSPARSE_INDEX_BASE_ZERO
    HIPSPARSE_INDEX_BASE_ONE = chipsparse.HIPSPARSE_INDEX_BASE_ONE
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseOperation_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseOperation_t(_hipsparseOperation_t__Base):
    HIPSPARSE_OPERATION_NON_TRANSPOSE = chipsparse.HIPSPARSE_OPERATION_NON_TRANSPOSE
    HIPSPARSE_OPERATION_TRANSPOSE = chipsparse.HIPSPARSE_OPERATION_TRANSPOSE
    HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = chipsparse.HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseHybPartition_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseHybPartition_t(_hipsparseHybPartition_t__Base):
    HIPSPARSE_HYB_PARTITION_AUTO = chipsparse.HIPSPARSE_HYB_PARTITION_AUTO
    HIPSPARSE_HYB_PARTITION_USER = chipsparse.HIPSPARSE_HYB_PARTITION_USER
    HIPSPARSE_HYB_PARTITION_MAX = chipsparse.HIPSPARSE_HYB_PARTITION_MAX
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSolvePolicy_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSolvePolicy_t(_hipsparseSolvePolicy_t__Base):
    HIPSPARSE_SOLVE_POLICY_NO_LEVEL = chipsparse.HIPSPARSE_SOLVE_POLICY_NO_LEVEL
    HIPSPARSE_SOLVE_POLICY_USE_LEVEL = chipsparse.HIPSPARSE_SOLVE_POLICY_USE_LEVEL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSideMode_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSideMode_t(_hipsparseSideMode_t__Base):
    HIPSPARSE_SIDE_LEFT = chipsparse.HIPSPARSE_SIDE_LEFT
    HIPSPARSE_SIDE_RIGHT = chipsparse.HIPSPARSE_SIDE_RIGHT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseDirection_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseDirection_t(_hipsparseDirection_t__Base):
    HIPSPARSE_DIRECTION_ROW = chipsparse.HIPSPARSE_DIRECTION_ROW
    HIPSPARSE_DIRECTION_COLUMN = chipsparse.HIPSPARSE_DIRECTION_COLUMN
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def hipsparseCreate(object handle):
    r"""Create a hipsparse handle

    ``hipsparseCreate`` creates the hipSPARSE library context. It must be
    initialized before any other hipSPARSE API function is invoked and must be passed to
    all subsequent library function calls. The handle should be destroyed at the end
    using hipsparseDestroy().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCreate__retval = hipsparseStatus_t(chipsparse.hipsparseCreate(
        <void **>hip._util.types.Pointer.from_pyobj(handle)._ptr))    # fully specified
    return (_hipsparseCreate__retval,)


@cython.embedsignature(True)
def hipsparseDestroy(object handle):
    r"""Destroy a hipsparse handle

    ``hipsparseDestroy`` destroys the hipSPARSE library context and releases all
    resources used by the hipSPARSE library.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroy__retval = hipsparseStatus_t(chipsparse.hipsparseDestroy(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr))    # fully specified
    return (_hipsparseDestroy__retval,)


@cython.embedsignature(True)
def hipsparseGetVersion(object handle, object version):
    r"""Get hipSPARSE version

    ``hipsparseGetVersion`` gets the hipSPARSE library version number.
    - patch = version % 100
    - minor = version / 100 % 1000
    - major = version / 100000

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseGetVersion__retval = hipsparseStatus_t(chipsparse.hipsparseGetVersion(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(version)._ptr))    # fully specified
    return (_hipsparseGetVersion__retval,)


@cython.embedsignature(True)
def hipsparseGetGitRevision(object handle, char * rev):
    r"""Get hipSPARSE git revision

    ``hipsparseGetGitRevision`` gets the hipSPARSE library git commit revision (SHA-1).

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseGetGitRevision__retval = hipsparseStatus_t(chipsparse.hipsparseGetGitRevision(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,rev))    # fully specified
    return (_hipsparseGetGitRevision__retval,)


@cython.embedsignature(True)
def hipsparseSetStream(object handle, object streamId):
    r"""Specify user defined HIP stream

    ``hipsparseSetStream`` specifies the stream to be used by the hipSPARSE library
    context and all subsequent function calls.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSetStream__retval = hipsparseStatus_t(chipsparse.hipsparseSetStream(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        ihipStream_t.from_pyobj(streamId)._ptr))    # fully specified
    return (_hipsparseSetStream__retval,)


@cython.embedsignature(True)
def hipsparseGetStream(object handle):
    r"""Get current stream from library context

    ``hipsparseGetStream`` gets the hipSPARSE library context stream which is currently
    used for all subsequent function calls.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    streamId = ihipStream_t.from_ptr(NULL)
    _hipsparseGetStream__retval = hipsparseStatus_t(chipsparse.hipsparseGetStream(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,&streamId._ptr))    # fully specified
    return (_hipsparseGetStream__retval,streamId)


@cython.embedsignature(True)
def hipsparseSetPointerMode(object handle, object mode):
    r"""Specify pointer mode

    ``hipsparseSetPointerMode`` specifies the pointer mode to be used by the hipSPARSE
    library context and all subsequent function calls. By default, all values are passed
    by reference on the host. Valid pointer modes are ``HIPSPARSE_POINTER_MODE_HOST`` 
    or ``HIPSPARSE_POINTER_MODE_DEVICE.``

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(mode,_hipsparsePointerMode_t__Base):
        raise TypeError("argument 'mode' must be of type '_hipsparsePointerMode_t__Base'")
    _hipsparseSetPointerMode__retval = hipsparseStatus_t(chipsparse.hipsparseSetPointerMode(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mode.value))    # fully specified
    return (_hipsparseSetPointerMode__retval,)


@cython.embedsignature(True)
def hipsparseGetPointerMode(object handle, object mode):
    r"""Get current pointer mode from library context

    ``hipsparseGetPointerMode`` gets the hipSPARSE library context pointer mode which
    is currently used for all subsequent function calls.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseGetPointerMode__retval = hipsparseStatus_t(chipsparse.hipsparseGetPointerMode(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <chipsparse.hipsparsePointerMode_t *>hip._util.types.Pointer.from_pyobj(mode)._ptr))    # fully specified
    return (_hipsparseGetPointerMode__retval,)


@cython.embedsignature(True)
def hipsparseCreateMatDescr(object descrA):
    r"""Create a matrix descriptor

    ``hipsparseCreateMatDescr`` creates a matrix descriptor. It initializes
    ``hipsparseMatrixType_t`` to ``HIPSPARSE_MATRIX_TYPE_GENERAL`` and
    ``hipsparseIndexBase_t`` to ``HIPSPARSE_INDEX_BASE_ZERO`` . It should be destroyed
    at the end using hipsparseDestroyMatDescr().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCreateMatDescr__retval = hipsparseStatus_t(chipsparse.hipsparseCreateMatDescr(
        <void **>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseCreateMatDescr__retval,)


@cython.embedsignature(True)
def hipsparseDestroyMatDescr(object descrA):
    r"""Destroy a matrix descriptor

    ``hipsparseDestroyMatDescr`` destroys a matrix descriptor and releases all
    resources used by the descriptor.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyMatDescr__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyMatDescr(
        <void *>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseDestroyMatDescr__retval,)


@cython.embedsignature(True)
def hipsparseCopyMatDescr(object dest, object src):
    r"""Copy a matrix descriptor

    ``hipsparseCopyMatDescr`` copies a matrix descriptor. Both, source and destination
    matrix descriptors must be initialized prior to calling ``hipsparseCopyMatDescr.``

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCopyMatDescr__retval = hipsparseStatus_t(chipsparse.hipsparseCopyMatDescr(
        <void *>hip._util.types.Pointer.from_pyobj(dest)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(src)._ptr))    # fully specified
    return (_hipsparseCopyMatDescr__retval,)


@cython.embedsignature(True)
def hipsparseSetMatType(object descrA, object type):
    r"""Specify the matrix type of a matrix descriptor

    ``hipsparseSetMatType`` sets the matrix type of a matrix descriptor. Valid
    matrix types are ``HIPSPARSE_MATRIX_TYPE_GENERAL`` ,
    ``HIPSPARSE_MATRIX_TYPE_SYMMETRIC`` , ``HIPSPARSE_MATRIX_TYPE_HERMITIAN`` or
    ``HIPSPARSE_MATRIX_TYPE_TRIANGULAR`` .

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(type,_hipsparseMatrixType_t__Base):
        raise TypeError("argument 'type' must be of type '_hipsparseMatrixType_t__Base'")
    _hipsparseSetMatType__retval = hipsparseStatus_t(chipsparse.hipsparseSetMatType(
        <void *>hip._util.types.Pointer.from_pyobj(descrA)._ptr,type.value))    # fully specified
    return (_hipsparseSetMatType__retval,)


@cython.embedsignature(True)
def hipsparseGetMatType(object descrA):
    r"""Get the matrix type of a matrix descriptor

    ``hipsparseGetMatType`` returns the matrix type of a matrix descriptor.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseMatrixType_t`
    """
    _hipsparseGetMatType__retval = hipsparseMatrixType_t(chipsparse.hipsparseGetMatType(
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseGetMatType__retval,)


@cython.embedsignature(True)
def hipsparseSetMatFillMode(object descrA, object fillMode):
    r"""Specify the matrix fill mode of a matrix descriptor

    ``hipsparseSetMatFillMode`` sets the matrix fill mode of a matrix descriptor.
    Valid fill modes are ``HIPSPARSE_FILL_MODE_LOWER`` or
    ``HIPSPARSE_FILL_MODE_UPPER`` .

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(fillMode,_hipsparseFillMode_t__Base):
        raise TypeError("argument 'fillMode' must be of type '_hipsparseFillMode_t__Base'")
    _hipsparseSetMatFillMode__retval = hipsparseStatus_t(chipsparse.hipsparseSetMatFillMode(
        <void *>hip._util.types.Pointer.from_pyobj(descrA)._ptr,fillMode.value))    # fully specified
    return (_hipsparseSetMatFillMode__retval,)


@cython.embedsignature(True)
def hipsparseGetMatFillMode(object descrA):
    r"""Get the matrix fill mode of a matrix descriptor

    ``hipsparseGetMatFillMode`` returns the matrix fill mode of a matrix descriptor.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseFillMode_t`
    """
    _hipsparseGetMatFillMode__retval = hipsparseFillMode_t(chipsparse.hipsparseGetMatFillMode(
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseGetMatFillMode__retval,)


@cython.embedsignature(True)
def hipsparseSetMatDiagType(object descrA, object diagType):
    r"""Specify the matrix diagonal type of a matrix descriptor

    ``hipsparseSetMatDiagType`` sets the matrix diagonal type of a matrix
    descriptor. Valid diagonal types are ``HIPSPARSE_DIAG_TYPE_UNIT`` or
    ``HIPSPARSE_DIAG_TYPE_NON_UNIT`` .

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(diagType,_hipsparseDiagType_t__Base):
        raise TypeError("argument 'diagType' must be of type '_hipsparseDiagType_t__Base'")
    _hipsparseSetMatDiagType__retval = hipsparseStatus_t(chipsparse.hipsparseSetMatDiagType(
        <void *>hip._util.types.Pointer.from_pyobj(descrA)._ptr,diagType.value))    # fully specified
    return (_hipsparseSetMatDiagType__retval,)


@cython.embedsignature(True)
def hipsparseGetMatDiagType(object descrA):
    r"""Get the matrix diagonal type of a matrix descriptor

    ``hipsparseGetMatDiagType`` returns the matrix diagonal type of a matrix
    descriptor.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseDiagType_t`
    """
    _hipsparseGetMatDiagType__retval = hipsparseDiagType_t(chipsparse.hipsparseGetMatDiagType(
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseGetMatDiagType__retval,)


@cython.embedsignature(True)
def hipsparseSetMatIndexBase(object descrA, object base):
    r"""Specify the index base of a matrix descriptor

    ``hipsparseSetMatIndexBase`` sets the index base of a matrix descriptor. Valid
    options are ``HIPSPARSE_INDEX_BASE_ZERO`` or ``HIPSPARSE_INDEX_BASE_ONE`` .

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(base,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'base' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSetMatIndexBase__retval = hipsparseStatus_t(chipsparse.hipsparseSetMatIndexBase(
        <void *>hip._util.types.Pointer.from_pyobj(descrA)._ptr,base.value))    # fully specified
    return (_hipsparseSetMatIndexBase__retval,)


@cython.embedsignature(True)
def hipsparseGetMatIndexBase(object descrA):
    r"""Get the index base of a matrix descriptor

    ``hipsparseGetMatIndexBase`` returns the index base of a matrix descriptor.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseIndexBase_t`
    """
    _hipsparseGetMatIndexBase__retval = hipsparseIndexBase_t(chipsparse.hipsparseGetMatIndexBase(
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr))    # fully specified
    return (_hipsparseGetMatIndexBase__retval,)


@cython.embedsignature(True)
def hipsparseCreateHybMat(object hybA):
    r"""Create a ``HYB`` matrix structure

    ``hipsparseCreateHybMat`` creates a structure that holds the matrix in ``HYB``
    storage format. It should be destroyed at the end using hipsparseDestroyHybMat().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCreateHybMat__retval = hipsparseStatus_t(chipsparse.hipsparseCreateHybMat(
        <void **>hip._util.types.Pointer.from_pyobj(hybA)._ptr))    # fully specified
    return (_hipsparseCreateHybMat__retval,)


@cython.embedsignature(True)
def hipsparseDestroyHybMat(object hybA):
    r"""Destroy a ``HYB`` matrix structure

    ``hipsparseDestroyHybMat`` destroys a ``HYB`` structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyHybMat__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyHybMat(
        <void *>hip._util.types.Pointer.from_pyobj(hybA)._ptr))    # fully specified
    return (_hipsparseDestroyHybMat__retval,)


@cython.embedsignature(True)
def hipsparseCreateBsrsv2Info():
    r"""Create a bsrsv2 info structure

    ``hipsparseCreateBsrsv2Info`` creates a structure that holds the bsrsv2 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyBsrsv2Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = bsrsv2Info.from_ptr(NULL)
    _hipsparseCreateBsrsv2Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateBsrsv2Info(&info._ptr))    # fully specified
    return (_hipsparseCreateBsrsv2Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyBsrsv2Info(object info):
    r"""Destroy a bsrsv2 info structure

    ``hipsparseDestroyBsrsv2Info`` destroys a bsrsv2 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyBsrsv2Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyBsrsv2Info(
        bsrsv2Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyBsrsv2Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateBsrsm2Info():
    r"""Create a bsrsm2 info structure

    ``hipsparseCreateBsrsm2Info`` creates a structure that holds the bsrsm2 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyBsrsm2Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = bsrsm2Info.from_ptr(NULL)
    _hipsparseCreateBsrsm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateBsrsm2Info(&info._ptr))    # fully specified
    return (_hipsparseCreateBsrsm2Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyBsrsm2Info(object info):
    r"""Destroy a bsrsm2 info structure

    ``hipsparseDestroyBsrsm2Info`` destroys a bsrsm2 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyBsrsm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyBsrsm2Info(
        bsrsm2Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyBsrsm2Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateBsrilu02Info():
    r"""Create a bsrilu02 info structure

    ``hipsparseCreateBsrilu02Info`` creates a structure that holds the bsrilu02 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyBsrilu02Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = bsrilu02Info.from_ptr(NULL)
    _hipsparseCreateBsrilu02Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateBsrilu02Info(&info._ptr))    # fully specified
    return (_hipsparseCreateBsrilu02Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyBsrilu02Info(object info):
    r"""Destroy a bsrilu02 info structure

    ``hipsparseDestroyBsrilu02Info`` destroys a bsrilu02 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyBsrilu02Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyBsrilu02Info(
        bsrilu02Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyBsrilu02Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateBsric02Info():
    r"""Create a bsric02 info structure

    ``hipsparseCreateBsric02Info`` creates a structure that holds the bsric02 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyBsric02Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = bsric02Info.from_ptr(NULL)
    _hipsparseCreateBsric02Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateBsric02Info(&info._ptr))    # fully specified
    return (_hipsparseCreateBsric02Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyBsric02Info(object info):
    r"""Destroy a bsric02 info structure

    ``hipsparseDestroyBsric02Info`` destroys a bsric02 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyBsric02Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyBsric02Info(
        bsric02Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyBsric02Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsrsv2Info():
    r"""Create a csrsv2 info structure

    ``hipsparseCreateCsrsv2Info`` creates a structure that holds the csrsv2 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsrsv2Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csrsv2Info.from_ptr(NULL)
    _hipsparseCreateCsrsv2Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsrsv2Info(&info._ptr))    # fully specified
    return (_hipsparseCreateCsrsv2Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsrsv2Info(object info):
    r"""Destroy a csrsv2 info structure

    ``hipsparseDestroyCsrsv2Info`` destroys a csrsv2 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsrsv2Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsrsv2Info(
        csrsv2Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsrsv2Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsrsm2Info():
    r"""Create a csrsm2 info structure

    ``hipsparseCreateCsrsm2Info`` creates a structure that holds the csrsm2 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsrsm2Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csrsm2Info.from_ptr(NULL)
    _hipsparseCreateCsrsm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsrsm2Info(&info._ptr))    # fully specified
    return (_hipsparseCreateCsrsm2Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsrsm2Info(object info):
    r"""Destroy a csrsm2 info structure

    ``hipsparseDestroyCsrsm2Info`` destroys a csrsm2 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsrsm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsrsm2Info(
        csrsm2Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsrsm2Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsrilu02Info():
    r"""Create a csrilu02 info structure

    ``hipsparseCreateCsrilu02Info`` creates a structure that holds the csrilu02 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsrilu02Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csrilu02Info.from_ptr(NULL)
    _hipsparseCreateCsrilu02Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsrilu02Info(&info._ptr))    # fully specified
    return (_hipsparseCreateCsrilu02Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsrilu02Info(object info):
    r"""Destroy a csrilu02 info structure

    ``hipsparseDestroyCsrilu02Info`` destroys a csrilu02 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsrilu02Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsrilu02Info(
        csrilu02Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsrilu02Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsric02Info():
    r"""Create a csric02 info structure

    ``hipsparseCreateCsric02Info`` creates a structure that holds the csric02 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsric02Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csric02Info.from_ptr(NULL)
    _hipsparseCreateCsric02Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsric02Info(&info._ptr))    # fully specified
    return (_hipsparseCreateCsric02Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsric02Info(object info):
    r"""Destroy a csric02 info structure

    ``hipsparseDestroyCsric02Info`` destroys a csric02 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsric02Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsric02Info(
        csric02Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsric02Info__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsru2csrInfo():
    r"""Create a csru2csr info structure

    ``hipsparseCreateCsru2csrInfo`` creates a structure that holds the csru2csr info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsru2csrInfo().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csru2csrInfo.from_ptr(NULL)
    _hipsparseCreateCsru2csrInfo__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsru2csrInfo(&info._ptr))    # fully specified
    return (_hipsparseCreateCsru2csrInfo__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsru2csrInfo(object info):
    r"""Destroy a csru2csr info structure

    ``hipsparseDestroyCsru2csrInfo`` destroys a csru2csr info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsru2csrInfo__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsru2csrInfo(
        csru2csrInfo.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsru2csrInfo__retval,)


@cython.embedsignature(True)
def hipsparseCreateColorInfo(object info):
    r"""Create a color info structure

    ``hipsparseCreateColorInfo`` creates a structure that holds the color info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyColorInfo().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCreateColorInfo__retval = hipsparseStatus_t(chipsparse.hipsparseCreateColorInfo(
        <void **>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseCreateColorInfo__retval,)


@cython.embedsignature(True)
def hipsparseDestroyColorInfo(object info):
    r"""Destroy a color info structure

    ``hipsparseDestroyColorInfo`` destroys a color info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyColorInfo__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyColorInfo(
        <void *>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyColorInfo__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsrgemm2Info():
    r"""Create a csrgemm2 info structure

    ``hipsparseCreateCsrgemm2Info`` creates a structure that holds the csrgemm2 info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyCsrgemm2Info().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = csrgemm2Info.from_ptr(NULL)
    _hipsparseCreateCsrgemm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsrgemm2Info(&info._ptr))    # fully specified
    return (_hipsparseCreateCsrgemm2Info__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyCsrgemm2Info(object info):
    r"""Destroy a csrgemm2 info structure

    ``hipsparseDestroyCsrgemm2Info`` destroys a csrgemm2 info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyCsrgemm2Info__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyCsrgemm2Info(
        csrgemm2Info.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyCsrgemm2Info__retval,)


@cython.embedsignature(True)
def hipsparseCreatePruneInfo():
    r"""Create a prune info structure

    ``hipsparseCreatePruneInfo`` creates a structure that holds the prune info data
    that is gathered during the analysis routines available. It should be destroyed
    at the end using hipsparseDestroyPruneInfo().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    info = pruneInfo.from_ptr(NULL)
    _hipsparseCreatePruneInfo__retval = hipsparseStatus_t(chipsparse.hipsparseCreatePruneInfo(&info._ptr))    # fully specified
    return (_hipsparseCreatePruneInfo__retval,info)


@cython.embedsignature(True)
def hipsparseDestroyPruneInfo(object info):
    r"""Destroy a prune info structure

    ``hipsparseDestroyPruneInfo`` destroys a prune info structure.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyPruneInfo__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyPruneInfo(
        pruneInfo.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDestroyPruneInfo__retval,)


@cython.embedsignature(True)
def hipsparseSaxpyi(object handle, int nnz, object alpha, object xVal, object xInd, object y, object idxBase):
    r"""Scale a sparse vector and add it to a dense vector.

    ``hipsparseXaxpyi`` multiplies the sparse vector :math:`x` with scalar :math:`\alpha` and
    adds the result to the dense vector :math:`y`, such that

    .. math::

       y := y + \alpha \cdot x

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSaxpyi__retval = hipsparseStatus_t(chipsparse.hipsparseSaxpyi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSaxpyi__retval,)


@cython.embedsignature(True)
def hipsparseDaxpyi(object handle, int nnz, object alpha, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDaxpyi__retval = hipsparseStatus_t(chipsparse.hipsparseDaxpyi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDaxpyi__retval,)


@cython.embedsignature(True)
def hipsparseCaxpyi(object handle, int nnz, object alpha, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCaxpyi__retval = hipsparseStatus_t(chipsparse.hipsparseCaxpyi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(alpha)._ptr,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        float2.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCaxpyi__retval,)


@cython.embedsignature(True)
def hipsparseZaxpyi(object handle, int nnz, object alpha, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZaxpyi__retval = hipsparseStatus_t(chipsparse.hipsparseZaxpyi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(alpha)._ptr,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        double2.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZaxpyi__retval,)


@cython.embedsignature(True)
def hipsparseSdoti(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""Compute the dot product of a sparse vector with a dense vector.

    ``hipsparseXdoti`` computes the dot product of the sparse vector :math:`x` with the
    dense vector :math:`y`, such that

    .. math::

       \text{result} := y^T x

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           result += x_val[i] * y[x_ind[i]];
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSdoti__retval = hipsparseStatus_t(chipsparse.hipsparseSdoti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSdoti__retval,)


@cython.embedsignature(True)
def hipsparseDdoti(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDdoti__retval = hipsparseStatus_t(chipsparse.hipsparseDdoti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDdoti__retval,)


@cython.embedsignature(True)
def hipsparseCdoti(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCdoti__retval = hipsparseStatus_t(chipsparse.hipsparseCdoti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        float2.from_pyobj(y)._ptr,
        float2.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCdoti__retval,)


@cython.embedsignature(True)
def hipsparseZdoti(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZdoti__retval = hipsparseStatus_t(chipsparse.hipsparseZdoti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        double2.from_pyobj(y)._ptr,
        double2.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZdoti__retval,)


@cython.embedsignature(True)
def hipsparseCdotci(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""Compute the dot product of a complex conjugate sparse vector with a dense
    vector.

    ``hipsparseXdotci`` computes the dot product of the complex conjugate sparse vector
    :math:`x` with the dense vector :math:`y`, such that

    .. math::

       \text{result} := \bar{x}^H y

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           result += conj(x_val[i]) * y[x_ind[i]];
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCdotci__retval = hipsparseStatus_t(chipsparse.hipsparseCdotci(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        float2.from_pyobj(y)._ptr,
        float2.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCdotci__retval,)


@cython.embedsignature(True)
def hipsparseZdotci(object handle, int nnz, object xVal, object xInd, object y, object result, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZdotci__retval = hipsparseStatus_t(chipsparse.hipsparseZdotci(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        double2.from_pyobj(y)._ptr,
        double2.from_pyobj(result)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZdotci__retval,)


@cython.embedsignature(True)
def hipsparseSgthr(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""Gather elements from a dense vector and store them into a sparse vector.

    ``hipsparseXgthr`` gathers the elements that are listed in ``x_ind`` from the dense
    vector :math:`y` and stores them in the sparse vector :math:`x`.

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           x_val[i] = y[x_ind[i]];
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSgthr__retval = hipsparseStatus_t(chipsparse.hipsparseSgthr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSgthr__retval,)


@cython.embedsignature(True)
def hipsparseDgthr(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDgthr__retval = hipsparseStatus_t(chipsparse.hipsparseDgthr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDgthr__retval,)


@cython.embedsignature(True)
def hipsparseCgthr(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCgthr__retval = hipsparseStatus_t(chipsparse.hipsparseCgthr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(y)._ptr,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCgthr__retval,)


@cython.embedsignature(True)
def hipsparseZgthr(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZgthr__retval = hipsparseStatus_t(chipsparse.hipsparseZgthr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(y)._ptr,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZgthr__retval,)


@cython.embedsignature(True)
def hipsparseSgthrz(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""Gather and zero out elements from a dense vector and store them into a sparse
    vector.

    ``hipsparseXgthrz`` gathers the elements that are listed in ``x_ind`` from the dense
    vector :math:`y` and stores them in the sparse vector :math:`x`. The gathered elements
    in :math:`y` are replaced by zero.

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           x_val[i]    = y[x_ind[i]];
           y[x_ind[i]] = 0;
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSgthrz__retval = hipsparseStatus_t(chipsparse.hipsparseSgthrz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSgthrz__retval,)


@cython.embedsignature(True)
def hipsparseDgthrz(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDgthrz__retval = hipsparseStatus_t(chipsparse.hipsparseDgthrz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDgthrz__retval,)


@cython.embedsignature(True)
def hipsparseCgthrz(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCgthrz__retval = hipsparseStatus_t(chipsparse.hipsparseCgthrz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(y)._ptr,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCgthrz__retval,)


@cython.embedsignature(True)
def hipsparseZgthrz(object handle, int nnz, object y, object xVal, object xInd, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZgthrz__retval = hipsparseStatus_t(chipsparse.hipsparseZgthrz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(y)._ptr,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZgthrz__retval,)


@cython.embedsignature(True)
def hipsparseSroti(object handle, int nnz, object xVal, object xInd, object y, object c, object s, object idxBase):
    r"""Apply Givens rotation to a dense and a sparse vector.

    ``hipsparseXroti`` applies the Givens rotation matrix :math:`G` to the sparse vector
    :math:`x` and the dense vector :math:`y`, where

    .. math::

       G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           x_tmp = x_val[i];
           y_tmp = y[x_ind[i]];

           x_val[i]    = c * x_tmp + s * y_tmp;
           y[x_ind[i]] = c * y_tmp - s * x_tmp;
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSroti__retval = hipsparseStatus_t(chipsparse.hipsparseSroti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(c)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(s)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSroti__retval,)


@cython.embedsignature(True)
def hipsparseDroti(object handle, int nnz, object xVal, object xInd, object y, object c, object s, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDroti__retval = hipsparseStatus_t(chipsparse.hipsparseDroti(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(c)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(s)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDroti__retval,)


@cython.embedsignature(True)
def hipsparseSsctr(object handle, int nnz, object xVal, object xInd, object y, object idxBase):
    r"""Scatter elements from a dense vector across a sparse vector.

    ``hipsparseXsctr`` scatters the elements that are listed in ``x_ind`` from the sparse
    vector :math:`x` into the dense vector :math:`y`. Indices of :math:`y` that are not listed
    in ``x_ind`` remain unchanged.

    .. code-block::

       for(i = 0; i < nnz; ++i)
       {
           y[x_ind[i]] = x_val[i];
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSsctr__retval = hipsparseStatus_t(chipsparse.hipsparseSsctr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseSsctr__retval,)


@cython.embedsignature(True)
def hipsparseDsctr(object handle, int nnz, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDsctr__retval = hipsparseStatus_t(chipsparse.hipsparseDsctr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseDsctr__retval,)


@cython.embedsignature(True)
def hipsparseCsctr(object handle, int nnz, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCsctr__retval = hipsparseStatus_t(chipsparse.hipsparseCsctr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        float2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        float2.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseCsctr__retval,)


@cython.embedsignature(True)
def hipsparseZsctr(object handle, int nnz, object xVal, object xInd, object y, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZsctr__retval = hipsparseStatus_t(chipsparse.hipsparseZsctr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,nnz,
        double2.from_pyobj(xVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        double2.from_pyobj(y)._ptr,idxBase.value))    # fully specified
    return (_hipsparseZsctr__retval,)


@cython.embedsignature(True)
def hipsparseScsrmv(object handle, object transA, int m, int n, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object x, object beta, object y):
    r"""Sparse matrix vector multiplication using CSR storage format

    ``hipsparseXcsrmv`` multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n`
    matrix, defined in CSR storage format, and the dense vector :math:`x` and adds the
    result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`,
    such that

    .. math::

       y := \alpha \cdot op(A) \cdot x + \beta \cdot y,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    .. code-block::

       for(i = 0; i < m; ++i)
       {
           y[i] = beta * y[i];

           for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
           {
               y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
           }
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseScsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseScsrmv__retval,)


@cython.embedsignature(True)
def hipsparseDcsrmv(object handle, object transA, int m, int n, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseDcsrmv__retval,)


@cython.embedsignature(True)
def hipsparseCcsrmv(object handle, object transA, int m, int n, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(x)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseCcsrmv__retval,)


@cython.embedsignature(True)
def hipsparseZcsrmv(object handle, object transA, int m, int n, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(x)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseZcsrmv__retval,)


@cython.embedsignature(True)
def hipsparseXcsrsv2_zeroPivot(object handle, object info, object position):
    r"""Sparse triangular solve using CSR storage format

    ``hipsparseXcsrsv2_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseScsrsv2_solve(),
    hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
    computation. The first zero pivot :math:`j` at :math:`A_{j,j}` is stored in ``position,``
    using same index base as the CSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXcsrsv2_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrsv2_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrsv2_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXcsrsv2_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseScsrsv2_bufferSize(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""Sparse triangular solve using CSR storage format

    ``hipsparseXcsrsv2_bufferSize`` returns the size of the temporary storage buffer that
    is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
    hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
    hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsv2_bufferSize(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsv2_bufferSize(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsv2_bufferSize(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseScsrsv2_bufferSizeExt(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""Sparse triangular solve using CSR storage format

    ``hipsparseXcsrsv2_bufferSizeExt`` returns the size of the temporary storage buffer that
    is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
    hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
    hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseScsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsv2_bufferSizeExt(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDcsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsv2_bufferSizeExt(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCcsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsv2_bufferSizeExt(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZcsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseScsrsv2_analysis(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""Sparse triangular solve using CSR storage format

    ``hipsparseXcsrsv2_analysis`` performs the analysis step for hipsparseScsrsv2_solve(),
    hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsv2_analysis(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsv2_analysis(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsv2_analysis(object handle, object transA, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseScsrsv2_solve(object handle, object transA, int m, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object f, object x, object policy, object pBuffer):
    r"""Sparse triangular solve using CSR storage format

    ``hipsparseXcsrsv2_solve`` solves a sparse triangular linear system of a sparse
    :math:`m \times m` matrix, defined in CSR storage format, a dense solution vector
    :math:`y` and the right-hand side :math:`x` that is multiplied by :math:`\alpha`, such that

    .. math::

       op(A) \cdot y = \alpha \cdot x,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        The sparse CSR matrix has to be sorted. This can be achieved by calling
        hipsparseXcsrsort().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` and
        ``trans`` == ``HIPSPARSE_OPERATION_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(f)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsv2_solve(object handle, object transA, int m, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(f)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsv2_solve(object handle, object transA, int m, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        float2.from_pyobj(f)._ptr,
        float2.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsv2_solve(object handle, object transA, int m, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrsv2Info.from_pyobj(info)._ptr,
        double2.from_pyobj(f)._ptr,
        double2.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseShybmv(object handle, object transA, object alpha, object descrA, object hybA, object x, object beta, object y):
    r"""Sparse matrix vector multiplication using HYB storage format

    ``hipsparseXhybmv`` multiplies the scalar :math:`\alpha` with a sparse :math:`m \times n`
    matrix, defined in HYB storage format, and the dense vector :math:`x` and adds the
    result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`,
    such that

    .. math::

       y := \alpha \cdot op(A) \cdot x + \beta \cdot y,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseShybmv__retval = hipsparseStatus_t(chipsparse.hipsparseShybmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseShybmv__retval,)


@cython.embedsignature(True)
def hipsparseDhybmv(object handle, object transA, object alpha, object descrA, object hybA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDhybmv__retval = hipsparseStatus_t(chipsparse.hipsparseDhybmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseDhybmv__retval,)


@cython.embedsignature(True)
def hipsparseChybmv(object handle, object transA, object alpha, object descrA, object hybA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseChybmv__retval = hipsparseStatus_t(chipsparse.hipsparseChybmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        float2.from_pyobj(x)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseChybmv__retval,)


@cython.embedsignature(True)
def hipsparseZhybmv(object handle, object transA, object alpha, object descrA, object hybA, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZhybmv__retval = hipsparseStatus_t(chipsparse.hipsparseZhybmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        double2.from_pyobj(x)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseZhybmv__retval,)


@cython.embedsignature(True)
def hipsparseSbsrmv(object handle, object dirA, object transA, int mb, int nb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object x, object beta, object y):
    r"""Sparse matrix vector multiplication using BSR storage format

    ``hipsparseXbsrmv`` multiplies the scalar :math:`\alpha` with a sparse

    math:`(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})`
    matrix, defined in BSR storage format, and the dense vector :math:`x` and adds the
    result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`,
    such that

    .. math::

       y := \alpha \cdot op(A) \cdot x + \beta \cdot y,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseSbsrmv__retval,)


@cython.embedsignature(True)
def hipsparseDbsrmv(object handle, object dirA, object transA, int mb, int nb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseDbsrmv__retval,)


@cython.embedsignature(True)
def hipsparseCbsrmv(object handle, object dirA, object transA, int mb, int nb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nb,nnzb,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        float2.from_pyobj(x)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseCbsrmv__retval,)


@cython.embedsignature(True)
def hipsparseZbsrmv(object handle, object dirA, object transA, int mb, int nb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrmv__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nb,nnzb,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        double2.from_pyobj(x)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseZbsrmv__retval,)


@cython.embedsignature(True)
def hipsparseSbsrxmv(object handle, object dir, object trans, int sizeOfMask, int mb, int nb, int nnzb, object alpha, object descr, object bsrVal, object bsrMaskPtr, object bsrRowPtr, object bsrEndPtr, object bsrColInd, int blockDim, object x, object beta, object y):
    r"""Sparse matrix vector multiplication with mask operation using BSR storage format

    ``hipsparseXbsrxmv`` multiplies the scalar :math:`\alpha` with a sparse

    math:`(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})`
    modified matrix, defined in BSR storage format, and the dense vector :math:`x` and adds the
    result to the dense vector :math:`y` that is multiplied by the scalar :math:`\beta`,
    such that

    .. math::

       y := \left( \alpha \cdot op(A) \cdot x + \beta \cdot y \right)\left( \text{mask} \right),

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.
        Currently, ``block_dim`` == 1 is not supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(trans,_hipsparseOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrxmv__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrxmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,trans.value,sizeOfMask,mb,nb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrMaskPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrEndPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColInd)._ptr,blockDim,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseSbsrxmv__retval,)


@cython.embedsignature(True)
def hipsparseDbsrxmv(object handle, object dir, object trans, int sizeOfMask, int mb, int nb, int nnzb, object alpha, object descr, object bsrVal, object bsrMaskPtr, object bsrRowPtr, object bsrEndPtr, object bsrColInd, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(trans,_hipsparseOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrxmv__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrxmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,trans.value,sizeOfMask,mb,nb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrMaskPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrEndPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColInd)._ptr,blockDim,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseDbsrxmv__retval,)


@cython.embedsignature(True)
def hipsparseCbsrxmv(object handle, object dir, object trans, int sizeOfMask, int mb, int nb, int nnzb, object alpha, object descr, object bsrVal, object bsrMaskPtr, object bsrRowPtr, object bsrEndPtr, object bsrColInd, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(trans,_hipsparseOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrxmv__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrxmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,trans.value,sizeOfMask,mb,nb,nnzb,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        float2.from_pyobj(bsrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrMaskPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrEndPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColInd)._ptr,blockDim,
        float2.from_pyobj(x)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseCbsrxmv__retval,)


@cython.embedsignature(True)
def hipsparseZbsrxmv(object handle, object dir, object trans, int sizeOfMask, int mb, int nb, int nnzb, object alpha, object descr, object bsrVal, object bsrMaskPtr, object bsrRowPtr, object bsrEndPtr, object bsrColInd, int blockDim, object x, object beta, object y):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(trans,_hipsparseOperation_t__Base):
        raise TypeError("argument 'trans' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrxmv__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrxmv(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,trans.value,sizeOfMask,mb,nb,nnzb,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        double2.from_pyobj(bsrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrMaskPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrEndPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColInd)._ptr,blockDim,
        double2.from_pyobj(x)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(y)._ptr))    # fully specified
    return (_hipsparseZbsrxmv__retval,)


@cython.embedsignature(True)
def hipsparseXbsrsv2_zeroPivot(object handle, object info, object position):
    r"""Sparse triangular solve using BSR storage format

    ``hipsparseXbsrsv2_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXbsrsv2_analysis() or
    hipsparseXbsrsv2_solve() computation. The first zero pivot :math:`j` at :math:`A_{j,j}`
    is stored in ``position,`` using same index base as the BSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXbsrsv2_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXbsrsv2_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXbsrsv2_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXbsrsv2_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsv2_bufferSize(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""Sparse triangular solve using BSR storage format

    ``hipsparseXbsrsv2_bufferSize`` returns the size of the temporary storage buffer that
    is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSbsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsv2_bufferSize(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDbsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsv2_bufferSize(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCbsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsv2_bufferSize(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrsv2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsv2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZbsrsv2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsv2_bufferSizeExt(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSize):
    r"""Sparse triangular solve using BSR storage format

    ``hipsparseXbsrsv2_bufferSizeExt`` returns the size of the temporary storage buffer that
    is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseSbsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsv2_bufferSizeExt(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDbsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsv2_bufferSizeExt(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCbsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsv2_bufferSizeExt(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZbsrsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsv2_analysis(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Sparse triangular solve using BSR storage format

    ``hipsparseXbsrsv2_analysis`` performs the analysis step for hipsparseXbsrsv2_solve().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsv2_analysis(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsv2_analysis(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsv2_analysis(object handle, object dirA, object transA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrsv2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsv2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrsv2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsv2_solve(object handle, object dirA, object transA, int mb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object f, object x, object policy, object pBuffer):
    r"""Sparse triangular solve using BSR storage format

    ``hipsparseXbsrsv2_solve`` solves a sparse triangular linear system of a sparse
    :math:`m \times m` matrix, defined in BSR storage format, a dense solution vector
    :math:`y` and the right-hand side :math:`x` that is multiplied by :math:`\alpha`, such that

    .. math::

       op(A) \cdot y = \alpha \cdot x,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        The sparse BSR matrix has to be sorted.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` and
        ``trans`` == ``HIPSPARSE_OPERATION_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(f)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsv2_solve(object handle, object dirA, object transA, int mb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(f)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsv2_solve(object handle, object dirA, object transA, int mb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        float2.from_pyobj(f)._ptr,
        float2.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsv2_solve(object handle, object dirA, object transA, int mb, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object f, object x, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrsv2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsv2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,mb,nnzb,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsv2Info.from_pyobj(info)._ptr,
        double2.from_pyobj(f)._ptr,
        double2.from_pyobj(x)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrsv2_solve__retval,)


@cython.embedsignature(True)
def hipsparseSgemvi_bufferSize(object handle, object transA, int m, int n, int nnz, object pBufferSize):
    r"""Dense matrix sparse vector multiplication

    ``hipsparseXgemvi_bufferSize`` returns the size of the temporary storage buffer
    required by hipsparseXgemvi(). The temporary storage buffer must be allocated by the
    user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSgemvi_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSgemvi_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseSgemvi_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDgemvi_bufferSize(object handle, object transA, int m, int n, int nnz, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDgemvi_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDgemvi_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDgemvi_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCgemvi_bufferSize(object handle, object transA, int m, int n, int nnz, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCgemvi_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCgemvi_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCgemvi_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZgemvi_bufferSize(object handle, object transA, int m, int n, int nnz, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZgemvi_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZgemvi_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZgemvi_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSgemvi(object handle, object transA, int m, int n, object alpha, object A, int lda, int nnz, object x, object xInd, object beta, object y, object idxBase, object pBuffer):
    r"""Dense matrix sparse vector multiplication

    ``hipsparseXgemvi`` multiplies the scalar :math:`\alpha` with a dense :math:`m \times n`
    matrix :math:`A` and the sparse vector :math:`x` and adds the result to the dense vector
    :math:`y` that is multiplied by the scalar :math:`\beta`, such that

    .. math::

       y := \alpha \cdot op(A) \cdot x + \beta \cdot y,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSgemvi__retval = hipsparseStatus_t(chipsparse.hipsparseSgemvi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgemvi__retval,)


@cython.embedsignature(True)
def hipsparseDgemvi(object handle, object transA, int m, int n, object alpha, object A, int lda, int nnz, object x, object xInd, object beta, object y, object idxBase, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDgemvi__retval = hipsparseStatus_t(chipsparse.hipsparseDgemvi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(y)._ptr,idxBase.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgemvi__retval,)


@cython.embedsignature(True)
def hipsparseCgemvi(object handle, object transA, int m, int n, object alpha, object A, int lda, int nnz, object x, object xInd, object beta, object y, object idxBase, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCgemvi__retval = hipsparseStatus_t(chipsparse.hipsparseCgemvi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,
        float2.from_pyobj(alpha)._ptr,
        float2.from_pyobj(A)._ptr,lda,nnz,
        float2.from_pyobj(x)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(y)._ptr,idxBase.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgemvi__retval,)


@cython.embedsignature(True)
def hipsparseZgemvi(object handle, object transA, int m, int n, object alpha, object A, int lda, int nnz, object x, object xInd, object beta, object y, object idxBase, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZgemvi__retval = hipsparseStatus_t(chipsparse.hipsparseZgemvi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,
        double2.from_pyobj(alpha)._ptr,
        double2.from_pyobj(A)._ptr,lda,nnz,
        double2.from_pyobj(x)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(xInd)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(y)._ptr,idxBase.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgemvi__retval,)


@cython.embedsignature(True)
def hipsparseSbsrmm(object handle, object dirA, object transA, object transB, int mb, int n, int kb, int nnzb, object alpha, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object B, int ldb, object beta, object C, int ldc):
    r"""Sparse matrix dense matrix multiplication using BSR storage format

    ``hipsparseXbsrmm`` multiplies the scalar :math:`\alpha` with a sparse :math:`mb \times kb`
    matrix :math:`A`, defined in BSR storage format, and the dense :math:`k \times n`
    matrix :math:`B` (where :math:`k = block\_dim \times kb`) and adds the result to the dense
    :math:`m \times n` matrix :math:`C` (where :math:`m = block\_dim \times mb`) that
    is multiplied by the scalar :math:`\beta`, such that

    .. math::

       C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
       \end{array}
       \right.

    and

    .. math::

       op(B) = \left\{
       \begin{array}{ll}
           B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans_A`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transB.value,mb,n,kb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseSbsrmm__retval,)


@cython.embedsignature(True)
def hipsparseDbsrmm(object handle, object dirA, object transA, object transB, int mb, int n, int kb, int nnzb, object alpha, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transB.value,mb,n,kb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseDbsrmm__retval,)


@cython.embedsignature(True)
def hipsparseCbsrmm(object handle, object dirA, object transA, object transB, int mb, int n, int kb, int nnzb, object alpha, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transB.value,mb,n,kb,nnzb,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        float2.from_pyobj(B)._ptr,ldb,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseCbsrmm__retval,)


@cython.embedsignature(True)
def hipsparseZbsrmm(object handle, object dirA, object transA, object transB, int mb, int n, int kb, int nnzb, object alpha, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transB.value,mb,n,kb,nnzb,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        double2.from_pyobj(B)._ptr,ldb,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseZbsrmm__retval,)


@cython.embedsignature(True)
def hipsparseScsrmm(object handle, object transA, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""Sparse matrix dense matrix multiplication using CSR storage format

    ``hipsparseXcsrmm`` multiplies the scalar :math:`\alpha` with a sparse :math:`m \times k`
    matrix :math:`A`, defined in CSR storage format, and the dense :math:`k \times n`
    matrix :math:`B` and adds the result to the dense :math:`m \times n` matrix :math:`C` that
    is multiplied by the scalar :math:`\beta`, such that

    .. math::

       C := \alpha \cdot op(A) \cdot B + \beta \cdot C,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    .. code-block::

       for(i = 0; i < ldc; ++i)
       {
           for(j = 0; j < n; ++j)
           {
               C[i][j] = beta * C[i][j];

               for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
               {
                   C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
               }
           }
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseScsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,k,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseScsrmm__retval,)


@cython.embedsignature(True)
def hipsparseDcsrmm(object handle, object transA, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,k,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseDcsrmm__retval,)


@cython.embedsignature(True)
def hipsparseCcsrmm(object handle, object transA, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,k,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseCcsrmm__retval,)


@cython.embedsignature(True)
def hipsparseZcsrmm(object handle, object transA, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrmm__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrmm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,m,n,k,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseZcsrmm__retval,)


@cython.embedsignature(True)
def hipsparseScsrmm2(object handle, object transA, object transB, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""Sparse matrix dense matrix multiplication using CSR storage format

    ``hipsparseXcsrmm2`` multiplies the scalar :math:`\alpha` with a sparse :math:`m \times k`
    matrix :math:`A`, defined in CSR storage format, and the dense :math:`k \times n`
    matrix :math:`B` and adds the result to the dense :math:`m \times n` matrix :math:`C` that
    is multiplied by the scalar :math:`\beta`, such that

    .. math::

       C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    and

    .. math::

       op(B) = \left\{
       \begin{array}{ll}
           B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
           B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    .. code-block::

       for(i = 0; i < ldc; ++i)
       {
           for(j = 0; j < n; ++j)
           {
               C[i][j] = beta * C[i][j];

               for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
               {
                   C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
               }
           }
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrmm2__retval = hipsparseStatus_t(chipsparse.hipsparseScsrmm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseScsrmm2__retval,)


@cython.embedsignature(True)
def hipsparseDcsrmm2(object handle, object transA, object transB, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrmm2__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrmm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseDcsrmm2__retval,)


@cython.embedsignature(True)
def hipsparseCcsrmm2(object handle, object transA, object transB, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrmm2__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrmm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseCcsrmm2__retval,)


@cython.embedsignature(True)
def hipsparseZcsrmm2(object handle, object transA, object transB, int m, int n, int k, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrmm2__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrmm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseZcsrmm2__retval,)


@cython.embedsignature(True)
def hipsparseXbsrsm2_zeroPivot(object handle, object info, object position):
    r"""Sparse triangular system solve using BSR storage format

    ``hipsparseXbsrsm2_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXbsrsm2_analysis() or
    hipsparseXbsrsm2_solve() computation. The first zero pivot :math:`j` at :math:`A_{j,j}`
    is stored in ``position,`` using same index base as the BSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXbsrsm2_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXbsrsm2_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXbsrsm2_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXbsrsm2_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsm2_bufferSize(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""Sparse triangular system solve using BSR storage format

    ``hipsparseXbsrsm2_buffer_size`` returns the size of the temporary storage buffer that
    is required by hipsparseXbsrsm2_analysis() and hipsparseXbsrsm2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseSbsrsm2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsm2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSbsrsm2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsm2_bufferSize(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDbsrsm2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsm2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDbsrsm2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsm2_bufferSize(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCbsrsm2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsm2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCbsrsm2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsm2_bufferSize(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZbsrsm2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsm2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZbsrsm2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsm2_analysis(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Sparse triangular system solve using BSR storage format

    ``hipsparseXbsrsm2_analysis`` performs the analysis step for hipsparseXbsrsm2_solve().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsm2_analysis(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsm2_analysis(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsm2_analysis(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSbsrsm2_solve(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object B, int ldb, object X, int ldx, object policy, object pBuffer):
    r"""Sparse triangular system solve using BSR storage format

    ``hipsparseXbsrsm2_solve`` solves a sparse triangular linear system of a sparse
    :math:`m \times m` matrix, defined in BSR storage format, a dense solution matrix
    :math:`X` and the right-hand side matrix :math:`B` that is multiplied by :math:`\alpha`, such that

    .. math::

       op(A) \cdot op(X) = \alpha \cdot op(B),

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    ,

    .. math::

       op(X) = \left\{
       \begin{array}{ll}
           X,   & \text{if trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           X^T, & \text{if trans_X == HIPSPARSE_OPERATION_TRANSPOSE} \\
           X^H, & \text{if trans_X == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        The sparse BSR matrix has to be sorted.

    Note:
        Operation type of B and X must match, if :math:`op(B)=B, op(X)=X`.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans_A`` != ``HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`` and
        ``trans_X`` != ``HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <float *>hip._util.types.Pointer.from_pyobj(X)._ptr,ldx,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseDbsrsm2_solve(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object B, int ldb, object X, int ldx, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <double *>hip._util.types.Pointer.from_pyobj(X)._ptr,ldx,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseCbsrsm2_solve(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object B, int ldb, object X, int ldx, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        float2.from_pyobj(X)._ptr,ldx,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseZbsrsm2_solve(object handle, object dirA, object transA, object transX, int mb, int nrhs, int nnzb, object alpha, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object B, int ldb, object X, int ldx, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,transA.value,transX.value,mb,nrhs,nnzb,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrsm2Info.from_pyobj(info)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        double2.from_pyobj(X)._ptr,ldx,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseXcsrsm2_zeroPivot(object handle, object info, object position):
    r"""Sparse triangular system solve using CSR storage format

    ``hipsparseXcsrsm2_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXcsrsm2_analysis() or
    hipsparseXcsrsm2_solve() computation. The first zero pivot :math:`j` at :math:`A_{j,j}`
    is stored in ``position,`` using same index base as the CSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXcsrsm2_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrsm2_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrsm2_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrsm2Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXcsrsm2_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseScsrsm2_bufferSizeExt(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBufferSize):
    r"""Sparse triangular system solve using CSR storage format

    ``hipsparseXcsrsm2_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsrsm2_analysis() and hipsparseXcsrsm2_solve(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrsm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseScsrsm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsm2_bufferSizeExt(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrsm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDcsrsm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsm2_bufferSizeExt(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrsm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCcsrsm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsm2_bufferSizeExt(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrsm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZcsrsm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseScsrsm2_analysis(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""Sparse triangular system solve using CSR storage format

    ``hipsparseXcsrsm2_analysis`` performs the analysis step for hipsparseXcsrsm2_solve().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsm2_analysis(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsm2_analysis(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsm2_analysis(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrsm2_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsm2_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrsm2_analysis__retval,)


@cython.embedsignature(True)
def hipsparseScsrsm2_solve(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""Sparse triangular system solve using CSR storage format

    ``hipsparseXcsrsm2_solve`` solves a sparse triangular linear system of a sparse
    :math:`m \times m` matrix, defined in CSR storage format, a dense solution matrix
    :math:`X` and the right-hand side matrix :math:`B` that is multiplied by :math:`\alpha`, such that

    .. math::

       op(A) \cdot op(X) = \alpha \cdot op(B),

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    ,

    .. math::

       op(B) = \left\{
       \begin{array}{ll}
           B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
           B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    and

    .. math::

       op(X) = \left\{
       \begin{array}{ll}
           X,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           X^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
           X^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        The sparse CSR matrix has to be sorted. This can be achieved by calling
        hipsparseXcsrsort().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``trans_A`` != ``HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`` and
        ``trans_B`` != ``HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseScsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseDcsrsm2_solve(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseCcsrsm2_solve(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseZcsrsm2_solve(object handle, int algo, object transA, object transB, int m, int nrhs, int nnz, object alpha, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object B, int ldb, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrsm2_solve__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrsm2_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,transA.value,transB.value,m,nrhs,nnz,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        csrsm2Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrsm2_solve__retval,)


@cython.embedsignature(True)
def hipsparseSgemmi(object handle, int m, int n, int k, int nnz, object alpha, object A, int lda, object cscValB, object cscColPtrB, object cscRowIndB, object beta, object C, int ldc):
    r"""Dense matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXgemmi`` multiplies the scalar :math:`\alpha` with a dense :math:`m \times k`
    matrix :math:`A` and the sparse :math:`k \times n` matrix :math:`B`, defined in CSR
    storage format and adds the result to the dense :math:`m \times n` matrix :math:`C` that
    is multiplied by the scalar :math:`\beta`, such that

    .. math::

       C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    and

    .. math::

       op(B) = \left\{
       \begin{array}{ll}
           B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
           B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgemmi__retval = hipsparseStatus_t(chipsparse.hipsparseSgemmi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const float *>hip._util.types.Pointer.from_pyobj(cscValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscRowIndB)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseSgemmi__retval,)


@cython.embedsignature(True)
def hipsparseDgemmi(object handle, int m, int n, int k, int nnz, object alpha, object A, int lda, object cscValB, object cscColPtrB, object cscRowIndB, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgemmi__retval = hipsparseStatus_t(chipsparse.hipsparseDgemmi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const double *>hip._util.types.Pointer.from_pyobj(cscValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscRowIndB)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseDgemmi__retval,)


@cython.embedsignature(True)
def hipsparseCgemmi(object handle, int m, int n, int k, int nnz, object alpha, object A, int lda, object cscValB, object cscColPtrB, object cscRowIndB, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgemmi__retval = hipsparseStatus_t(chipsparse.hipsparseCgemmi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,nnz,
        float2.from_pyobj(alpha)._ptr,
        float2.from_pyobj(A)._ptr,lda,
        float2.from_pyobj(cscValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscRowIndB)._ptr,
        float2.from_pyobj(beta)._ptr,
        float2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseCgemmi__retval,)


@cython.embedsignature(True)
def hipsparseZgemmi(object handle, int m, int n, int k, int nnz, object alpha, object A, int lda, object cscValB, object cscColPtrB, object cscRowIndB, object beta, object C, int ldc):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgemmi__retval = hipsparseStatus_t(chipsparse.hipsparseZgemmi(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,nnz,
        double2.from_pyobj(alpha)._ptr,
        double2.from_pyobj(A)._ptr,lda,
        double2.from_pyobj(cscValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscRowIndB)._ptr,
        double2.from_pyobj(beta)._ptr,
        double2.from_pyobj(C)._ptr,ldc))    # fully specified
    return (_hipsparseZgemmi__retval,)


@cython.embedsignature(True)
def hipsparseXcsrgeamNnz(object handle, int m, int n, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr):
    r"""Sparse matrix sparse matrix addition using CSR storage format

    ``hipsparseXcsrgeamNnz`` computes the total CSR non-zero elements and the CSR row
    offsets, that point to the start of every row of the sparse CSR matrix, of the
    resulting matrix C. It is assumed that ``csr_row_ptr_C`` has been allocated with
    size ``m`` + 1.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrgeamNnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrgeamNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseXcsrgeamNnz__retval,)


@cython.embedsignature(True)
def hipsparseScsrgeam(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object beta, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""Sparse matrix sparse matrix addition using CSR storage format

    ``hipsparseXcsrgeam`` multiplies the scalar :math:`\alpha` with the sparse
    :math:`m \times n` matrix :math:`A`, defined in CSR storage format, multiplies the
    scalar :math:`\beta` with the sparse :math:`m \times n` matrix :math:`B`, defined in CSR
    storage format, and adds both resulting matrices to obtain the sparse
    :math:`m \times n` matrix :math:`C`, defined in CSR storage format, such that

    .. math::

       C := \alpha \cdot A + \beta \cdot B.

    Note:
        Both scalars :math:`\alpha` and :math:`beta` have to be valid.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Note:
        This function is non blocking and executed asynchronously with respect to the
        host. It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrgeam__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgeam(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseScsrgeam__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgeam(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object beta, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrgeam__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgeam(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseDcsrgeam__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgeam(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object beta, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrgeam__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgeam(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        float2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        float2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseCcsrgeam__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgeam(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object beta, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrgeam__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgeam(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        double2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        double2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseZcsrgeam__retval,)


@cython.embedsignature(True)
def hipsparseScsrgeam2_bufferSizeExt(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBufferSizeInBytes):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgeam2_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsrgeam2Nnz() and hipsparseXcsrgeam2(). The temporary
    storage buffer must be allocated by the user.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrgeam2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgeam2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsrgeam2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgeam2_bufferSizeExt(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrgeam2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgeam2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsrgeam2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgeam2_bufferSizeExt(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrgeam2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgeam2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        float2.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrSortedValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsrgeam2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgeam2_bufferSizeExt(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrgeam2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgeam2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        double2.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrSortedValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsrgeam2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseXcsrgeam2Nnz(object handle, int m, int n, object descrA, int nnzA, object csrSortedRowPtrA, object csrSortedColIndA, object descrB, int nnzB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedRowPtrC, object nnzTotalDevHostPtr, object workspace):
    r"""Sparse matrix sparse matrix addition using CSR storage format

    ``hipsparseXcsrgeam2Nnz`` computes the total CSR non-zero elements and the CSR row
    offsets, that point to the start of every row of the sparse CSR matrix, of the
    resulting matrix C. It is assumed that ``csr_row_ptr_C`` has been allocated with
    size ``m`` + 1.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrgeam2Nnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrgeam2Nnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(workspace)._ptr))    # fully specified
    return (_hipsparseXcsrgeam2Nnz__retval,)


@cython.embedsignature(True)
def hipsparseScsrgeam2(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBuffer):
    r"""Sparse matrix sparse matrix addition using CSR storage format

    ``hipsparseXcsrgeam2`` multiplies the scalar :math:`\alpha` with the sparse
    :math:`m \times n` matrix :math:`A`, defined in CSR storage format, multiplies the
    scalar :math:`\beta` with the sparse :math:`m \times n` matrix :math:`B`, defined in CSR
    storage format, and adds both resulting matrices to obtain the sparse
    :math:`m \times n` matrix :math:`C`, defined in CSR storage format, such that

    .. math::

       C := \alpha \cdot A + \beta \cdot B.

    Note:
        Both scalars :math:`\alpha` and :math:`beta` have to be valid.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Note:
        This function is non blocking and executed asynchronously with respect to the
        host. It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrgeam2__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgeam2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrgeam2__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgeam2(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrgeam2__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgeam2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrgeam2__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgeam2(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrgeam2__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgeam2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        float2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        float2.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrSortedValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrgeam2__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgeam2(object handle, int m, int n, object alpha, object descrA, int nnzA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object beta, object descrB, int nnzB, object csrSortedValB, object csrSortedRowPtrB, object csrSortedColIndB, object descrC, object csrSortedValC, object csrSortedRowPtrC, object csrSortedColIndC, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrgeam2__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgeam2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        double2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        double2.from_pyobj(csrSortedValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrSortedValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrgeam2__retval,)


@cython.embedsignature(True)
def hipsparseXcsrgemmNnz(object handle, object transA, object transB, int m, int n, int k, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgemmNnz`` computes the total CSR non-zero elements and the CSR row
    offsets, that point to the start of every row of the sparse CSR matrix, of the
    resulting multiplied matrix C. It is assumed that ``csr_row_ptr_C`` has been allocated
    with size ``m`` + 1.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Please note, that for matrix products with more than 8192 intermediate products per
        row, additional temporary storage buffer is allocated by the algorithm.

    Note:
        Currently, only ``trans_A`` == ``trans_B`` == ``HIPSPARSE_OPERATION_NONE`` is
        supported.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseXcsrgemmNnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrgemmNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseXcsrgemmNnz__retval,)


@cython.embedsignature(True)
def hipsparseScsrgemm(object handle, object transA, object transB, int m, int n, int k, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgemm`` multiplies the sparse :math:`m \times k` matrix :math:`A`, defined in
    CSR storage format with the sparse :math:`k \times n` matrix :math:`B`, defined in CSR
    storage format, and stores the result in the sparse :math:`m \times n` matrix :math:`C`,
    defined in CSR storage format, such that

    .. math::

       C := op(A) \cdot op(B),

    with

    .. math::

       op(A) = \left\{
       \begin{array}{ll}
           A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
           A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    and

    .. math::

       op(B) = \left\{
       \begin{array}{ll}
           B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
           B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
           B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
       \end{array}
       \right.

    Note:
        Currently, only ``trans_A`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Note:
        Currently, only ``trans_B`` == ``HIPSPARSE_OPERATION_NON_TRANSPOSE`` is supported.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Note:
        This function is non blocking and executed asynchronously with respect to the
        host. It may return before the actual computation has finished.

    Note:
        Please note, that for matrix products with more than 4096 non-zero entries per
        row, additional temporary storage buffer is allocated by the algorithm.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseScsrgemm__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgemm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseScsrgemm__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgemm(object handle, object transA, object transB, int m, int n, int k, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseDcsrgemm__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgemm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseDcsrgemm__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgemm(object handle, object transA, object transB, int m, int n, int k, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseCcsrgemm__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgemm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        float2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseCcsrgemm__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgemm(object handle, object transA, object transB, int m, int n, int k, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(transA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(transB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'transB' must be of type '_hipsparseOperation_t__Base'")
    _hipsparseZcsrgemm__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgemm(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,transA.value,transB.value,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        double2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseZcsrgemm__retval,)


@cython.embedsignature(True)
def hipsparseScsrgemm2_bufferSizeExt(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrRowPtrD, object csrColIndD, object info, object pBufferSizeInBytes):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgemm2_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsrgemm2Nnz() and hipsparseXcsrgemm2(). The temporary
    storage buffer must be allocated by the user.

    Note:
        Please note, that for matrix products with more than 4096 non-zero entries per row,
        additional temporary storage buffer is allocated by the algorithm.

    Note:
        Please note, that for matrix products with more than 8192 intermediate products per
        row, additional temporary storage buffer is allocated by the algorithm.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrgemm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgemm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsrgemm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgemm2_bufferSizeExt(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrRowPtrD, object csrColIndD, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrgemm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgemm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsrgemm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgemm2_bufferSizeExt(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrRowPtrD, object csrColIndD, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrgemm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgemm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        float2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsrgemm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgemm2_bufferSizeExt(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrRowPtrD, object csrColIndD, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrgemm2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgemm2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        double2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsrgemm2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseXcsrgemm2Nnz(object handle, int m, int n, int k, object descrA, int nnzA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrRowPtrB, object csrColIndB, object descrD, int nnzD, object csrRowPtrD, object csrColIndD, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr, object info, object pBuffer):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgemm2Nnz`` computes the total CSR non-zero elements and the CSR row
    offsets, that point to the start of every row of the sparse CSR matrix, of the
    resulting multiplied matrix C. It is assumed that ``csr_row_ptr_C`` has been allocated
    with size ``m`` + 1.
    The required buffer size can be obtained by hipsparseXcsrgemm2_bufferSizeExt().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Note:
        Please note, that for matrix products with more than 8192 intermediate products per
        row, additional temporary storage buffer is allocated by the algorithm.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrgemm2Nnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrgemm2Nnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseXcsrgemm2Nnz__retval,)


@cython.embedsignature(True)
def hipsparseScsrgemm2(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrValD, object csrRowPtrD, object csrColIndD, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object pBuffer):
    r"""Sparse matrix sparse matrix multiplication using CSR storage format

    ``hipsparseXcsrgemm2`` multiplies the scalar :math:`\alpha` with the sparse
    :math:`m \times k` matrix :math:`A`, defined in CSR storage format, and the sparse
    :math:`k \times n` matrix :math:`B`, defined in CSR storage format, and adds the result
    to the sparse :math:`m \times n` matrix :math:`D` that is multiplied by :math:`\beta`. The
    final result is stored in the sparse :math:`m \times n` matrix :math:`C`, defined in CSR
    storage format, such
    that

    .. math::

       C := \alpha \cdot A \cdot B + \beta \cdot D

    Note:
        If :math:`\alpha == 0`, then :math:`C = \beta \cdot D` will be computed.

    Note:
        If :math:`\beta == 0`, then :math:`C = \alpha \cdot A \cdot B` will be computed.

    Note:
        math:`\alpha == beta == 0` is invalid.

    Note:
        Currently, only ``HIPSPARSE_MATRIX_TYPE_GENERAL`` is supported.

    Note:
        This function is non blocking and executed asynchronously with respect to the
        host. It may return before the actual computation has finished.

    Note:
        Please note, that for matrix products with more than 4096 non-zero entries per
        row, additional temporary storage buffer is allocated by the algorithm.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrgemm2__retval = hipsparseStatus_t(chipsparse.hipsparseScsrgemm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        <const float *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrgemm2__retval,)


@cython.embedsignature(True)
def hipsparseDcsrgemm2(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrValD, object csrRowPtrD, object csrColIndD, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrgemm2__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrgemm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        <const double *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrgemm2__retval,)


@cython.embedsignature(True)
def hipsparseCcsrgemm2(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrValD, object csrRowPtrD, object csrColIndD, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrgemm2__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrgemm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        float2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        float2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        float2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        float2.from_pyobj(csrValD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrgemm2__retval,)


@cython.embedsignature(True)
def hipsparseZcsrgemm2(object handle, int m, int n, int k, object alpha, object descrA, int nnzA, object csrValA, object csrRowPtrA, object csrColIndA, object descrB, int nnzB, object csrValB, object csrRowPtrB, object csrColIndB, object beta, object descrD, int nnzD, object csrValD, object csrRowPtrD, object csrColIndD, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrgemm2__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrgemm2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,k,
        double2.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,nnzA,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrB)._ptr,nnzB,
        double2.from_pyobj(csrValB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrB)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndB)._ptr,
        double2.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrD)._ptr,nnzD,
        double2.from_pyobj(csrValD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrD)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndD)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        csrgemm2Info.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrgemm2__retval,)


@cython.embedsignature(True)
def hipsparseXbsrilu02_zeroPivot(object handle, object info, object position):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
    format

    ``hipsparseXbsrilu02_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXbsrilu02_analysis() or
    hipsparseXbsrilu02() computation. The first zero pivot :math:`j` at :math:`A_{j,j}` is
    stored in ``position,`` using same index base as the BSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        If a zero pivot is found, ``position`` :math:`=j` means that either the diagonal block
        :math:`A_{j,j}` is missing (structural zero) or the diagonal block :math:`A_{j,j}` is not
        invertible (numerical zero).

    Note:
        ``hipsparseXbsrilu02_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXbsrilu02_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXbsrilu02_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXbsrilu02_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseSbsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
    format

    ``hipsparseXbsrilu02_numericBoost`` enables the user to replace a numerical value in
    an incomplete LU factorization. ``tol`` is used to determine whether a numerical value
    is replaced by ``boost_val,`` such that :math:`A_{j,j} = \text{boost_val}` if
    :math:`\text{tol} \ge \left|A_{j,j}\right|`.

    Note:
        The boost value is enabled by setting ``enable_boost`` to 1 or disabled by
        setting ``enable_boost`` to 0.

    Note:
        ``tol`` and ``boost_val`` can be in host or device memory.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSbsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseSbsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseDbsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDbsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseDbsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseCbsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCbsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        float2.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseCbsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseZbsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZbsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        double2.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseZbsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseSbsrilu02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
    format

    ``hipsparseXbsrilu02_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXbsrilu02_analysis() and hipsparseXbsrilu02_solve().
    The temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSbsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSbsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDbsrilu02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDbsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDbsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCbsrilu02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCbsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCbsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZbsrilu02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZbsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZbsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSbsrilu02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
    format

    ``hipsparseXbsrilu02_analysis`` performs the analysis step for hipsparseXbsrilu02().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDbsrilu02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCbsrilu02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZbsrilu02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSbsrilu02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA_valM, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
    format

    ``hipsparseXbsrilu02`` computes the incomplete LU factorization with 0 fill-ins and no
    pivoting of a sparse :math:`mb \times mb` BSR matrix :math:`A`, such that

    .. math::

       A \approx LU

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseSbsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseDbsrilu02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA_valM, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseDbsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseCbsrilu02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA_valM, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseCbsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseZbsrilu02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrSortedValA_valM, object bsrSortedRowPtrA, object bsrSortedColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseZbsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrSortedColIndA)._ptr,blockDim,
        bsrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseXcsrilu02_zeroPivot(object handle, object info, object position):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsrilu02_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXcsrilu02() computation.
    The first zero pivot :math:`j` at :math:`A_{j,j}` is stored in ``position,`` using same
    index base as the CSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXcsrilu02_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrilu02_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrilu02_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXcsrilu02_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseScsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
    format

    ``hipsparseXcsrilu02_numericBoost`` enables the user to replace a numerical value in
    an incomplete LU factorization. ``tol`` is used to determine whether a numerical value
    is replaced by ``boost_val,`` such that :math:`A_{j,j} = \text{boost_val}` if
    :math:`\text{tol} \ge \left|A_{j,j}\right|`.

    Note:
        The boost value is enabled by setting ``enable_boost`` to 1 or disabled by
        setting ``enable_boost`` to 0.

    Note:
        ``tol`` and ``boost_val`` can be in host or device memory.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseScsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseScsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseDcsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseDcsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseCcsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        float2.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseCcsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseZcsrilu02_numericBoost(object handle, object info, int enable_boost, object tol, object boost_val):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrilu02_numericBoost__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrilu02_numericBoost(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,enable_boost,
        <double *>hip._util.types.Pointer.from_pyobj(tol)._ptr,
        double2.from_pyobj(boost_val)._ptr))    # fully specified
    return (_hipsparseZcsrilu02_numericBoost__retval,)


@cython.embedsignature(True)
def hipsparseScsrilu02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsrilu02_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseScsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDcsrilu02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCcsrilu02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZcsrilu02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrilu02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrilu02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsrilu02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseScsrilu02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsrilu02_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrilu02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsrilu02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseScsrilu02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsrilu02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrilu02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrilu02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDcsrilu02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsrilu02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrilu02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrilu02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCcsrilu02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsrilu02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrilu02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrilu02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZcsrilu02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseScsrilu02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsrilu02_analysis`` performs the analysis step for hipsparseXcsrilu02().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseScsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDcsrilu02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCcsrilu02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZcsrilu02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrilu02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrilu02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrilu02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseScsrilu02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsrilu02`` computes the incomplete LU factorization with 0 fill-ins and no
    pivoting of a sparse :math:`m \times m` CSR matrix :math:`A`, such that

    .. math::

       A \approx LU

    Note:
        The sparse CSR matrix has to be sorted. This can be achieved by calling
        hipsparseXcsrsort().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseScsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseDcsrilu02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseCcsrilu02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseZcsrilu02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsrilu02__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrilu02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csrilu02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsrilu02__retval,)


@cython.embedsignature(True)
def hipsparseXbsric02_zeroPivot(object handle, object info, object position):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
    storage format

    ``hipsparseXbsric02_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXbsric02_analysis() or
    hipsparseXbsric02() computation. The first zero pivot :math:`j` at :math:`A_{j,j}` is
    stored in ``position,`` using same index base as the BSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        If a zero pivot is found, ``position=j`` means that either the diagonal block ``A(j,j)``
        is missing (structural zero) or the diagonal block ``A(j,j)`` is not positive definite
        (numerical zero).

    Note:
        ``hipsparseXbsric02_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXbsric02_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXbsric02_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        bsric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXbsric02_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseSbsric02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
    storage format

    ``hipsparseXbsric02_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXbsric02_analysis() and hipsparseXbsric02(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSbsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSbsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSbsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDbsric02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDbsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDbsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDbsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCbsric02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCbsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCbsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCbsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZbsric02_bufferSize(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZbsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZbsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZbsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSbsric02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
    storage format

    ``hipsparseXbsric02_analysis`` performs the analysis step for hipsparseXbsric02().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSbsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDbsric02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDbsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCbsric02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCbsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZbsric02_analysis(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZbsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSbsric02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
    storage format

    ``hipsparseXbsric02`` computes the incomplete Cholesky factorization with 0 fill-ins
    and no pivoting of a sparse :math:`mb \times mb` BSR matrix :math:`A`, such that

    .. math::

       A \approx LL^T

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseSbsric02__retval = hipsparseStatus_t(chipsparse.hipsparseSbsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSbsric02__retval,)


@cython.embedsignature(True)
def hipsparseDbsric02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDbsric02__retval = hipsparseStatus_t(chipsparse.hipsparseDbsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDbsric02__retval,)


@cython.embedsignature(True)
def hipsparseCbsric02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCbsric02__retval = hipsparseStatus_t(chipsparse.hipsparseCbsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCbsric02__retval,)


@cython.embedsignature(True)
def hipsparseZbsric02(object handle, object dirA, int mb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")                    
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZbsric02__retval = hipsparseStatus_t(chipsparse.hipsparseZbsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        bsric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZbsric02__retval,)


@cython.embedsignature(True)
def hipsparseXcsric02_zeroPivot(object handle, object info, object position):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsric02_zeroPivot`` returns ``HIPSPARSE_STATUS_ZERO_PIVOT`` , if either a
    structural or numerical zero has been found during hipsparseXcsric02_analysis() or
    hipsparseXcsric02() computation. The first zero pivot :math:`j` at :math:`A_{j,j}`
    is stored in ``position,`` using same index base as the CSR matrix.

    ``position`` can be in host or device memory. If no zero pivot has been found,
    ``position`` is set to -1 and ``HIPSPARSE_STATUS_SUCCESS`` is returned instead.

    Note:
        ``hipsparseXcsric02_zeroPivot`` is a blocking function. It might influence
        performance negatively.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsric02_zeroPivot__retval = hipsparseStatus_t(chipsparse.hipsparseXcsric02_zeroPivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(position)._ptr))    # fully specified
    return (_hipsparseXcsric02_zeroPivot__retval,)


@cython.embedsignature(True)
def hipsparseScsric02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsric02_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseScsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDcsric02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDcsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCcsric02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCcsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZcsric02_bufferSize(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsric02_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZcsric02_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsric02_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseScsric02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsric02_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsric02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsric02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseScsric02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsric02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsric02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsric02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseDcsric02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsric02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsric02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsric02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseCcsric02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsric02_bufferSizeExt(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object pBufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsric02_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsric02_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSize)._ptr))    # fully specified
    return (_hipsparseZcsric02_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseScsric02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsric02_analysis`` performs the analysis step for hipsparseXcsric02().

    Note:
        If the matrix sparsity pattern changes, the gathered information will become invalid.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseScsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDcsric02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDcsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseCcsric02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseCcsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseZcsric02_analysis(object handle, int m, int nnz, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsric02_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseZcsric02_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsric02_analysis__retval,)


@cython.embedsignature(True)
def hipsparseScsric02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
    storage format

    ``hipsparseXcsric02`` computes the incomplete Cholesky factorization with 0 fill-ins
    and no pivoting of a sparse :math:`m \times m` CSR matrix :math:`A`, such that

    .. math::

       A \approx LL^T

    Note:
        The sparse CSR matrix has to be sorted. This can be achieved by calling
        hipsparseXcsrsort().

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseScsric02__retval = hipsparseStatus_t(chipsparse.hipsparseScsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsric02__retval,)


@cython.embedsignature(True)
def hipsparseDcsric02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseDcsric02__retval = hipsparseStatus_t(chipsparse.hipsparseDcsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsric02__retval,)


@cython.embedsignature(True)
def hipsparseCcsric02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseCcsric02__retval = hipsparseStatus_t(chipsparse.hipsparseCcsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsric02__retval,)


@cython.embedsignature(True)
def hipsparseZcsric02(object handle, int m, int nnz, object descrA, object csrSortedValA_valM, object csrSortedRowPtrA, object csrSortedColIndA, object info, object policy, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(policy,_hipsparseSolvePolicy_t__Base):
        raise TypeError("argument 'policy' must be of type '_hipsparseSolvePolicy_t__Base'")
    _hipsparseZcsric02__retval = hipsparseStatus_t(chipsparse.hipsparseZcsric02(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA_valM)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        csric02Info.from_pyobj(info)._ptr,policy.value,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsric02__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""Tridiagonal solver with pivoting

    ``hipsparseXgtsv2_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXgtsv2(). The temporary storage buffer must be
    allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSgtsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int db, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,db,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDgtsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCgtsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZgtsv2_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""Tridiagonal solver with pivoting

    ``hipsparseXgtsv2`` solves a tridiagonal system for multiple right hand sides using pivoting.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgtsv2__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgtsv2__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgtsv2__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgtsv2__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2_nopivot_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""Tridiagonal solver (no pivoting)

    ``hipsparseXgtsv2_nopivot_bufferSizeExt`` returns the size of the temporary storage
    buffer that is required by hipsparseXgtsv2_nopivot(). The temporary storage buffer
    must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2_nopivot_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2_nopivot_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSgtsv2_nopivot_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2_nopivot_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int db, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2_nopivot_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2_nopivot_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(B)._ptr,db,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDgtsv2_nopivot_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2_nopivot_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2_nopivot_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2_nopivot_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCgtsv2_nopivot_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2_nopivot_bufferSizeExt(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2_nopivot_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2_nopivot_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZgtsv2_nopivot_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2_nopivot(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""Tridiagonal solver (no pivoting)

    ``hipsparseXgtsv2_nopivot`` solves a tridiagonal linear system for multiple right-hand sides

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2_nopivot__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2_nopivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgtsv2_nopivot__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2_nopivot(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2_nopivot__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2_nopivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgtsv2_nopivot__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2_nopivot(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2_nopivot__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2_nopivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgtsv2_nopivot__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2_nopivot(object handle, int m, int n, object dl, object d, object du, object B, int ldb, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2_nopivot__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2_nopivot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(B)._ptr,ldb,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgtsv2_nopivot__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2StridedBatch_bufferSizeExt(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBufferSizeInBytes):
    r"""Strided Batch tridiagonal solver (no pivoting)

    ``hipsparseXgtsv2StridedBatch_bufferSizeExt`` returns the size of the temporary storage
    buffer that is required by hipsparseXgtsv2StridedBatch(). The temporary storage buffer
    must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2StridedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2StridedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,batchStride,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSgtsv2StridedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2StridedBatch_bufferSizeExt(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2StridedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2StridedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,batchStride,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDgtsv2StridedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2StridedBatch_bufferSizeExt(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2StridedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2StridedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,batchStride,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCgtsv2StridedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2StridedBatch_bufferSizeExt(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2StridedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2StridedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,batchStride,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZgtsv2StridedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSgtsv2StridedBatch(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBuffer):
    r"""Strided Batch tridiagonal solver (no pivoting)

    ``hipsparseXgtsv2StridedBatch`` solves a batched tridiagonal linear system

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsv2StridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsv2StridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,batchStride,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgtsv2StridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseDgtsv2StridedBatch(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsv2StridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsv2StridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,batchStride,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgtsv2StridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseCgtsv2StridedBatch(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsv2StridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsv2StridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,batchStride,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgtsv2StridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseZgtsv2StridedBatch(object handle, int m, object dl, object d, object du, object x, int batchCount, int batchStride, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsv2StridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsv2StridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,batchStride,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgtsv2StridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseSgtsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBufferSizeInBytes):
    r"""Interleaved Batch tridiagonal solver

    ``hipsparseXgtsvInterleavedBatch_bufferSizeExt`` returns the size of the temporary storage
    buffer that is required by hipsparseXgtsvInterleavedBatch(). The temporary storage buffer
    must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSgtsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDgtsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDgtsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCgtsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCgtsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZgtsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZgtsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSgtsvInterleavedBatch(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBuffer):
    r"""Interleaved Batch tridiagonal solver

    ``hipsparseXgtsvInterleavedBatch`` solves a batched tridiagonal linear system

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgtsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseSgtsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgtsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseDgtsvInterleavedBatch(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgtsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseDgtsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgtsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseCgtsvInterleavedBatch(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgtsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseCgtsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgtsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseZgtsvInterleavedBatch(object handle, int algo, int m, object dl, object d, object du, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgtsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseZgtsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgtsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseSgpsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBufferSizeInBytes):
    r"""Interleaved Batch pentadiagonal solver

    ``hipsparseXgpsvInterleavedBatch_bufferSizeExt`` returns the size of the temporary storage
    buffer that is required by hipsparseXgpsvInterleavedBatch(). The temporary storage buffer
    must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgpsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSgpsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <const float *>hip._util.types.Pointer.from_pyobj(ds)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(dw)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseSgpsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDgpsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgpsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDgpsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <const double *>hip._util.types.Pointer.from_pyobj(ds)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(dw)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDgpsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCgpsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgpsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCgpsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        float2.from_pyobj(ds)._ptr,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(dw)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCgpsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZgpsvInterleavedBatch_bufferSizeExt(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgpsvInterleavedBatch_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZgpsvInterleavedBatch_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        double2.from_pyobj(ds)._ptr,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(dw)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZgpsvInterleavedBatch_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSgpsvInterleavedBatch(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBuffer):
    r"""Interleaved Batch pentadiagonal solver

    ``hipsparseXgpsvInterleavedBatch`` solves a batched pentadiagonal linear system

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgpsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseSgpsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <float *>hip._util.types.Pointer.from_pyobj(ds)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(dw)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseSgpsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseDgpsvInterleavedBatch(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgpsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseDgpsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        <double *>hip._util.types.Pointer.from_pyobj(ds)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(dl)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(d)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(du)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(dw)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDgpsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseCgpsvInterleavedBatch(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgpsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseCgpsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        float2.from_pyobj(ds)._ptr,
        float2.from_pyobj(dl)._ptr,
        float2.from_pyobj(d)._ptr,
        float2.from_pyobj(du)._ptr,
        float2.from_pyobj(dw)._ptr,
        float2.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCgpsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseZgpsvInterleavedBatch(object handle, int algo, int m, object ds, object dl, object d, object du, object dw, object x, int batchCount, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgpsvInterleavedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseZgpsvInterleavedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,algo,m,
        double2.from_pyobj(ds)._ptr,
        double2.from_pyobj(dl)._ptr,
        double2.from_pyobj(d)._ptr,
        double2.from_pyobj(du)._ptr,
        double2.from_pyobj(dw)._ptr,
        double2.from_pyobj(x)._ptr,batchCount,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZgpsvInterleavedBatch__retval,)


@cython.embedsignature(True)
def hipsparseSnnz(object handle, object dirA, int m, int n, object descrA, object A, int lda, object nnzPerRowColumn, object nnzTotalDevHostPtr):
    r"""This function computes the number of nonzero elements per row or column and the total
    number of nonzero elements in a dense matrix.

    The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSnnz__retval = hipsparseStatus_t(chipsparse.hipsparseSnnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRowColumn)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseSnnz__retval,)


@cython.embedsignature(True)
def hipsparseDnnz(object handle, object dirA, int m, int n, object descrA, object A, int lda, object nnzPerRowColumn, object nnzTotalDevHostPtr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDnnz__retval = hipsparseStatus_t(chipsparse.hipsparseDnnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRowColumn)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseDnnz__retval,)


@cython.embedsignature(True)
def hipsparseCnnz(object handle, object dirA, int m, int n, object descrA, object A, int lda, object nnzPerRowColumn, object nnzTotalDevHostPtr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCnnz__retval = hipsparseStatus_t(chipsparse.hipsparseCnnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRowColumn)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseCnnz__retval,)


@cython.embedsignature(True)
def hipsparseZnnz(object handle, object dirA, int m, int n, object descrA, object A, int lda, object nnzPerRowColumn, object nnzTotalDevHostPtr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZnnz__retval = hipsparseStatus_t(chipsparse.hipsparseZnnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(A)._ptr,lda,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRowColumn)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr))    # fully specified
    return (_hipsparseZnnz__retval,)


@cython.embedsignature(True)
def hipsparseSdense2csr(object handle, int m, int n, object descr, object A, int ld, object nnz_per_rows, object csr_val, object csr_row_ptr, object csr_col_ind):
    r"""This function converts the matrix A in dense format into a sparse matrix in CSR format.
    All the parameters are assumed to have been pre-allocated by the user and the arrays
    are filled in based on nnz_per_row, which can be pre-computed with hipsparseXnnz().
    It is executed asynchronously with respect to the host and may return control to the
    application on the host before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSdense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseSdense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_rows)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr))    # fully specified
    return (_hipsparseSdense2csr__retval,)


@cython.embedsignature(True)
def hipsparseDdense2csr(object handle, int m, int n, object descr, object A, int ld, object nnz_per_rows, object csr_val, object csr_row_ptr, object csr_col_ind):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDdense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDdense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_rows)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr))    # fully specified
    return (_hipsparseDdense2csr__retval,)


@cython.embedsignature(True)
def hipsparseCdense2csr(object handle, int m, int n, object descr, object A, int ld, object nnz_per_rows, object csr_val, object csr_row_ptr, object csr_col_ind):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCdense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseCdense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        float2.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_rows)._ptr,
        float2.from_pyobj(csr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr))    # fully specified
    return (_hipsparseCdense2csr__retval,)


@cython.embedsignature(True)
def hipsparseZdense2csr(object handle, int m, int n, object descr, object A, int ld, object nnz_per_rows, object csr_val, object csr_row_ptr, object csr_col_ind):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZdense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseZdense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        double2.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_rows)._ptr,
        double2.from_pyobj(csr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr))    # fully specified
    return (_hipsparseZdense2csr__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csr_bufferSize(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object bufferSize):
    r"""This function computes the the size of the user allocated temporary storage buffer used when converting and pruning
    a dense matrix to a CSR matrix.

    ``hipsparseXpruneDense2csr_bufferSizeExt`` returns the size of the temporary storage buffer
    that is required by hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr(). The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csr_bufferSize(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csr_bufferSizeExt(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csr_bufferSizeExt(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csrNnz(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrRowPtr, object nnzTotalDevHostPtr, object buffer):
    r"""This function computes the number of nonzero elements per row and the total number of
    nonzero elements in a dense matrix once elements less than the threshold are pruned
    from the matrix.

    The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csrNnz__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csrNnz(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrRowPtr, object nnzTotalDevHostPtr, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csrNnz__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csr(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object buffer):
    r"""This function converts the matrix A in dense format into a sparse matrix in CSR format
    while pruning values that are less than the threshold. All the parameters are assumed
    to have been pre-allocated by the user.

    The user first allocates ``csrRowPtr`` to have ``m+1`` elements and then calls
    hipsparseXpruneDense2csrNnz() which fills in the ``csrRowPtr`` array and stores the
    number of elements that are larger than the pruning threshold in ``nnzTotalDevHostPtr.``
    The user then allocates ``csrColInd`` and ``csrVal`` to have size ``nnzTotalDevHostPtr``
    and completes the conversion by calling hipsparseXpruneDense2csr(). A temporary storage
    buffer is used by both hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr() and
    must be allocated by the user and whose size is determined by
    hipsparseXpruneDense2csr_bufferSizeExt(). The routine hipsparseXpruneDense2csr() is
    executed asynchronously with respect to the host and may return control to the
    application on the host before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csr__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csr(object handle, int m, int n, object A, int lda, object threshold, object descr, object csrVal, object csrRowPtr, object csrColInd, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csr__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csrByPercentage_bufferSize(object handle, int m, int n, object A, int lda, float percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object bufferSize):
    r"""This function computes the size of the user allocated temporary storage buffer used
    when converting and pruning by percentage a dense matrix to a CSR matrix.

    When converting and pruning a dense matrix A to a CSR matrix by percentage the
    following steps are performed. First the user calls
    ``hipsparseXpruneDense2csrByPercentage_bufferSize`` which determines the size of the
    temporary storage buffer. Once determined, this buffer must be allocated by the user.
    Next the user allocates the csr_row_ptr array to have ``m+1`` elements and calls
    ``hipsparseXpruneDense2csrNnzByPercentage.`` Finally the user finishes the conversion
    by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
    at nnz_total_dev_host_ptr) and calling ``hipsparseXpruneDense2csrByPercentage.``

    The pruning by percentage works by first sorting the absolute values of the dense
    matrix ``A.`` We then determine a position in this sorted array by

    .. math::

       pos = ceil(m*n*(percentage/100)) - 1
       pos = min(pos, m*n-1)
       pos = max(pos, 0)
       threshold = sorted_A[pos]

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csrByPercentage_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csrByPercentage_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csrByPercentage_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csrByPercentage_bufferSize(object handle, int m, int n, object A, int lda, double percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csrByPercentage_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csrByPercentage_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csrByPercentage_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csrByPercentage_bufferSizeExt(object handle, int m, int n, object A, int lda, float percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object bufferSize):
    r"""This function computes the size of the user allocated temporary storage buffer used
    when converting and pruning by percentage a dense matrix to a CSR matrix.

    When converting and pruning a dense matrix A to a CSR matrix by percentage the
    following steps are performed. First the user calls
    ``hipsparseXpruneDense2csrByPercentage_bufferSizeExt`` which determines the size of the
    temporary storage buffer. Once determined, this buffer must be allocated by the user.
    Next the user allocates the csr_row_ptr array to have ``m+1`` elements and calls
    ``hipsparseXpruneDense2csrNnzByPercentage.`` Finally the user finishes the conversion
    by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
    at nnz_total_dev_host_ptr) and calling ``hipsparseXpruneDense2csrByPercentage.``

    The pruning by percentage works by first sorting the absolute values of the dense
    matrix ``A.`` We then determine a position in this sorted array by

    .. math::

       pos = ceil(m*n*(percentage/100)) - 1
       pos = min(pos, m*n-1)
       pos = max(pos, 0)
       threshold = sorted_A[pos]

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csrByPercentage_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csrByPercentage_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csrByPercentage_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csrByPercentage_bufferSizeExt(object handle, int m, int n, object A, int lda, double percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csrByPercentage_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csrByPercentage_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csrByPercentage_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csrNnzByPercentage(object handle, int m, int n, object A, int lda, float percentage, object descr, object csrRowPtr, object nnzTotalDevHostPtr, object info, object buffer):
    r"""This function computes the number of nonzero elements per row and the total number of
    nonzero elements in a dense matrix when converting and pruning by percentage a dense
    matrix to a CSR matrix.

    When converting and pruning a dense matrix A to a CSR matrix by percentage the
    following steps are performed. First the user calls
    ``hipsparseXpruneDense2csrByPercentage_bufferSize`` which determines the size of the
    temporary storage buffer. Once determined, this buffer must be allocated by the user.
    Next the user allocates the csr_row_ptr array to have ``m+1`` elements and calls
    ``hipsparseXpruneDense2csrNnzByPercentage.`` Finally the user finishes the conversion
    by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
    at nnz_total_dev_host_ptr) and calling ``hipsparseXpruneDense2csrByPercentage.``

    The pruning by percentage works by first sorting the absolute values of the dense
    matrix ``A.`` We then determine a position in this sorted array by

    .. math::

       pos = ceil(m*n*(percentage/100)) - 1
       pos = min(pos, m*n-1)
       pos = max(pos, 0)
       threshold = sorted_A[pos]

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csrNnzByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csrNnzByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csrNnzByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csrNnzByPercentage(object handle, int m, int n, object A, int lda, double percentage, object descr, object csrRowPtr, object nnzTotalDevHostPtr, object info, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csrNnzByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csrNnzByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csrNnzByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseSpruneDense2csrByPercentage(object handle, int m, int n, object A, int lda, float percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object buffer):
    r"""This function computes the number of nonzero elements per row and the total number of
    nonzero elements in a dense matrix when converting and pruning by percentage a dense
    matrix to a CSR matrix.

    When converting and pruning a dense matrix A to a CSR matrix by percentage the
    following steps are performed. First the user calls
    ``hipsparseXpruneDense2csrByPercentage_bufferSize`` which determines the size of the
    temporary storage buffer. Once determined, this buffer must be allocated by the user.
    Next the user allocates the csr_row_ptr array to have ``m+1`` elements and calls
    ``hipsparseXpruneDense2csrNnzByPercentage.`` Finally the user finishes the conversion
    by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
    at nnz_total_dev_host_ptr) and calling ``hipsparseXpruneDense2csrByPercentage.``

    The pruning by percentage works by first sorting the absolute values of the dense
    matrix ``A.`` We then determine a position in this sorted array by

    .. math::

       pos = ceil(m*n*(percentage/100)) - 1
       pos = min(pos, m*n-1)
       pos = max(pos, 0)
       threshold = sorted_A[pos]

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneDense2csrByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneDense2csrByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneDense2csrByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseDpruneDense2csrByPercentage(object handle, int m, int n, object A, int lda, double percentage, object descr, object csrVal, object csrRowPtr, object csrColInd, object info, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneDense2csrByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneDense2csrByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,lda,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneDense2csrByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseSdense2csc(object handle, int m, int n, object descr, object A, int ld, object nnz_per_columns, object csc_val, object csc_row_ind, object csc_col_ptr):
    r"""

    This function converts the matrix A in dense format into a sparse matrix in CSC format.
    All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with hipsparseXnnz().
    It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSdense2csc__retval = hipsparseStatus_t(chipsparse.hipsparseSdense2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_columns)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr))    # fully specified
    return (_hipsparseSdense2csc__retval,)


@cython.embedsignature(True)
def hipsparseDdense2csc(object handle, int m, int n, object descr, object A, int ld, object nnz_per_columns, object csc_val, object csc_row_ind, object csc_col_ptr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDdense2csc__retval = hipsparseStatus_t(chipsparse.hipsparseDdense2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_columns)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr))    # fully specified
    return (_hipsparseDdense2csc__retval,)


@cython.embedsignature(True)
def hipsparseCdense2csc(object handle, int m, int n, object descr, object A, int ld, object nnz_per_columns, object csc_val, object csc_row_ind, object csc_col_ptr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCdense2csc__retval = hipsparseStatus_t(chipsparse.hipsparseCdense2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        float2.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_columns)._ptr,
        float2.from_pyobj(csc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr))    # fully specified
    return (_hipsparseCdense2csc__retval,)


@cython.embedsignature(True)
def hipsparseZdense2csc(object handle, int m, int n, object descr, object A, int ld, object nnz_per_columns, object csc_val, object csc_row_ind, object csc_col_ptr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZdense2csc__retval = hipsparseStatus_t(chipsparse.hipsparseZdense2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        double2.from_pyobj(A)._ptr,ld,
        <const int *>hip._util.types.Pointer.from_pyobj(nnz_per_columns)._ptr,
        double2.from_pyobj(csc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr))    # fully specified
    return (_hipsparseZdense2csc__retval,)


@cython.embedsignature(True)
def hipsparseScsr2dense(object handle, int m, int n, object descr, object csr_val, object csr_row_ptr, object csr_col_ind, object A, int ld):
    r"""This function converts the sparse matrix in CSR format into a dense matrix.
    It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsr2dense__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseScsr2dense__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2dense(object handle, int m, int n, object descr, object csr_val, object csr_row_ptr, object csr_col_ind, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsr2dense__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseDcsr2dense__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2dense(object handle, int m, int n, object descr, object csr_val, object csr_row_ptr, object csr_col_ind, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsr2dense__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        float2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        float2.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseCcsr2dense__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2dense(object handle, int m, int n, object descr, object csr_val, object csr_row_ptr, object csr_col_ind, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsr2dense__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        double2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        double2.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseZcsr2dense__retval,)


@cython.embedsignature(True)
def hipsparseScsc2dense(object handle, int m, int n, object descr, object csc_val, object csc_row_ind, object csc_col_ptr, object A, int ld):
    r"""This function converts the sparse matrix in CSC format into a dense matrix.
    It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsc2dense__retval = hipsparseStatus_t(chipsparse.hipsparseScsc2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csc_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseScsc2dense__retval,)


@cython.embedsignature(True)
def hipsparseDcsc2dense(object handle, int m, int n, object descr, object csc_val, object csc_row_ind, object csc_col_ptr, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsc2dense__retval = hipsparseStatus_t(chipsparse.hipsparseDcsc2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csc_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseDcsc2dense__retval,)


@cython.embedsignature(True)
def hipsparseCcsc2dense(object handle, int m, int n, object descr, object csc_val, object csc_row_ind, object csc_col_ptr, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsc2dense__retval = hipsparseStatus_t(chipsparse.hipsparseCcsc2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        float2.from_pyobj(csc_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr,
        float2.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseCcsc2dense__retval,)


@cython.embedsignature(True)
def hipsparseZcsc2dense(object handle, int m, int n, object descr, object csc_val, object csc_row_ind, object csc_col_ptr, object A, int ld):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsc2dense__retval = hipsparseStatus_t(chipsparse.hipsparseZcsc2dense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descr)._ptr,
        double2.from_pyobj(csc_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_row_ind)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csc_col_ptr)._ptr,
        double2.from_pyobj(A)._ptr,ld))    # fully specified
    return (_hipsparseZcsc2dense__retval,)


@cython.embedsignature(True)
def hipsparseXcsr2bsrNnz(object handle, object dirA, int m, int n, object descrA, object csrRowPtrA, object csrColIndA, int blockDim, object descrC, object bsrRowPtrC, object bsrNnzb):
    r"""This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
    BSR matrix given a sparse CSR matrix as input.

    The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseXcsr2bsrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsr2bsrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrNnzb)._ptr))    # fully specified
    return (_hipsparseXcsr2bsrNnz__retval,)


@cython.embedsignature(True)
def hipsparseSnnz_compress(object handle, int m, object descrA, object csrValA, object csrRowPtrA, object nnzPerRow, object nnzC, float tol):
    r"""(No short description)

    Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
    in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
    of remaining elements in the matrix.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSnnz_compress__retval = hipsparseStatus_t(chipsparse.hipsparseSnnz_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzC)._ptr,tol))    # fully specified
    return (_hipsparseSnnz_compress__retval,)


@cython.embedsignature(True)
def hipsparseDnnz_compress(object handle, int m, object descrA, object csrValA, object csrRowPtrA, object nnzPerRow, object nnzC, double tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnnz_compress__retval = hipsparseStatus_t(chipsparse.hipsparseDnnz_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzC)._ptr,tol))    # fully specified
    return (_hipsparseDnnz_compress__retval,)


@cython.embedsignature(True)
def hipsparseCnnz_compress(object handle, int m, object descrA, object csrValA, object csrRowPtrA, object nnzPerRow, object nnzC, object tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCnnz_compress__retval = hipsparseStatus_t(chipsparse.hipsparseCnnz_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzC)._ptr,
        float2.from_pyobj(tol)._ptr[0]))    # fully specified
    return (_hipsparseCnnz_compress__retval,)


@cython.embedsignature(True)
def hipsparseZnnz_compress(object handle, int m, object descrA, object csrValA, object csrRowPtrA, object nnzPerRow, object nnzC, object tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZnnz_compress__retval = hipsparseStatus_t(chipsparse.hipsparseZnnz_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzC)._ptr,
        double2.from_pyobj(tol)._ptr[0]))    # fully specified
    return (_hipsparseZnnz_compress__retval,)


@cython.embedsignature(True)
def hipsparseXcsr2coo(object handle, object csrRowPtr, int nnz, int m, object cooRowInd, object idxBase):
    r"""Convert a sparse CSR matrix into a sparse COO matrix

    ``hipsparseXcsr2coo`` converts the CSR array containing the row offsets, that point
    to the start of every row, into a COO array of row indices.

    Note:
        It can also be used to convert a CSC array containing the column offsets into a COO
        array of column indices.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseXcsr2coo__retval = hipsparseStatus_t(chipsparse.hipsparseXcsr2coo(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,nnz,m,
        <int *>hip._util.types.Pointer.from_pyobj(cooRowInd)._ptr,idxBase.value))    # fully specified
    return (_hipsparseXcsr2coo__retval,)


@cython.embedsignature(True)
def hipsparseScsr2csc(object handle, int m, int n, int nnz, object csrSortedVal, object csrSortedRowPtr, object csrSortedColInd, object cscSortedVal, object cscSortedRowInd, object cscSortedColPtr, object copyValues, object idxBase):
    r"""Convert a sparse CSR matrix into a sparse CSC matrix

    ``hipsparseXcsr2csc`` converts a CSR matrix into a CSC matrix. ``hipsparseXcsr2csc``
    can also be used to convert a CSC matrix into a CSR matrix. ``copy_values`` decides
    whether ``csc_val`` is being filled during conversion (``HIPSPARSE_ACTION_NUMERIC`` )
    or not (``HIPSPARSE_ACTION_SYMBOLIC`` ).

    Note:
        The resulting matrix can also be seen as the transpose of the input matrix.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseScsr2csc__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColInd)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(cscSortedVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedRowInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedColPtr)._ptr,copyValues.value,idxBase.value))    # fully specified
    return (_hipsparseScsr2csc__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2csc(object handle, int m, int n, int nnz, object csrSortedVal, object csrSortedRowPtr, object csrSortedColInd, object cscSortedVal, object cscSortedRowInd, object cscSortedColPtr, object copyValues, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDcsr2csc__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColInd)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(cscSortedVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedRowInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedColPtr)._ptr,copyValues.value,idxBase.value))    # fully specified
    return (_hipsparseDcsr2csc__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2csc(object handle, int m, int n, int nnz, object csrSortedVal, object csrSortedRowPtr, object csrSortedColInd, object cscSortedVal, object cscSortedRowInd, object cscSortedColPtr, object copyValues, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCcsr2csc__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        float2.from_pyobj(csrSortedVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColInd)._ptr,
        float2.from_pyobj(cscSortedVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedRowInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedColPtr)._ptr,copyValues.value,idxBase.value))    # fully specified
    return (_hipsparseCcsr2csc__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2csc(object handle, int m, int n, int nnz, object csrSortedVal, object csrSortedRowPtr, object csrSortedColInd, object cscSortedVal, object cscSortedRowInd, object cscSortedColPtr, object copyValues, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZcsr2csc__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2csc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        double2.from_pyobj(csrSortedVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColInd)._ptr,
        double2.from_pyobj(cscSortedVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedRowInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscSortedColPtr)._ptr,copyValues.value,idxBase.value))    # fully specified
    return (_hipsparseZcsr2csc__retval,)


class _hipsparseCsr2CscAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseCsr2CscAlg_t(_hipsparseCsr2CscAlg_t__Base):
    HIPSPARSE_CSR2CSC_ALG1 = chipsparse.HIPSPARSE_CSR2CSC_ALG1
    HIPSPARSE_CSR2CSC_ALG2 = chipsparse.HIPSPARSE_CSR2CSC_ALG2
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def hipsparseCsr2cscEx2_bufferSize(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object cscVal, object cscColPtr, object cscRowInd, object valType, object copyValues, object idxBase, object alg, object bufferSize):
    r"""This function computes the size of the user allocated temporary storage buffer used
    when converting a sparse CSR matrix into a sparse CSC matrix.

    ``hipsparseXcsr2cscEx2_bufferSize`` calculates the required user allocated temporary buffer needed
    by ``hipsparseXcsr2cscEx2`` to convert a CSR matrix into a CSC matrix. ``hipsparseXcsr2cscEx2``
    can also be used to convert a CSC matrix into a CSR matrix. ``copy_values`` decides
    whether ``csc_val`` is being filled during conversion (``HIPSPARSE_ACTION_NUMERIC`` )
    or not (``HIPSPARSE_ACTION_SYMBOLIC`` ).

    Note:
        The resulting matrix can also be seen as the transpose of the input matrix.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(valType,_hipDataType__Base):
        raise TypeError("argument 'valType' must be of type '_hipDataType__Base'")                    
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(alg,_hipsparseCsr2CscAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseCsr2CscAlg_t__Base'")
    _hipsparseCsr2cscEx2_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCsr2cscEx2_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const void *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscColPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,valType.value,copyValues.value,idxBase.value,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseCsr2cscEx2_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCsr2cscEx2(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object cscVal, object cscColPtr, object cscRowInd, object valType, object copyValues, object idxBase, object alg, object buffer):
    r"""Convert a sparse CSR matrix into a sparse CSC matrix

    ``hipsparseXcsr2cscEx2`` converts a CSR matrix into a CSC matrix. ``hipsparseXcsr2cscEx2``
    can also be used to convert a CSC matrix into a CSR matrix. ``copy_values`` decides
    whether ``csc_val`` is being filled during conversion (``HIPSPARSE_ACTION_NUMERIC`` )
    or not (``HIPSPARSE_ACTION_SYMBOLIC`` ).

    Note:
        The resulting matrix can also be seen as the transpose of the input matrix.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(valType,_hipDataType__Base):
        raise TypeError("argument 'valType' must be of type '_hipDataType__Base'")                    
    if not isinstance(copyValues,_hipsparseAction_t__Base):
        raise TypeError("argument 'copyValues' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(alg,_hipsparseCsr2CscAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseCsr2CscAlg_t__Base'")
    _hipsparseCsr2cscEx2__retval = hipsparseStatus_t(chipsparse.hipsparseCsr2cscEx2(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const void *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscVal)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscColPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,valType.value,copyValues.value,idxBase.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseCsr2cscEx2__retval,)


@cython.embedsignature(True)
def hipsparseScsr2hyb(object handle, int m, int n, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object hybA, int userEllWidth, object partitionType):
    r"""Convert a sparse CSR matrix into a sparse HYB matrix

    ``hipsparseXcsr2hyb`` converts a CSR matrix into a HYB matrix. It is assumed
    that ``hyb`` has been initialized with hipsparseCreateHybMat().

    Note:
        This function requires a significant amount of storage for the HYB matrix,
        depending on the matrix structure.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(partitionType,_hipsparseHybPartition_t__Base):
        raise TypeError("argument 'partitionType' must be of type '_hipsparseHybPartition_t__Base'")
    _hipsparseScsr2hyb__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2hyb(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(hybA)._ptr,userEllWidth,partitionType.value))    # fully specified
    return (_hipsparseScsr2hyb__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2hyb(object handle, int m, int n, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object hybA, int userEllWidth, object partitionType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(partitionType,_hipsparseHybPartition_t__Base):
        raise TypeError("argument 'partitionType' must be of type '_hipsparseHybPartition_t__Base'")
    _hipsparseDcsr2hyb__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2hyb(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(hybA)._ptr,userEllWidth,partitionType.value))    # fully specified
    return (_hipsparseDcsr2hyb__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2hyb(object handle, int m, int n, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object hybA, int userEllWidth, object partitionType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(partitionType,_hipsparseHybPartition_t__Base):
        raise TypeError("argument 'partitionType' must be of type '_hipsparseHybPartition_t__Base'")
    _hipsparseCcsr2hyb__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2hyb(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(hybA)._ptr,userEllWidth,partitionType.value))    # fully specified
    return (_hipsparseCcsr2hyb__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2hyb(object handle, int m, int n, object descrA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA, object hybA, int userEllWidth, object partitionType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(partitionType,_hipsparseHybPartition_t__Base):
        raise TypeError("argument 'partitionType' must be of type '_hipsparseHybPartition_t__Base'")
    _hipsparseZcsr2hyb__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2hyb(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(hybA)._ptr,userEllWidth,partitionType.value))    # fully specified
    return (_hipsparseZcsr2hyb__retval,)


@cython.embedsignature(True)
def hipsparseSgebsr2gebsc_bufferSize(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix

    ``hipsparseXgebsr2gebsc_bufferSize`` returns the size of the temporary storage buffer
    required by hipsparseXgebsr2gebsc().
    The temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSgebsr2gebsc_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSgebsr2gebsc_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseSgebsr2gebsc_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDgebsr2gebsc_bufferSize(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDgebsr2gebsc_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDgebsr2gebsc_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseDgebsr2gebsc_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCgebsr2gebsc_bufferSize(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCgebsr2gebsc_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCgebsr2gebsc_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        float2.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseCgebsr2gebsc_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZgebsr2gebsc_bufferSize(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZgebsr2gebsc_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZgebsr2gebsc_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        double2.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseZgebsr2gebsc_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSgebsr2gebsc(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object bsc_val, object bsc_row_ind, object bsc_col_ptr, object copy_values, object idx_base, object temp_buffer):
    r"""Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix

    ``hipsparseXgebsr2gebsc`` converts a GEneral BSR matrix into a GEneral BSC matrix. ``hipsparseXgebsr2gebsc``
    can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. ``copy_values`` decides
    whether ``bsc_val`` is being filled during conversion (``HIPSPARSE_ACTION_NUMERIC`` )
    or not (``HIPSPARSE_ACTION_SYMBOLIC`` ).

    ``hipsparseXgebsr2gebsc`` requires extra temporary storage buffer that has to be allocated
    by the user. Storage buffer size can be determined by hipsparseXgebsr2gebsc_bufferSize().

    Note:
        The resulting matrix can also be seen as the transpose of the input matrix.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copy_values,_hipsparseAction_t__Base):
        raise TypeError("argument 'copy_values' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idx_base,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idx_base' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseSgebsr2gebsc__retval = hipsparseStatus_t(chipsparse.hipsparseSgebsr2gebsc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        <const float *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <float *>hip._util.types.Pointer.from_pyobj(bsc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_col_ptr)._ptr,copy_values.value,idx_base.value,
        <void *>hip._util.types.Pointer.from_pyobj(temp_buffer)._ptr))    # fully specified
    return (_hipsparseSgebsr2gebsc__retval,)


@cython.embedsignature(True)
def hipsparseDgebsr2gebsc(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object bsc_val, object bsc_row_ind, object bsc_col_ptr, object copy_values, object idx_base, object temp_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copy_values,_hipsparseAction_t__Base):
        raise TypeError("argument 'copy_values' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idx_base,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idx_base' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseDgebsr2gebsc__retval = hipsparseStatus_t(chipsparse.hipsparseDgebsr2gebsc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        <const double *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <double *>hip._util.types.Pointer.from_pyobj(bsc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_col_ptr)._ptr,copy_values.value,idx_base.value,
        <void *>hip._util.types.Pointer.from_pyobj(temp_buffer)._ptr))    # fully specified
    return (_hipsparseDgebsr2gebsc__retval,)


@cython.embedsignature(True)
def hipsparseCgebsr2gebsc(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object bsc_val, object bsc_row_ind, object bsc_col_ptr, object copy_values, object idx_base, object temp_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copy_values,_hipsparseAction_t__Base):
        raise TypeError("argument 'copy_values' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idx_base,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idx_base' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseCgebsr2gebsc__retval = hipsparseStatus_t(chipsparse.hipsparseCgebsr2gebsc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        float2.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        float2.from_pyobj(bsc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_col_ptr)._ptr,copy_values.value,idx_base.value,
        <void *>hip._util.types.Pointer.from_pyobj(temp_buffer)._ptr))    # fully specified
    return (_hipsparseCgebsr2gebsc__retval,)


@cython.embedsignature(True)
def hipsparseZgebsr2gebsc(object handle, int mb, int nb, int nnzb, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object bsc_val, object bsc_row_ind, object bsc_col_ptr, object copy_values, object idx_base, object temp_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(copy_values,_hipsparseAction_t__Base):
        raise TypeError("argument 'copy_values' must be of type '_hipsparseAction_t__Base'")                    
    if not isinstance(idx_base,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idx_base' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseZgebsr2gebsc__retval = hipsparseStatus_t(chipsparse.hipsparseZgebsr2gebsc(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,mb,nb,nnzb,
        double2.from_pyobj(bsr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        double2.from_pyobj(bsc_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_row_ind)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsc_col_ptr)._ptr,copy_values.value,idx_base.value,
        <void *>hip._util.types.Pointer.from_pyobj(temp_buffer)._ptr))    # fully specified
    return (_hipsparseZgebsr2gebsc__retval,)


@cython.embedsignature(True)
def hipsparseScsr2gebsr_bufferSize(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""

    ``hipsparseXcsr2gebsr_bufferSize`` returns the size of the temporary buffer that
    is required by ``hipsparseXcsr2gebcsrNnz`` and ``hipsparseXcsr2gebcsr.``
    The temporary storage buffer must be allocated by the user.

    This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
    GEneral BSR matrix given a sparse CSR matrix as input.

    The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseScsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseScsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2gebsr_bufferSize(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDcsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseDcsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2gebsr_bufferSize(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCcsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        float2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseCcsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2gebsr_bufferSize(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, int row_block_dim, int col_block_dim, object p_buffer_size):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZcsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        double2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,row_block_dim,col_block_dim,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(p_buffer_size)._ptr))    # fully specified
    return (_hipsparseZcsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseXcsr2gebsrNnz(object handle, object dir, int m, int n, object csr_descr, object csr_row_ptr, object csr_col_ind, object bsr_descr, object bsr_row_ptr, int row_block_dim, int col_block_dim, object bsr_nnz_devhost, object p_buffer):
    r"""This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
    GEneral BSR matrix given a sparse CSR matrix as input.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseXcsr2gebsrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseXcsr2gebsrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(bsr_descr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,row_block_dim,col_block_dim,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_nnz_devhost)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(p_buffer)._ptr))    # fully specified
    return (_hipsparseXcsr2gebsrNnz__retval,)


@cython.embedsignature(True)
def hipsparseScsr2gebsr(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, object bsr_descr, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer):
    r"""Convert a sparse CSR matrix into a sparse GEneral BSR matrix

    ``hipsparseXcsr2gebsr`` converts a CSR matrix into a GEneral BSR matrix. It is assumed,
    that ``bsr_val,`` ``bsr_col_ind`` and ``bsr_row_ptr`` are allocated. Allocation size
    for ``bsr_row_ptr`` is computed as ``mb+1`` where ``mb`` is the number of block rows in
    the GEneral BSR matrix. Allocation size for ``bsr_val`` and ``bsr_col_ind`` is computed using
    ``csr2gebsr_nnz()`` which also fills in ``bsr_row_ptr.``

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseScsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(bsr_descr)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <void *>hip._util.types.Pointer.from_pyobj(p_buffer)._ptr))    # fully specified
    return (_hipsparseScsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2gebsr(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, object bsr_descr, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDcsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(bsr_descr)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <void *>hip._util.types.Pointer.from_pyobj(p_buffer)._ptr))    # fully specified
    return (_hipsparseDcsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2gebsr(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, object bsr_descr, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCcsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        float2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(bsr_descr)._ptr,
        float2.from_pyobj(bsr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <void *>hip._util.types.Pointer.from_pyobj(p_buffer)._ptr))    # fully specified
    return (_hipsparseCcsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2gebsr(object handle, object dir, int m, int n, object csr_descr, object csr_val, object csr_row_ptr, object csr_col_ind, object bsr_descr, object bsr_val, object bsr_row_ptr, object bsr_col_ind, int row_block_dim, int col_block_dim, object p_buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dir,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dir' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZcsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dir.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(csr_descr)._ptr,
        double2.from_pyobj(csr_val)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_row_ptr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csr_col_ind)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(bsr_descr)._ptr,
        double2.from_pyobj(bsr_val)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_row_ptr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsr_col_ind)._ptr,row_block_dim,col_block_dim,
        <void *>hip._util.types.Pointer.from_pyobj(p_buffer)._ptr))    # fully specified
    return (_hipsparseZcsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseScsr2bsr(object handle, object dirA, int m, int n, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, int blockDim, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC):
    r"""Convert a sparse CSR matrix into a sparse BSR matrix

    ``hipsparseXcsr2bsr`` converts a CSR matrix into a BSR matrix. It is assumed,
    that ``bsr_val,`` ``bsr_col_ind`` and ``bsr_row_ptr`` are allocated. Allocation size
    for ``bsr_row_ptr`` is computed as ``mb+1`` where ``mb`` is the number of block rows in
    the BSR matrix. Allocation size for ``bsr_val`` and ``bsr_col_ind`` is computed using
    ``csr2bsr_nnz()`` which also fills in ``bsr_row_ptr.``

    ``hipsparseXcsr2bsr`` requires extra temporary storage that is allocated internally if
    ``block_dim>16``

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseScsr2bsr__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2bsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr))    # fully specified
    return (_hipsparseScsr2bsr__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2bsr(object handle, object dirA, int m, int n, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, int blockDim, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDcsr2bsr__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2bsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr))    # fully specified
    return (_hipsparseDcsr2bsr__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2bsr(object handle, object dirA, int m, int n, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, int blockDim, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCcsr2bsr__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2bsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr))    # fully specified
    return (_hipsparseCcsr2bsr__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2bsr(object handle, object dirA, int m, int n, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, int blockDim, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZcsr2bsr__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2bsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr))    # fully specified
    return (_hipsparseZcsr2bsr__retval,)


@cython.embedsignature(True)
def hipsparseSbsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""Convert a sparse BSR matrix into a sparse CSR matrix

    ``hipsparseXbsr2csr`` converts a BSR matrix into a CSR matrix. It is assumed,
    that ``csr_val,`` ``csr_col_ind`` and ``csr_row_ptr`` are allocated. Allocation size
    for ``csr_row_ptr`` is computed by the number of block rows multiplied by the block
    dimension plus one. Allocation for ``csr_val`` and ``csr_col_ind`` is computed by the
    the number of blocks in the BSR matrix multiplied by the block dimension squared.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSbsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseSbsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseSbsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseDbsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDbsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDbsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseDbsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseCbsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCbsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseCbsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseCbsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseZbsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int blockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZbsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseZbsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,blockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseZbsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseSgebsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDim, int colBlockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""Convert a sparse general BSR matrix into a sparse CSR matrix

    ``hipsparseXgebsr2csr`` converts a BSR matrix into a CSR matrix. It is assumed,
    that ``csr_val,`` ``csr_col_ind`` and ``csr_row_ptr`` are allocated. Allocation size
    for ``csr_row_ptr`` is computed by the number of block rows multiplied by the block
    dimension plus one. Allocation for ``csr_val`` and ``csr_col_ind`` is computed by the
    the number of blocks in the BSR matrix multiplied by the product of the block dimensions.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSgebsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseSgebsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDim,colBlockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseSgebsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseDgebsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDim, int colBlockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDgebsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDgebsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDim,colBlockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseDgebsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseCgebsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDim, int colBlockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCgebsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseCgebsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDim,colBlockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseCgebsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseZgebsr2csr(object handle, object dirA, int mb, int nb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDim, int colBlockDim, object descrC, object csrValC, object csrRowPtrC, object csrColIndC):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZgebsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseZgebsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDim,colBlockDim,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr))    # fully specified
    return (_hipsparseZgebsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseScsr2csr_compress(object handle, int m, int n, object descrA, object csrValA, object csrColIndA, object csrRowPtrA, int nnzA, object nnzPerRow, object csrValC, object csrColIndC, object csrRowPtrC, float tol):
    r"""Convert a sparse CSR matrix into a compressed sparse CSR matrix

    ``hipsparseXcsr2csr_compress`` converts a CSR matrix into a compressed CSR matrix by
    removing entries in the input CSR matrix that are below a non-negative threshold ``tol``

    Note:
        In the case of complex matrices only the magnitude of the real part of ``tol`` is used.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsr2csr_compress__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2csr_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,tol))    # fully specified
    return (_hipsparseScsr2csr_compress__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2csr_compress(object handle, int m, int n, object descrA, object csrValA, object csrColIndA, object csrRowPtrA, int nnzA, object nnzPerRow, object csrValC, object csrColIndC, object csrRowPtrC, double tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsr2csr_compress__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2csr_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,tol))    # fully specified
    return (_hipsparseDcsr2csr_compress__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2csr_compress(object handle, int m, int n, object descrA, object csrValA, object csrColIndA, object csrRowPtrA, int nnzA, object nnzPerRow, object csrValC, object csrColIndC, object csrRowPtrC, object tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsr2csr_compress__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2csr_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        float2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        float2.from_pyobj(tol)._ptr[0]))    # fully specified
    return (_hipsparseCcsr2csr_compress__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2csr_compress(object handle, int m, int n, object descrA, object csrValA, object csrColIndA, object csrRowPtrA, int nnzA, object nnzPerRow, object csrValC, object csrColIndC, object csrRowPtrC, object tol):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsr2csr_compress__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2csr_compress(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,nnzA,
        <const int *>hip._util.types.Pointer.from_pyobj(nnzPerRow)._ptr,
        double2.from_pyobj(csrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        double2.from_pyobj(tol)._ptr[0]))    # fully specified
    return (_hipsparseZcsr2csr_compress__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csr_bufferSize(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object bufferSize):
    r"""Convert and prune sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csr_bufferSize`` returns the size of the temporary buffer that
    is required by ``hipsparseXpruneCsr2csrNnz`` and hipsparseXpruneCsr2csr. The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csr_bufferSize(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csr_bufferSizeExt(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object bufferSize):
    r"""Convert and prune sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csr_bufferSizeExt`` returns the size of the temporary buffer that
    is required by ``hipsparseXpruneCsr2csrNnz`` and hipsparseXpruneCsr2csr. The
    temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csr_bufferSizeExt(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csrNnz(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr, object buffer):
    r"""Convert and prune sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csrNnz`` computes the number of nonzero elements per row and the total
    number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
    pruned from the matrix.

    Note:
        The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csrNnz__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csrNnz(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csrNnz__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csr(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object buffer):
    r"""Convert and prune sparse CSR matrix into a sparse CSR matrix

    This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
    that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
    The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
    by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
    allocates ``csr_row_ptr_C`` to have ``m+1`` elements and then calls hipsparseXpruneCsr2csrNnz() which fills
    in the ``csr_row_ptr_C`` array stores then number of elements that are larger than the pruning threshold
    in ``nnz_total_dev_host_ptr.`` The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
    is executed asynchronously with respect to the host and may return control to the application on the host
    before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csr(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object threshold, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(threshold)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csr__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csrByPercentage_bufferSize(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, float percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object bufferSize):
    r"""Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csrByPercentage_bufferSize`` returns the size of the temporary buffer that
    is required by ``hipsparseXpruneCsr2csrNnzByPercentage.``
    The temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csrByPercentage_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csrByPercentage_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csrByPercentage_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csrByPercentage_bufferSize(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, double percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csrByPercentage_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csrByPercentage_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csrByPercentage_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, float percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object bufferSize):
    r"""Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csrByPercentage_bufferSizeExt`` returns the size of the temporary buffer that
    is required by ``hipsparseXpruneCsr2csrNnzByPercentage.``
    The temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, double percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csrNnzByPercentage(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, float percentage, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr, object info, object buffer):
    r"""Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix

    ``hipsparseXpruneCsr2csrNnzByPercentage`` computes the number of nonzero elements per row and the total
    number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
    pruned from the matrix.

    Note:
        The routine does support asynchronous execution if the pointer mode is set to device.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csrNnzByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csrNnzByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csrNnzByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csrNnzByPercentage(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, double percentage, object descrC, object csrRowPtrC, object nnzTotalDevHostPtr, object info, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csrNnzByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csrNnzByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csrNnzByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseSpruneCsr2csrByPercentage(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, float percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object buffer):
    r"""Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix

    This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
    that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
    The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
    by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
    allocates ``csr_row_ptr_C`` to have ``m+1`` elements and then calls hipsparseXpruneCsr2csrNnz() which fills
    in the ``csr_row_ptr_C`` array stores then number of elements that are larger than the pruning threshold
    in ``nnz_total_dev_host_ptr.`` The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
    is executed asynchronously with respect to the host and may return control to the application on the host
    before the entire result is ready.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpruneCsr2csrByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseSpruneCsr2csrByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSpruneCsr2csrByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseDpruneCsr2csrByPercentage(object handle, int m, int n, int nnzA, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, double percentage, object descrC, object csrValC, object csrRowPtrC, object csrColIndC, object info, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDpruneCsr2csrByPercentage__retval = hipsparseStatus_t(chipsparse.hipsparseDpruneCsr2csrByPercentage(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnzA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,percentage,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrValC)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColIndC)._ptr,
        pruneInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDpruneCsr2csrByPercentage__retval,)


@cython.embedsignature(True)
def hipsparseShyb2csr(object handle, object descrA, object hybA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA):
    r"""Convert a sparse HYB matrix into a sparse CSR matrix

    ``hipsparseXhyb2csr`` converts a HYB matrix into a CSR matrix.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseShyb2csr__retval = hipsparseStatus_t(chipsparse.hipsparseShyb2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr))    # fully specified
    return (_hipsparseShyb2csr__retval,)


@cython.embedsignature(True)
def hipsparseDhyb2csr(object handle, object descrA, object hybA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDhyb2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDhyb2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrSortedValA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr))    # fully specified
    return (_hipsparseDhyb2csr__retval,)


@cython.embedsignature(True)
def hipsparseChyb2csr(object handle, object descrA, object hybA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseChyb2csr__retval = hipsparseStatus_t(chipsparse.hipsparseChyb2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        float2.from_pyobj(csrSortedValA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr))    # fully specified
    return (_hipsparseChyb2csr__retval,)


@cython.embedsignature(True)
def hipsparseZhyb2csr(object handle, object descrA, object hybA, object csrSortedValA, object csrSortedRowPtrA, object csrSortedColIndA):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZhyb2csr__retval = hipsparseStatus_t(chipsparse.hipsparseZhyb2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(hybA)._ptr,
        double2.from_pyobj(csrSortedValA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedRowPtrA)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrSortedColIndA)._ptr))    # fully specified
    return (_hipsparseZhyb2csr__retval,)


@cython.embedsignature(True)
def hipsparseXcoo2csr(object handle, object cooRowInd, int nnz, int m, object csrRowPtr, object idxBase):
    r"""Convert a sparse COO matrix into a sparse CSR matrix

    ``hipsparseXcoo2csr`` converts the COO array containing the row indices into a
    CSR array of row offsets, that point to the start of every row.
    It is assumed that the COO row index array is sorted.

    Note:
        It can also be used, to convert a COO array containing the column indices into
        a CSC array of column offsets, that point to the start of every column. Then, it is
        assumed that the COO column index array is sorted, instead.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")
    _hipsparseXcoo2csr__retval = hipsparseStatus_t(chipsparse.hipsparseXcoo2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cooRowInd)._ptr,nnz,m,
        <int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,idxBase.value))    # fully specified
    return (_hipsparseXcoo2csr__retval,)


@cython.embedsignature(True)
def hipsparseCreateIdentityPermutation(object handle, int n, object p):
    r"""Create the identity map

    ``hipsparseCreateIdentityPermutation`` stores the identity map in ``p,`` such that
    :math:`p = 0:1:(n-1)`.

    .. code-block::

       for(i = 0; i < n; ++i)
       {
           p[i] = i;
       }

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCreateIdentityPermutation__retval = hipsparseStatus_t(chipsparse.hipsparseCreateIdentityPermutation(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,n,
        <int *>hip._util.types.Pointer.from_pyobj(p)._ptr))    # fully specified
    return (_hipsparseCreateIdentityPermutation__retval,)


@cython.embedsignature(True)
def hipsparseXcsrsort_bufferSizeExt(object handle, int m, int n, int nnz, object csrRowPtr, object csrColInd, object pBufferSizeInBytes):
    r"""Sort a sparse CSR matrix

    ``hipsparseXcsrsort_bufferSizeExt`` returns the size of the temporary storage buffer
    required by hipsparseXcsrsort(). The temporary storage buffer must be allocated by
    the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrsort_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrsort_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseXcsrsort_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseXcsrsort(object handle, int m, int n, int nnz, object descrA, object csrRowPtr, object csrColInd, object P, object pBuffer):
    r"""Sort a sparse CSR matrix

    ``hipsparseXcsrsort`` sorts a matrix in CSR format. The sorted permutation vector
    ``perm`` can be used to obtain sorted ``csr_val`` array. In this case, ``perm`` must be
    initialized as the identity permutation, see hipsparseCreateIdentityPermutation().

    ``hipsparseXcsrsort`` requires extra temporary storage buffer that has to be allocated by
    the user. Storage buffer size can be determined by hipsparseXcsrsort_bufferSizeExt().

    Note:
        ``perm`` can be ``NULL`` if a sorted permutation vector is not required.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcsrsort__retval = hipsparseStatus_t(chipsparse.hipsparseXcsrsort(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(P)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseXcsrsort__retval,)


@cython.embedsignature(True)
def hipsparseXcscsort_bufferSizeExt(object handle, int m, int n, int nnz, object cscColPtr, object cscRowInd, object pBufferSizeInBytes):
    r"""Sort a sparse CSC matrix

    ``hipsparseXcscsort_bufferSizeExt`` returns the size of the temporary storage buffer
    required by hipsparseXcscsort(). The temporary storage buffer must be allocated by
    the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcscsort_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseXcscsort_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtr)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseXcscsort_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseXcscsort(object handle, int m, int n, int nnz, object descrA, object cscColPtr, object cscRowInd, object P, object pBuffer):
    r"""Sort a sparse CSC matrix

    ``hipsparseXcscsort`` sorts a matrix in CSC format. The sorted permutation vector
    ``perm`` can be used to obtain sorted ``csc_val`` array. In this case, ``perm`` must be
    initialized as the identity permutation, see hipsparseCreateIdentityPermutation().

    ``hipsparseXcscsort`` requires extra temporary storage buffer that has to be allocated by
    the user. Storage buffer size can be determined by hipsparseXcscsort_bufferSizeExt().

    Note:
        ``perm`` can be ``NULL`` if a sorted permutation vector is not required.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcscsort__retval = hipsparseStatus_t(chipsparse.hipsparseXcscsort(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cscColPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(P)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseXcscsort__retval,)


@cython.embedsignature(True)
def hipsparseXcoosort_bufferSizeExt(object handle, int m, int n, int nnz, object cooRows, object cooCols, object pBufferSizeInBytes):
    r"""Sort a sparse COO matrix

    ``hipsparseXcoosort_bufferSizeExt`` returns the size of the temporary storage buffer
    required by hipsparseXcoosort(). The temporary storage buffer must be allocated by
    the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcoosort_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseXcoosort_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <const int *>hip._util.types.Pointer.from_pyobj(cooRows)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(cooCols)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseXcoosort_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseXcoosortByRow(object handle, int m, int n, int nnz, object cooRows, object cooCols, object P, object pBuffer):
    r"""Sort a sparse COO matrix by row

    ``hipsparseXcoosortByRow`` sorts a matrix in COO format by row. The sorted
    permutation vector ``perm`` can be used to obtain sorted ``coo_val`` array. In this
    case, ``perm`` must be initialized as the identity permutation, see
    hipsparseCreateIdentityPermutation().

    ``hipsparseXcoosortByRow`` requires extra temporary storage buffer that has to be
    allocated by the user. Storage buffer size can be determined by
    hipsparseXcoosort_bufferSizeExt().

    Note:
        ``perm`` can be ``NULL`` if a sorted permutation vector is not required.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcoosortByRow__retval = hipsparseStatus_t(chipsparse.hipsparseXcoosortByRow(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(cooRows)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cooCols)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(P)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseXcoosortByRow__retval,)


@cython.embedsignature(True)
def hipsparseXcoosortByColumn(object handle, int m, int n, int nnz, object cooRows, object cooCols, object P, object pBuffer):
    r"""Sort a sparse COO matrix by column

    ``hipsparseXcoosortByColumn`` sorts a matrix in COO format by column. The sorted
    permutation vector ``perm`` can be used to obtain sorted ``coo_val`` array. In this
    case, ``perm`` must be initialized as the identity permutation, see
    hipsparseCreateIdentityPermutation().

    ``hipsparseXcoosortByColumn`` requires extra temporary storage buffer that has to be
    allocated by the user. Storage buffer size can be determined by
    hipsparseXcoosort_bufferSizeExt().

    Note:
        ``perm`` can be ``NULL`` if a sorted permutation vector is not required.

    Note:
        This function is non blocking and executed asynchronously with respect to the host.
        It may return before the actual computation has finished.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseXcoosortByColumn__retval = hipsparseStatus_t(chipsparse.hipsparseXcoosortByColumn(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <int *>hip._util.types.Pointer.from_pyobj(cooRows)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(cooCols)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(P)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseXcoosortByColumn__retval,)


@cython.embedsignature(True)
def hipsparseSgebsr2gebsr_bufferSize(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, object bufferSize):
    r"""This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
    general BSR matrix to another sparse general BSR matrix.

    ``hipsparseXgebsr2gebsr_bufferSize`` returns the size of the temporary storage buffer
    that is required by hipsparseXgebsr2gebsrNnz() and hipsparseXgebsr2gebsr().
    The temporary storage buffer must be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSgebsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSgebsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,
        <int *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSgebsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDgebsr2gebsr_bufferSize(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDgebsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDgebsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,
        <int *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDgebsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseCgebsr2gebsr_bufferSize(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCgebsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseCgebsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,
        <int *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseCgebsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseZgebsr2gebsr_bufferSize(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZgebsr2gebsr_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseZgebsr2gebsr_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,
        <int *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseZgebsr2gebsr_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseXgebsr2gebsrNnz(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, object descrC, object bsrRowPtrC, int rowBlockDimC, int colBlockDimC, object nnzTotalDevHostPtr, object buffer):
    r"""This function is used when converting a general BSR sparse matrix ``A`` to another general BSR sparse matrix ``C.``
    Specifically, this function determines the number of non-zero blocks that will exist in ``C`` (stored using either a host
    or device pointer), and computes the row pointer array for ``C.``

    The routine does support asynchronous execution.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseXgebsr2gebsrNnz__retval = hipsparseStatus_t(chipsparse.hipsparseXgebsr2gebsrNnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,rowBlockDimC,colBlockDimC,
        <int *>hip._util.types.Pointer.from_pyobj(nnzTotalDevHostPtr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseXgebsr2gebsrNnz__retval,)


@cython.embedsignature(True)
def hipsparseSgebsr2gebsr(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC, int rowBlockDimC, int colBlockDimC, object buffer):
    r"""This function converts the general BSR sparse matrix ``A`` to another general BSR sparse matrix ``C.``

    The conversion uses three steps. First, the user calls hipsparseXgebsr2gebsr_bufferSize() to determine the size of
    the required temporary storage buffer. The user then allocates this buffer. Secondly, the user then allocates ``mb_C+1``
    integers for the row pointer array for ``C`` where ``mb_C=(m+row_block_dim_C-1)/row_block_dim_C.`` The user then calls
    hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for ``C`` ( ``bsr_row_ptr_C`` ) and determine the number of
    non-zero blocks that will exist in ``C.`` Finally, the user allocates space for the colimn indices array of ``C`` to have
    ``nnzb_C`` elements and space for the values array of ``C`` to have ``nnzb_C*roc_block_dim_C*col_block_dim_C`` and then calls
    hipsparseXgebsr2gebsr() to complete the conversion.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseSgebsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseSgebsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr,rowBlockDimC,colBlockDimC,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseSgebsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseDgebsr2gebsr(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC, int rowBlockDimC, int colBlockDimC, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseDgebsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseDgebsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr,rowBlockDimC,colBlockDimC,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseDgebsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseCgebsr2gebsr(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC, int rowBlockDimC, int colBlockDimC, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseCgebsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseCgebsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        float2.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr,rowBlockDimC,colBlockDimC,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseCgebsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseZgebsr2gebsr(object handle, object dirA, int mb, int nb, int nnzb, object descrA, object bsrValA, object bsrRowPtrA, object bsrColIndA, int rowBlockDimA, int colBlockDimA, object descrC, object bsrValC, object bsrRowPtrC, object bsrColIndC, int rowBlockDimC, int colBlockDimC, object buffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(dirA,_hipsparseDirection_t__Base):
        raise TypeError("argument 'dirA' must be of type '_hipsparseDirection_t__Base'")
    _hipsparseZgebsr2gebsr__retval = hipsparseStatus_t(chipsparse.hipsparseZgebsr2gebsr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,dirA.value,mb,nb,nnzb,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(bsrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(bsrColIndA)._ptr,rowBlockDimA,colBlockDimA,
        <void *const>hip._util.types.Pointer.from_pyobj(descrC)._ptr,
        double2.from_pyobj(bsrValC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrRowPtrC)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(bsrColIndC)._ptr,rowBlockDimC,colBlockDimC,
        <void *>hip._util.types.Pointer.from_pyobj(buffer)._ptr))    # fully specified
    return (_hipsparseZgebsr2gebsr__retval,)


@cython.embedsignature(True)
def hipsparseScsru2csr_bufferSizeExt(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object info, object pBufferSizeInBytes):
    r"""This function calculates the amount of temporary storage required for
    hipsparseXcsru2csr() and hipsparseXcsr2csru().

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsru2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseScsru2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseScsru2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseDcsru2csr_bufferSizeExt(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsru2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseDcsru2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseDcsru2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseCcsru2csr_bufferSizeExt(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsru2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseCcsru2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        float2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseCcsru2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseZcsru2csr_bufferSizeExt(object handle, int m, int n, int nnz, object csrVal, object csrRowPtr, object csrColInd, object info, object pBufferSizeInBytes):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsru2csr_bufferSizeExt__retval = hipsparseStatus_t(chipsparse.hipsparseZcsru2csr_bufferSizeExt(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        double2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(pBufferSizeInBytes)._ptr))    # fully specified
    return (_hipsparseZcsru2csr_bufferSizeExt__retval,)


@cython.embedsignature(True)
def hipsparseScsru2csr(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""This function converts unsorted CSR format to sorted CSR format. The required
    temporary storage has to be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsru2csr__retval = hipsparseStatus_t(chipsparse.hipsparseScsru2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsru2csr__retval,)


@cython.embedsignature(True)
def hipsparseDcsru2csr(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsru2csr__retval = hipsparseStatus_t(chipsparse.hipsparseDcsru2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsru2csr__retval,)


@cython.embedsignature(True)
def hipsparseCcsru2csr(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsru2csr__retval = hipsparseStatus_t(chipsparse.hipsparseCcsru2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsru2csr__retval,)


@cython.embedsignature(True)
def hipsparseZcsru2csr(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsru2csr__retval = hipsparseStatus_t(chipsparse.hipsparseZcsru2csr(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsru2csr__retval,)


@cython.embedsignature(True)
def hipsparseScsr2csru(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""This function converts sorted CSR format to unsorted CSR format. The required
    temporary storage has to be allocated by the user.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsr2csru__retval = hipsparseStatus_t(chipsparse.hipsparseScsr2csru(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseScsr2csru__retval,)


@cython.embedsignature(True)
def hipsparseDcsr2csru(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsr2csru__retval = hipsparseStatus_t(chipsparse.hipsparseDcsr2csru(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseDcsr2csru__retval,)


@cython.embedsignature(True)
def hipsparseCcsr2csru(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsr2csru__retval = hipsparseStatus_t(chipsparse.hipsparseCcsr2csru(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseCcsr2csru__retval,)


@cython.embedsignature(True)
def hipsparseZcsr2csru(object handle, int m, int n, int nnz, object descrA, object csrVal, object csrRowPtr, object csrColInd, object info, object pBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsr2csru__retval = hipsparseStatus_t(chipsparse.hipsparseZcsr2csru(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,n,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrVal)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        csru2csrInfo.from_pyobj(info)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(pBuffer)._ptr))    # fully specified
    return (_hipsparseZcsr2csru__retval,)


@cython.embedsignature(True)
def hipsparseScsrcolor(object handle, int m, int nnz, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object fractionToColor, object ncolors, object coloring, object reordering, object info):
    r"""Coloring of the adjacency graph of the matrix :math:`A` stored in the CSR format.

    ``hipsparseXcsrcolor`` performs the coloring of the undirected graph represented by the (symmetric) sparsity pattern of the matrix :math:`A` stored in CSR format. Graph coloring is a way of coloring the nodes of a graph such that no two adjacent nodes are of the same color. The ``fraction_to_color`` is a parameter to only color a given percentage of the graph nodes, the remaining uncolored nodes receive distinct new colors. The optional ``reordering`` array is a permutation array such that unknowns of the same color are grouped. The matrix :math:`A` must be stored as a general matrix with a symmetric sparsity pattern, and if the matrix :math:`A` is non-symmetric then the user is responsible to provide the symmetric part :math:`\frac{A+A^T}{2}`.

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScsrcolor__retval = hipsparseStatus_t(chipsparse.hipsparseScsrcolor(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(fractionToColor)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(ncolors)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(coloring)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(reordering)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseScsrcolor__retval,)


@cython.embedsignature(True)
def hipsparseDcsrcolor(object handle, int m, int nnz, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object fractionToColor, object ncolors, object coloring, object reordering, object info):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDcsrcolor__retval = hipsparseStatus_t(chipsparse.hipsparseDcsrcolor(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(fractionToColor)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(ncolors)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(coloring)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(reordering)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseDcsrcolor__retval,)


@cython.embedsignature(True)
def hipsparseCcsrcolor(object handle, int m, int nnz, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object fractionToColor, object ncolors, object coloring, object reordering, object info):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCcsrcolor__retval = hipsparseStatus_t(chipsparse.hipsparseCcsrcolor(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        float2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const float *>hip._util.types.Pointer.from_pyobj(fractionToColor)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(ncolors)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(coloring)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(reordering)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseCcsrcolor__retval,)


@cython.embedsignature(True)
def hipsparseZcsrcolor(object handle, int m, int nnz, object descrA, object csrValA, object csrRowPtrA, object csrColIndA, object fractionToColor, object ncolors, object coloring, object reordering, object info):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseZcsrcolor__retval = hipsparseStatus_t(chipsparse.hipsparseZcsrcolor(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,m,nnz,
        <void *const>hip._util.types.Pointer.from_pyobj(descrA)._ptr,
        double2.from_pyobj(csrValA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrRowPtrA)._ptr,
        <const int *>hip._util.types.Pointer.from_pyobj(csrColIndA)._ptr,
        <const double *>hip._util.types.Pointer.from_pyobj(fractionToColor)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(ncolors)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(coloring)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(reordering)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(info)._ptr))    # fully specified
    return (_hipsparseZcsrcolor__retval,)


cdef class hipsparseSpGEMMDescr:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipsparseSpGEMMDescr from_ptr(chipsparse.hipsparseSpGEMMDescr* ptr, bint owner=False):
        """Factory function to create ``hipsparseSpGEMMDescr`` objects from
        given ``chipsparse.hipsparseSpGEMMDescr`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipsparseSpGEMMDescr wrapper = hipsparseSpGEMMDescr.__new__(hipsparseSpGEMMDescr)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipsparseSpGEMMDescr from_pyobj(object pyobj):
        """Derives a hipsparseSpGEMMDescr from a Python object.

        Derives a hipsparseSpGEMMDescr from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipsparseSpGEMMDescr`` reference, this method
        returns it directly. No new ``hipsparseSpGEMMDescr`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `hipsparseSpGEMMDescr`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipsparseSpGEMMDescr!
        """
        cdef hipsparseSpGEMMDescr wrapper = hipsparseSpGEMMDescr.__new__(hipsparseSpGEMMDescr)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipsparseSpGEMMDescr):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.hipsparseSpGEMMDescr*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.hipsparseSpGEMMDescr*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.hipsparseSpGEMMDescr*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.hipsparseSpGEMMDescr*>wrapper._py_buffer.buf
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
        return f"<hipsparseSpGEMMDescr object, self.ptr={int(self)}>"
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


hipsparseSpGEMMDescr_t = hipsparseSpGEMMDescr

cdef class hipsparseSpSVDescr:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipsparseSpSVDescr from_ptr(chipsparse.hipsparseSpSVDescr* ptr, bint owner=False):
        """Factory function to create ``hipsparseSpSVDescr`` objects from
        given ``chipsparse.hipsparseSpSVDescr`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipsparseSpSVDescr wrapper = hipsparseSpSVDescr.__new__(hipsparseSpSVDescr)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipsparseSpSVDescr from_pyobj(object pyobj):
        """Derives a hipsparseSpSVDescr from a Python object.

        Derives a hipsparseSpSVDescr from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipsparseSpSVDescr`` reference, this method
        returns it directly. No new ``hipsparseSpSVDescr`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `hipsparseSpSVDescr`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipsparseSpSVDescr!
        """
        cdef hipsparseSpSVDescr wrapper = hipsparseSpSVDescr.__new__(hipsparseSpSVDescr)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipsparseSpSVDescr):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.hipsparseSpSVDescr*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.hipsparseSpSVDescr*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.hipsparseSpSVDescr*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.hipsparseSpSVDescr*>wrapper._py_buffer.buf
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
        return f"<hipsparseSpSVDescr object, self.ptr={int(self)}>"
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


hipsparseSpSVDescr_t = hipsparseSpSVDescr

cdef class hipsparseSpSMDescr:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipsparseSpSMDescr from_ptr(chipsparse.hipsparseSpSMDescr* ptr, bint owner=False):
        """Factory function to create ``hipsparseSpSMDescr`` objects from
        given ``chipsparse.hipsparseSpSMDescr`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipsparseSpSMDescr wrapper = hipsparseSpSMDescr.__new__(hipsparseSpSMDescr)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipsparseSpSMDescr from_pyobj(object pyobj):
        """Derives a hipsparseSpSMDescr from a Python object.

        Derives a hipsparseSpSMDescr from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipsparseSpSMDescr`` reference, this method
        returns it directly. No new ``hipsparseSpSMDescr`` is created in this case.

        Args:
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `hipsparseSpSMDescr`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipsparseSpSMDescr!
        """
        cdef hipsparseSpSMDescr wrapper = hipsparseSpSMDescr.__new__(hipsparseSpSMDescr)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipsparseSpSMDescr):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipsparse.hipsparseSpSMDescr*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipsparse.hipsparseSpSMDescr*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipsparse.hipsparseSpSMDescr*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipsparse.hipsparseSpSMDescr*>wrapper._py_buffer.buf
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
        return f"<hipsparseSpSMDescr object, self.ptr={int(self)}>"
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


hipsparseSpSMDescr_t = hipsparseSpSMDescr

class _hipsparseFormat_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseFormat_t(_hipsparseFormat_t__Base):
    HIPSPARSE_FORMAT_CSR = chipsparse.HIPSPARSE_FORMAT_CSR
    HIPSPARSE_FORMAT_CSC = chipsparse.HIPSPARSE_FORMAT_CSC
    HIPSPARSE_FORMAT_COO = chipsparse.HIPSPARSE_FORMAT_COO
    HIPSPARSE_FORMAT_COO_AOS = chipsparse.HIPSPARSE_FORMAT_COO_AOS
    HIPSPARSE_FORMAT_BLOCKED_ELL = chipsparse.HIPSPARSE_FORMAT_BLOCKED_ELL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseOrder_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseOrder_t(_hipsparseOrder_t__Base):
    HIPSPARSE_ORDER_ROW = chipsparse.HIPSPARSE_ORDER_ROW
    HIPSPARSE_ORDER_COLUMN = chipsparse.HIPSPARSE_ORDER_COLUMN
    HIPSPARSE_ORDER_COL = chipsparse.HIPSPARSE_ORDER_COL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseIndexType_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseIndexType_t(_hipsparseIndexType_t__Base):
    HIPSPARSE_INDEX_16U = chipsparse.HIPSPARSE_INDEX_16U
    HIPSPARSE_INDEX_32I = chipsparse.HIPSPARSE_INDEX_32I
    HIPSPARSE_INDEX_64I = chipsparse.HIPSPARSE_INDEX_64I
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpMVAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpMVAlg_t(_hipsparseSpMVAlg_t__Base):
    HIPSPARSE_MV_ALG_DEFAULT = chipsparse.HIPSPARSE_MV_ALG_DEFAULT
    HIPSPARSE_COOMV_ALG = chipsparse.HIPSPARSE_COOMV_ALG
    HIPSPARSE_CSRMV_ALG1 = chipsparse.HIPSPARSE_CSRMV_ALG1
    HIPSPARSE_CSRMV_ALG2 = chipsparse.HIPSPARSE_CSRMV_ALG2
    HIPSPARSE_SPMV_ALG_DEFAULT = chipsparse.HIPSPARSE_SPMV_ALG_DEFAULT
    HIPSPARSE_SPMV_COO_ALG1 = chipsparse.HIPSPARSE_SPMV_COO_ALG1
    HIPSPARSE_SPMV_COO_ALG2 = chipsparse.HIPSPARSE_SPMV_COO_ALG2
    HIPSPARSE_SPMV_CSR_ALG1 = chipsparse.HIPSPARSE_SPMV_CSR_ALG1
    HIPSPARSE_SPMV_CSR_ALG2 = chipsparse.HIPSPARSE_SPMV_CSR_ALG2
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpMMAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpMMAlg_t(_hipsparseSpMMAlg_t__Base):
    HIPSPARSE_MM_ALG_DEFAULT = chipsparse.HIPSPARSE_MM_ALG_DEFAULT
    HIPSPARSE_COOMM_ALG1 = chipsparse.HIPSPARSE_COOMM_ALG1
    HIPSPARSE_COOMM_ALG2 = chipsparse.HIPSPARSE_COOMM_ALG2
    HIPSPARSE_COOMM_ALG3 = chipsparse.HIPSPARSE_COOMM_ALG3
    HIPSPARSE_CSRMM_ALG1 = chipsparse.HIPSPARSE_CSRMM_ALG1
    HIPSPARSE_SPMM_ALG_DEFAULT = chipsparse.HIPSPARSE_SPMM_ALG_DEFAULT
    HIPSPARSE_SPMM_COO_ALG1 = chipsparse.HIPSPARSE_SPMM_COO_ALG1
    HIPSPARSE_SPMM_COO_ALG2 = chipsparse.HIPSPARSE_SPMM_COO_ALG2
    HIPSPARSE_SPMM_COO_ALG3 = chipsparse.HIPSPARSE_SPMM_COO_ALG3
    HIPSPARSE_SPMM_COO_ALG4 = chipsparse.HIPSPARSE_SPMM_COO_ALG4
    HIPSPARSE_SPMM_CSR_ALG1 = chipsparse.HIPSPARSE_SPMM_CSR_ALG1
    HIPSPARSE_SPMM_CSR_ALG2 = chipsparse.HIPSPARSE_SPMM_CSR_ALG2
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = chipsparse.HIPSPARSE_SPMM_BLOCKED_ELL_ALG1
    HIPSPARSE_SPMM_CSR_ALG3 = chipsparse.HIPSPARSE_SPMM_CSR_ALG3
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSparseToDenseAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSparseToDenseAlg_t(_hipsparseSparseToDenseAlg_t__Base):
    HIPSPARSE_SPARSETODENSE_ALG_DEFAULT = chipsparse.HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseDenseToSparseAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseDenseToSparseAlg_t(_hipsparseDenseToSparseAlg_t__Base):
    HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT = chipsparse.HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSDDMMAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSDDMMAlg_t(_hipsparseSDDMMAlg_t__Base):
    HIPSPARSE_SDDMM_ALG_DEFAULT = chipsparse.HIPSPARSE_SDDMM_ALG_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpSVAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpSVAlg_t(_hipsparseSpSVAlg_t__Base):
    HIPSPARSE_SPSV_ALG_DEFAULT = chipsparse.HIPSPARSE_SPSV_ALG_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpSMAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpSMAlg_t(_hipsparseSpSMAlg_t__Base):
    HIPSPARSE_SPSM_ALG_DEFAULT = chipsparse.HIPSPARSE_SPSM_ALG_DEFAULT
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpMatAttribute_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpMatAttribute_t(_hipsparseSpMatAttribute_t__Base):
    HIPSPARSE_SPMAT_FILL_MODE = chipsparse.HIPSPARSE_SPMAT_FILL_MODE
    HIPSPARSE_SPMAT_DIAG_TYPE = chipsparse.HIPSPARSE_SPMAT_DIAG_TYPE
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class _hipsparseSpGEMMAlg_t__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hipsparseSpGEMMAlg_t(_hipsparseSpGEMMAlg_t__Base):
    HIPSPARSE_SPGEMM_DEFAULT = chipsparse.HIPSPARSE_SPGEMM_DEFAULT
    HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC = chipsparse.HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC
    HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC = chipsparse.HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def hipsparseCreateSpVec(object spVecDescr, long size, long nnz, object indices, object values, object idxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(idxType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'idxType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateSpVec__retval = hipsparseStatus_t(chipsparse.hipsparseCreateSpVec(
        <void **>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr,size,nnz,
        <void *>hip._util.types.Pointer.from_pyobj(indices)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr,idxType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateSpVec__retval,)


@cython.embedsignature(True)
def hipsparseDestroySpVec(object spVecDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroySpVec__retval = hipsparseStatus_t(chipsparse.hipsparseDestroySpVec(
        <void *>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr))    # fully specified
    return (_hipsparseDestroySpVec__retval,)


@cython.embedsignature(True)
def hipsparseSpVecGet(object spVecDescr, object size, object nnz, object indices, object values, object idxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpVecGet__retval = hipsparseStatus_t(chipsparse.hipsparseSpVecGet(
        <void *const>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(size)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(nnz)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(indices)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(idxType)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseSpVecGet__retval,)


@cython.embedsignature(True)
def hipsparseSpVecGetIndexBase(object spVecDescr, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpVecGetIndexBase__retval = hipsparseStatus_t(chipsparse.hipsparseSpVecGetIndexBase(
        <void *const>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr))    # fully specified
    return (_hipsparseSpVecGetIndexBase__retval,)


@cython.embedsignature(True)
def hipsparseSpVecGetValues(object spVecDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpVecGetValues__retval = hipsparseStatus_t(chipsparse.hipsparseSpVecGetValues(
        <void *const>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseSpVecGetValues__retval,)


@cython.embedsignature(True)
def hipsparseSpVecSetValues(object spVecDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpVecSetValues__retval = hipsparseStatus_t(chipsparse.hipsparseSpVecSetValues(
        <void *>hip._util.types.Pointer.from_pyobj(spVecDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseSpVecSetValues__retval,)


@cython.embedsignature(True)
def hipsparseCreateCoo(object spMatDescr, long rows, long cols, long nnz, object cooRowInd, object cooColInd, object cooValues, object cooIdxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(cooIdxType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'cooIdxType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateCoo__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCoo(
        <void **>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,rows,cols,nnz,
        <void *>hip._util.types.Pointer.from_pyobj(cooRowInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooValues)._ptr,cooIdxType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateCoo__retval,)


@cython.embedsignature(True)
def hipsparseCreateCooAoS(object spMatDescr, long rows, long cols, long nnz, object cooInd, object cooValues, object cooIdxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(cooIdxType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'cooIdxType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateCooAoS__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCooAoS(
        <void **>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,rows,cols,nnz,
        <void *>hip._util.types.Pointer.from_pyobj(cooInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooValues)._ptr,cooIdxType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateCooAoS__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsr(object spMatDescr, long rows, long cols, long nnz, object csrRowOffsets, object csrColInd, object csrValues, object csrRowOffsetsType, object csrColIndType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(csrRowOffsetsType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'csrRowOffsetsType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(csrColIndType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'csrColIndType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateCsr__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsr(
        <void **>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,rows,cols,nnz,
        <void *>hip._util.types.Pointer.from_pyobj(csrRowOffsets)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(csrValues)._ptr,csrRowOffsetsType.value,csrColIndType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateCsr__retval,)


@cython.embedsignature(True)
def hipsparseCreateCsc(object spMatDescr, long rows, long cols, long nnz, object cscColOffsets, object cscRowInd, object cscValues, object cscColOffsetsType, object cscRowIndType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(cscColOffsetsType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'cscColOffsetsType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(cscRowIndType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'cscRowIndType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateCsc__retval = hipsparseStatus_t(chipsparse.hipsparseCreateCsc(
        <void **>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,rows,cols,nnz,
        <void *>hip._util.types.Pointer.from_pyobj(cscColOffsets)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscValues)._ptr,cscColOffsetsType.value,cscRowIndType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateCsc__retval,)


@cython.embedsignature(True)
def hipsparseCreateBlockedEll(object spMatDescr, long rows, long cols, long ellBlockSize, long ellCols, object ellColInd, object ellValue, object ellIdxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(ellIdxType,_hipsparseIndexType_t__Base):
        raise TypeError("argument 'ellIdxType' must be of type '_hipsparseIndexType_t__Base'")                    
    if not isinstance(idxBase,_hipsparseIndexBase_t__Base):
        raise TypeError("argument 'idxBase' must be of type '_hipsparseIndexBase_t__Base'")                    
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateBlockedEll__retval = hipsparseStatus_t(chipsparse.hipsparseCreateBlockedEll(
        <void **>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,rows,cols,ellBlockSize,ellCols,
        <void *>hip._util.types.Pointer.from_pyobj(ellColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(ellValue)._ptr,ellIdxType.value,idxBase.value,valueType.value))    # fully specified
    return (_hipsparseCreateBlockedEll__retval,)


@cython.embedsignature(True)
def hipsparseDestroySpMat(object spMatDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroySpMat__retval = hipsparseStatus_t(chipsparse.hipsparseDestroySpMat(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr))    # fully specified
    return (_hipsparseDestroySpMat__retval,)


@cython.embedsignature(True)
def hipsparseCooGet(object spMatDescr, object rows, object cols, object nnz, object cooRowInd, object cooColInd, object cooValues, object idxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCooGet__retval = hipsparseStatus_t(chipsparse.hipsparseCooGet(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(nnz)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(cooRowInd)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(cooColInd)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(cooValues)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(idxType)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseCooGet__retval,)


@cython.embedsignature(True)
def hipsparseCooAoSGet(object spMatDescr, object rows, object cols, object nnz, object cooInd, object cooValues, object idxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCooAoSGet__retval = hipsparseStatus_t(chipsparse.hipsparseCooAoSGet(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(nnz)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(cooInd)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(cooValues)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(idxType)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseCooAoSGet__retval,)


@cython.embedsignature(True)
def hipsparseCsrGet(object spMatDescr, object rows, object cols, object nnz, object csrRowOffsets, object csrColInd, object csrValues, object csrRowOffsetsType, object csrColIndType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCsrGet__retval = hipsparseStatus_t(chipsparse.hipsparseCsrGet(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(nnz)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(csrRowOffsets)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(csrValues)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(csrRowOffsetsType)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(csrColIndType)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseCsrGet__retval,)


@cython.embedsignature(True)
def hipsparseBlockedEllGet(object spMatDescr, object rows, object cols, object ellBlockSize, object ellCols, object ellColInd, object ellValue, object ellIdxType, object idxBase, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseBlockedEllGet__retval = hipsparseStatus_t(chipsparse.hipsparseBlockedEllGet(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(ellBlockSize)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(ellCols)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(ellColInd)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(ellValue)._ptr,
        <chipsparse.hipsparseIndexType_t *>hip._util.types.Pointer.from_pyobj(ellIdxType)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseBlockedEllGet__retval,)


@cython.embedsignature(True)
def hipsparseCsrSetPointers(object spMatDescr, object csrRowOffsets, object csrColInd, object csrValues):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCsrSetPointers__retval = hipsparseStatus_t(chipsparse.hipsparseCsrSetPointers(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(csrRowOffsets)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(csrColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(csrValues)._ptr))    # fully specified
    return (_hipsparseCsrSetPointers__retval,)


@cython.embedsignature(True)
def hipsparseCscSetPointers(object spMatDescr, object cscColOffsets, object cscRowInd, object cscValues):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCscSetPointers__retval = hipsparseStatus_t(chipsparse.hipsparseCscSetPointers(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscColOffsets)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscRowInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cscValues)._ptr))    # fully specified
    return (_hipsparseCscSetPointers__retval,)


@cython.embedsignature(True)
def hipsparseCooSetPointers(object spMatDescr, object cooRowInd, object cooColInd, object cooValues):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCooSetPointers__retval = hipsparseStatus_t(chipsparse.hipsparseCooSetPointers(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooRowInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooColInd)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(cooValues)._ptr))    # fully specified
    return (_hipsparseCooSetPointers__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetSize(object spMatDescr, object rows, object cols, object nnz):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatGetSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetSize(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(nnz)._ptr))    # fully specified
    return (_hipsparseSpMatGetSize__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetFormat(object spMatDescr, object format):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatGetFormat__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetFormat(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <chipsparse.hipsparseFormat_t *>hip._util.types.Pointer.from_pyobj(format)._ptr))    # fully specified
    return (_hipsparseSpMatGetFormat__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetIndexBase(object spMatDescr, object idxBase):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatGetIndexBase__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetIndexBase(
        <void *const>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <chipsparse.hipsparseIndexBase_t *>hip._util.types.Pointer.from_pyobj(idxBase)._ptr))    # fully specified
    return (_hipsparseSpMatGetIndexBase__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetValues(object spMatDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatGetValues__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetValues(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseSpMatGetValues__retval,)


@cython.embedsignature(True)
def hipsparseSpMatSetValues(object spMatDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatSetValues__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatSetValues(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseSpMatSetValues__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetStridedBatch(object spMatDescr, object batchCount):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatGetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(batchCount)._ptr))    # fully specified
    return (_hipsparseSpMatGetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseSpMatSetStridedBatch(object spMatDescr, int batchCount):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpMatSetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatSetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,batchCount))    # fully specified
    return (_hipsparseSpMatSetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseCooSetStridedBatch(object spMatDescr, int batchCount, long batchStride):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCooSetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseCooSetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,batchCount,batchStride))    # fully specified
    return (_hipsparseCooSetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseCsrSetStridedBatch(object spMatDescr, int batchCount, long offsetsBatchStride, long columnsValuesBatchStride):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseCsrSetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseCsrSetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,batchCount,offsetsBatchStride,columnsValuesBatchStride))    # fully specified
    return (_hipsparseCsrSetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseSpMatGetAttribute(object spMatDescr, object attribute, object data, unsigned long dataSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(attribute,_hipsparseSpMatAttribute_t__Base):
        raise TypeError("argument 'attribute' must be of type '_hipsparseSpMatAttribute_t__Base'")
    _hipsparseSpMatGetAttribute__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatGetAttribute(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,attribute.value,
        <void *>hip._util.types.Pointer.from_pyobj(data)._ptr,dataSize))    # fully specified
    return (_hipsparseSpMatGetAttribute__retval,)


@cython.embedsignature(True)
def hipsparseSpMatSetAttribute(object spMatDescr, object attribute, object data, unsigned long dataSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(attribute,_hipsparseSpMatAttribute_t__Base):
        raise TypeError("argument 'attribute' must be of type '_hipsparseSpMatAttribute_t__Base'")
    _hipsparseSpMatSetAttribute__retval = hipsparseStatus_t(chipsparse.hipsparseSpMatSetAttribute(
        <void *>hip._util.types.Pointer.from_pyobj(spMatDescr)._ptr,attribute.value,
        <const void *>hip._util.types.Pointer.from_pyobj(data)._ptr,dataSize))    # fully specified
    return (_hipsparseSpMatSetAttribute__retval,)


@cython.embedsignature(True)
def hipsparseCreateDnVec(object dnVecDescr, long size, object values, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")
    _hipsparseCreateDnVec__retval = hipsparseStatus_t(chipsparse.hipsparseCreateDnVec(
        <void **>hip._util.types.Pointer.from_pyobj(dnVecDescr)._ptr,size,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr,valueType.value))    # fully specified
    return (_hipsparseCreateDnVec__retval,)


@cython.embedsignature(True)
def hipsparseDestroyDnVec(object dnVecDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyDnVec__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyDnVec(
        <void *>hip._util.types.Pointer.from_pyobj(dnVecDescr)._ptr))    # fully specified
    return (_hipsparseDestroyDnVec__retval,)


@cython.embedsignature(True)
def hipsparseDnVecGet(object dnVecDescr, object size, object values, object valueType):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnVecGet__retval = hipsparseStatus_t(chipsparse.hipsparseDnVecGet(
        <void *const>hip._util.types.Pointer.from_pyobj(dnVecDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(size)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr))    # fully specified
    return (_hipsparseDnVecGet__retval,)


@cython.embedsignature(True)
def hipsparseDnVecGetValues(object dnVecDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnVecGetValues__retval = hipsparseStatus_t(chipsparse.hipsparseDnVecGetValues(
        <void *const>hip._util.types.Pointer.from_pyobj(dnVecDescr)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseDnVecGetValues__retval,)


@cython.embedsignature(True)
def hipsparseDnVecSetValues(object dnVecDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnVecSetValues__retval = hipsparseStatus_t(chipsparse.hipsparseDnVecSetValues(
        <void *>hip._util.types.Pointer.from_pyobj(dnVecDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseDnVecSetValues__retval,)


@cython.embedsignature(True)
def hipsparseCreateDnMat(object dnMatDescr, long rows, long cols, long ld, object values, object valueType, object order):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(valueType,_hipDataType__Base):
        raise TypeError("argument 'valueType' must be of type '_hipDataType__Base'")                    
    if not isinstance(order,_hipsparseOrder_t__Base):
        raise TypeError("argument 'order' must be of type '_hipsparseOrder_t__Base'")
    _hipsparseCreateDnMat__retval = hipsparseStatus_t(chipsparse.hipsparseCreateDnMat(
        <void **>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,rows,cols,ld,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr,valueType.value,order.value))    # fully specified
    return (_hipsparseCreateDnMat__retval,)


@cython.embedsignature(True)
def hipsparseDestroyDnMat(object dnMatDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDestroyDnMat__retval = hipsparseStatus_t(chipsparse.hipsparseDestroyDnMat(
        <void *>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr))    # fully specified
    return (_hipsparseDestroyDnMat__retval,)


@cython.embedsignature(True)
def hipsparseDnMatGet(object dnMatDescr, object rows, object cols, object ld, object values, object valueType, object order):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnMatGet__retval = hipsparseStatus_t(chipsparse.hipsparseDnMatGet(
        <void *const>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(rows)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(cols)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(ld)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr,
        <chipsparse.hipDataType *>hip._util.types.Pointer.from_pyobj(valueType)._ptr,
        <chipsparse.hipsparseOrder_t *>hip._util.types.Pointer.from_pyobj(order)._ptr))    # fully specified
    return (_hipsparseDnMatGet__retval,)


@cython.embedsignature(True)
def hipsparseDnMatGetValues(object dnMatDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnMatGetValues__retval = hipsparseStatus_t(chipsparse.hipsparseDnMatGetValues(
        <void *const>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,
        <void **>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseDnMatGetValues__retval,)


@cython.embedsignature(True)
def hipsparseDnMatSetValues(object dnMatDescr, object values):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnMatSetValues__retval = hipsparseStatus_t(chipsparse.hipsparseDnMatSetValues(
        <void *>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(values)._ptr))    # fully specified
    return (_hipsparseDnMatSetValues__retval,)


@cython.embedsignature(True)
def hipsparseDnMatGetStridedBatch(object dnMatDescr, object batchCount, object batchStride):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnMatGetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseDnMatGetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(batchCount)._ptr,
        <long *>hip._util.types.Pointer.from_pyobj(batchStride)._ptr))    # fully specified
    return (_hipsparseDnMatGetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseDnMatSetStridedBatch(object dnMatDescr, int batchCount, long batchStride):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseDnMatSetStridedBatch__retval = hipsparseStatus_t(chipsparse.hipsparseDnMatSetStridedBatch(
        <void *>hip._util.types.Pointer.from_pyobj(dnMatDescr)._ptr,batchCount,batchStride))    # fully specified
    return (_hipsparseDnMatSetStridedBatch__retval,)


@cython.embedsignature(True)
def hipsparseAxpby(object handle, object alpha, object vecX, object beta, object vecY):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseAxpby__retval = hipsparseStatus_t(chipsparse.hipsparseAxpby(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr))    # fully specified
    return (_hipsparseAxpby__retval,)


@cython.embedsignature(True)
def hipsparseGather(object handle, object vecY, object vecX):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseGather__retval = hipsparseStatus_t(chipsparse.hipsparseGather(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr))    # fully specified
    return (_hipsparseGather__retval,)


@cython.embedsignature(True)
def hipsparseScatter(object handle, object vecX, object vecY):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseScatter__retval = hipsparseStatus_t(chipsparse.hipsparseScatter(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr))    # fully specified
    return (_hipsparseScatter__retval,)


@cython.embedsignature(True)
def hipsparseRot(object handle, object c_coeff, object s_coeff, object vecX, object vecY):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseRot__retval = hipsparseStatus_t(chipsparse.hipsparseRot(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(c_coeff)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(s_coeff)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr))    # fully specified
    return (_hipsparseRot__retval,)


@cython.embedsignature(True)
def hipsparseSparseToDense_bufferSize(object handle, object matA, object matB, object alg, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(alg,_hipsparseSparseToDenseAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSparseToDenseAlg_t__Base'")
    _hipsparseSparseToDense_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSparseToDense_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSparseToDense_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSparseToDense(object handle, object matA, object matB, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(alg,_hipsparseSparseToDenseAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSparseToDenseAlg_t__Base'")
    _hipsparseSparseToDense__retval = hipsparseStatus_t(chipsparse.hipsparseSparseToDense(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSparseToDense__retval,)


@cython.embedsignature(True)
def hipsparseDenseToSparse_bufferSize(object handle, object matA, object matB, object alg, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(alg,_hipsparseDenseToSparseAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseDenseToSparseAlg_t__Base'")
    _hipsparseDenseToSparse_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseDenseToSparse_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseDenseToSparse_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseDenseToSparse_analysis(object handle, object matA, object matB, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(alg,_hipsparseDenseToSparseAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseDenseToSparseAlg_t__Base'")
    _hipsparseDenseToSparse_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseDenseToSparse_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseDenseToSparse_analysis__retval,)


@cython.embedsignature(True)
def hipsparseDenseToSparse_convert(object handle, object matA, object matB, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(alg,_hipsparseDenseToSparseAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseDenseToSparseAlg_t__Base'")
    _hipsparseDenseToSparse_convert__retval = hipsparseStatus_t(chipsparse.hipsparseDenseToSparse_convert(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseDenseToSparse_convert__retval,)


@cython.embedsignature(True)
def hipsparseSpVV_bufferSize(object handle, object opX, object vecX, object vecY, object result, object computeType, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")
    _hipsparseSpVV_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpVV_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opX.value,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(result)._ptr,computeType.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpVV_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpVV(object handle, object opX, object vecX, object vecY, object result, object computeType, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opX,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opX' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")
    _hipsparseSpVV__retval = hipsparseStatus_t(chipsparse.hipsparseSpVV(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opX.value,
        <void *>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(vecY)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(result)._ptr,computeType.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpVV__retval,)


@cython.embedsignature(True)
def hipsparseSpMV_bufferSize(object handle, object opA, object alpha, object matA, object vecX, object beta, object vecY, object computeType, object alg, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMVAlg_t__Base'")
    _hipsparseSpMV_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpMV_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecY)._ptr,computeType.value,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpMV_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpMV_preprocess(object handle, object opA, object alpha, object matA, object vecX, object beta, object vecY, object computeType, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMVAlg_t__Base'")
    _hipsparseSpMV_preprocess__retval = hipsparseStatus_t(chipsparse.hipsparseSpMV_preprocess(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecY)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpMV_preprocess__retval,)


@cython.embedsignature(True)
def hipsparseSpMV(object handle, object opA, object alpha, object matA, object vecX, object beta, object vecY, object computeType, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMVAlg_t__Base'")
    _hipsparseSpMV__retval = hipsparseStatus_t(chipsparse.hipsparseSpMV(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecX)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(vecY)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpMV__retval,)


@cython.embedsignature(True)
def hipsparseSpMM_bufferSize(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMMAlg_t__Base'")
    _hipsparseSpMM_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpMM_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpMM_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpMM_preprocess(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMMAlg_t__Base'")
    _hipsparseSpMM_preprocess__retval = hipsparseStatus_t(chipsparse.hipsparseSpMM_preprocess(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpMM_preprocess__retval,)


@cython.embedsignature(True)
def hipsparseSpMM(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpMMAlg_t__Base'")
    _hipsparseSpMM__retval = hipsparseStatus_t(chipsparse.hipsparseSpMM(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpMM__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMM_createDescr():
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    descr = hipsparseSpGEMMDescr.from_ptr(NULL)
    _hipsparseSpGEMM_createDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMM_createDescr(&descr._ptr))    # fully specified
    return (_hipsparseSpGEMM_createDescr__retval,descr)


@cython.embedsignature(True)
def hipsparseSpGEMM_destroyDescr(object descr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpGEMM_destroyDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMM_destroyDescr(
        hipsparseSpGEMMDescr.from_pyobj(descr)._ptr))    # fully specified
    return (_hipsparseSpGEMM_destroyDescr__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMM_workEstimation(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object spgemmDescr, object bufferSize1, object externalBuffer1):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMM_workEstimation__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMM_workEstimation(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize1)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer1)._ptr))    # fully specified
    return (_hipsparseSpGEMM_workEstimation__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMM_compute(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object spgemmDescr, object bufferSize2, object externalBuffer2):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMM_compute__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMM_compute(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize2)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer2)._ptr))    # fully specified
    return (_hipsparseSpGEMM_compute__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMM_copy(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object spgemmDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMM_copy__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMM_copy(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr))    # fully specified
    return (_hipsparseSpGEMM_copy__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMMreuse_workEstimation(object handle, object opA, object opB, object matA, object matB, object matC, object alg, object spgemmDescr, object bufferSize1, object externalBuffer1):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMMreuse_workEstimation__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMMreuse_workEstimation(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize1)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer1)._ptr))    # fully specified
    return (_hipsparseSpGEMMreuse_workEstimation__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMMreuse_nnz(object handle, object opA, object opB, object matA, object matB, object matC, object alg, object spgemmDescr, object bufferSize2, object externalBuffer2, object bufferSize3, object externalBuffer3, object bufferSize4, object externalBuffer4):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMMreuse_nnz__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMMreuse_nnz(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize2)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer2)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize3)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer3)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize4)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer4)._ptr))    # fully specified
    return (_hipsparseSpGEMMreuse_nnz__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMMreuse_compute(object handle, object opA, object opB, object alpha, object matA, object matB, object beta, object matC, object computeType, object alg, object spgemmDescr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMMreuse_compute__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMMreuse_compute(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr))    # fully specified
    return (_hipsparseSpGEMMreuse_compute__retval,)


@cython.embedsignature(True)
def hipsparseSpGEMMreuse_copy(object handle, object opA, object opB, object matA, object matB, object matC, object alg, object spgemmDescr, object bufferSize5, object externalBuffer5):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(alg,_hipsparseSpGEMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpGEMMAlg_t__Base'")
    _hipsparseSpGEMMreuse_copy__retval = hipsparseStatus_t(chipsparse.hipsparseSpGEMMreuse_copy(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <void *>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(matC)._ptr,alg.value,
        hipsparseSpGEMMDescr.from_pyobj(spgemmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize5)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer5)._ptr))    # fully specified
    return (_hipsparseSpGEMMreuse_copy__retval,)


@cython.embedsignature(True)
def hipsparseSDDMM(object handle, object opA, object opB, object alpha, object A, object B, object beta, object C, object computeType, object alg, object tempBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSDDMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSDDMMAlg_t__Base'")
    _hipsparseSDDMM__retval = hipsparseStatus_t(chipsparse.hipsparseSDDMM(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(A)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(B)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(C)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(tempBuffer)._ptr))    # fully specified
    return (_hipsparseSDDMM__retval,)


@cython.embedsignature(True)
def hipsparseSDDMM_bufferSize(object handle, object opA, object opB, object alpha, object A, object B, object beta, object C, object computeType, object alg, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSDDMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSDDMMAlg_t__Base'")
    _hipsparseSDDMM_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSDDMM_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(A)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(B)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(C)._ptr,computeType.value,alg.value,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSDDMM_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSDDMM_preprocess(object handle, object opA, object opB, object alpha, object A, object B, object beta, object C, object computeType, object alg, object tempBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSDDMMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSDDMMAlg_t__Base'")
    _hipsparseSDDMM_preprocess__retval = hipsparseStatus_t(chipsparse.hipsparseSDDMM_preprocess(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(A)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(B)._ptr,
        <const void *>hip._util.types.Pointer.from_pyobj(beta)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(C)._ptr,computeType.value,alg.value,
        <void *>hip._util.types.Pointer.from_pyobj(tempBuffer)._ptr))    # fully specified
    return (_hipsparseSDDMM_preprocess__retval,)


@cython.embedsignature(True)
def hipsparseSpSV_createDescr():
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    descr = hipsparseSpSVDescr.from_ptr(NULL)
    _hipsparseSpSV_createDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpSV_createDescr(&descr._ptr))    # fully specified
    return (_hipsparseSpSV_createDescr__retval,descr)


@cython.embedsignature(True)
def hipsparseSpSV_destroyDescr(object descr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpSV_destroyDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpSV_destroyDescr(
        hipsparseSpSVDescr.from_pyobj(descr)._ptr))    # fully specified
    return (_hipsparseSpSV_destroyDescr__retval,)


@cython.embedsignature(True)
def hipsparseSpSV_bufferSize(object handle, object opA, object alpha, object matA, object x, object y, object computeType, object alg, object spsvDescr, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSVAlg_t__Base'")
    _hipsparseSpSV_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpSV_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(y)._ptr,computeType.value,alg.value,
        hipsparseSpSVDescr.from_pyobj(spsvDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpSV_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpSV_analysis(object handle, object opA, object alpha, object matA, object x, object y, object computeType, object alg, object spsvDescr, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSVAlg_t__Base'")
    _hipsparseSpSV_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSpSV_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(y)._ptr,computeType.value,alg.value,
        hipsparseSpSVDescr.from_pyobj(spsvDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpSV_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSpSV_solve(object handle, object opA, object alpha, object matA, object x, object y, object computeType, object alg, object spsvDescr, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSVAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSVAlg_t__Base'")
    _hipsparseSpSV_solve__retval = hipsparseStatus_t(chipsparse.hipsparseSpSV_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(x)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(y)._ptr,computeType.value,alg.value,
        hipsparseSpSVDescr.from_pyobj(spsvDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpSV_solve__retval,)


@cython.embedsignature(True)
def hipsparseSpSM_createDescr():
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    descr = hipsparseSpSMDescr.from_ptr(NULL)
    _hipsparseSpSM_createDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpSM_createDescr(&descr._ptr))    # fully specified
    return (_hipsparseSpSM_createDescr__retval,descr)


@cython.embedsignature(True)
def hipsparseSpSM_destroyDescr(object descr):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    _hipsparseSpSM_destroyDescr__retval = hipsparseStatus_t(chipsparse.hipsparseSpSM_destroyDescr(
        hipsparseSpSMDescr.from_pyobj(descr)._ptr))    # fully specified
    return (_hipsparseSpSM_destroyDescr__retval,)


@cython.embedsignature(True)
def hipsparseSpSM_bufferSize(object handle, object opA, object opB, object alpha, object matA, object matB, object matC, object computeType, object alg, object spsmDescr, object bufferSize):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSMAlg_t__Base'")
    _hipsparseSpSM_bufferSize__retval = hipsparseStatus_t(chipsparse.hipsparseSpSM_bufferSize(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpSMDescr.from_pyobj(spsmDescr)._ptr,
        <unsigned long *>hip._util.types.Pointer.from_pyobj(bufferSize)._ptr))    # fully specified
    return (_hipsparseSpSM_bufferSize__retval,)


@cython.embedsignature(True)
def hipsparseSpSM_analysis(object handle, object opA, object opB, object alpha, object matA, object matB, object matC, object computeType, object alg, object spsmDescr, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSMAlg_t__Base'")
    _hipsparseSpSM_analysis__retval = hipsparseStatus_t(chipsparse.hipsparseSpSM_analysis(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpSMDescr.from_pyobj(spsmDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpSM_analysis__retval,)


@cython.embedsignature(True)
def hipsparseSpSM_solve(object handle, object opA, object opB, object alpha, object matA, object matB, object matC, object computeType, object alg, object spsmDescr, object externalBuffer):
    r"""(No short description)

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hipsparseStatus_t`
    """
    if not isinstance(opA,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opA' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(opB,_hipsparseOperation_t__Base):
        raise TypeError("argument 'opB' must be of type '_hipsparseOperation_t__Base'")                    
    if not isinstance(computeType,_hipDataType__Base):
        raise TypeError("argument 'computeType' must be of type '_hipDataType__Base'")                    
    if not isinstance(alg,_hipsparseSpSMAlg_t__Base):
        raise TypeError("argument 'alg' must be of type '_hipsparseSpSMAlg_t__Base'")
    _hipsparseSpSM_solve__retval = hipsparseStatus_t(chipsparse.hipsparseSpSM_solve(
        <void *>hip._util.types.Pointer.from_pyobj(handle)._ptr,opA.value,opB.value,
        <const void *>hip._util.types.Pointer.from_pyobj(alpha)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matA)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matB)._ptr,
        <void *const>hip._util.types.Pointer.from_pyobj(matC)._ptr,computeType.value,alg.value,
        hipsparseSpSMDescr.from_pyobj(spsmDescr)._ptr,
        <void *>hip._util.types.Pointer.from_pyobj(externalBuffer)._ptr))    # fully specified
    return (_hipsparseSpSM_solve__retval,)