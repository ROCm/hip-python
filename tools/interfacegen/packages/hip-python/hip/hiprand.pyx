# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Attributes:
    HIPRAND_VERSION (`~.int`):
        Macro constant.

    HIPRAND_DEFAULT_MAX_BLOCK_SIZE (`~.int`):
        Macro constant.

    HIPRAND_DEFAULT_MIN_WARPS_PER_EU (`~.int`):
        Macro constant.

    rocrand_discrete_distribution:
        alias of `~.rocrand_discrete_distribution_st`

    rocrand_generator:
        alias of `~.rocrand_generator_base_type`

    hiprandGenerator_st:
        alias of `~.rocrand_generator_base_type`

    hiprandDiscreteDistribution_st:
        alias of `~.rocrand_discrete_distribution_st`

    hiprandGenerator_t:
        alias of `~.rocrand_generator_base_type`

    hiprandDiscreteDistribution_t:
        alias of `~.rocrand_discrete_distribution_st`

    hiprandStatus_t:
        alias of `~.hiprandStatus`

    hiprandRngType_t:
        alias of `~.hiprandRngType`

"""

import cython
import ctypes
import enum
HIPRAND_VERSION = chiprand.HIPRAND_VERSION

HIPRAND_DEFAULT_MAX_BLOCK_SIZE = chiprand.HIPRAND_DEFAULT_MAX_BLOCK_SIZE

HIPRAND_DEFAULT_MIN_WARPS_PER_EU = chiprand.HIPRAND_DEFAULT_MIN_WARPS_PER_EU

cdef class uint4:
    """Python wrapper type.
    
    Python wrapper for C type chiprand.uint4.

    If this type is initialized via its `__init__` method, it allocates a member of the underlying C type and
    destroys it again if the wrapper type is deallocted.

    This type also serves as adapter when appearing as argument type in a function signature.
    In this case, the type can further be initialized from the following Python objects
    that you can pass as argument instead:
    
    * `None`:
        This will set the ``self._ptr`` attribute to ``NULL`.
    * `~.Pointer` and its subclasses:
        Copies ``pyobj._ptr`` to ``self._ptr``.
    `~.Py_buffer` object ownership is not transferred!
    * `int`:
        Interprets the integer value as pointer address and writes it to ``self._ptr``.
    * `ctypes.c_void_p`:
        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
    * `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol:
        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple 
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.
    * `object` that implements the Python buffer protocol:
        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.
    
    Type checks are performed in the above order.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _ptr_owner (C type ``bint``, protected):
            If this wrapper is the owner of the underlying data.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    """
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
        the extension type to free the structure pointed to by ``ptr``
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
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `uint4`, `int`, or `ctypes.c_void_p`

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
                pyobj,
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

    @staticmethod
    cdef uint4 from_value(chiprand.uint4 other):
        """Allocate new C type and copy from ``other``.
        """
        wrapper = uint4.new()
        string.memcpy(wrapper._ptr, &other, sizeof(chiprand.uint4))
        return wrapper
   
    def __init__(self,*args,**kwargs):
        """
        """

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
    
    def __int__(self):
        """Returns the data's address as long integer.
        """
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<uint4 object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`
        """
        return ctypes.c_void_p(int(self))
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
    """Python wrapper type.
    
    Python wrapper for C type chiprand.rocrand_discrete_distribution_st.

    If this type is initialized via its `__init__` method, it allocates a member of the underlying C type and
    destroys it again if the wrapper type is deallocted.

    This type also serves as adapter when appearing as argument type in a function signature.
    In this case, the type can further be initialized from the following Python objects
    that you can pass as argument instead:
    
    * `None`:
        This will set the ``self._ptr`` attribute to ``NULL`.
    * `~.Pointer` and its subclasses:
        Copies ``pyobj._ptr`` to ``self._ptr``.
    `~.Py_buffer` object ownership is not transferred!
    * `int`:
        Interprets the integer value as pointer address and writes it to ``self._ptr``.
    * `ctypes.c_void_p`:
        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
    * `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol:
        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple 
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.
    * `object` that implements the Python buffer protocol:
        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.
    
    Type checks are performed in the above order.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _ptr_owner (C type ``bint``, protected):
            If this wrapper is the owner of the underlying data.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    """
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
        the extension type to free the structure pointed to by ``ptr``
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
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `rocrand_discrete_distribution_st`, `int`, or `ctypes.c_void_p`

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
                pyobj,
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

    @staticmethod
    cdef rocrand_discrete_distribution_st from_value(chiprand.rocrand_discrete_distribution_st other):
        """Allocate new C type and copy from ``other``.
        """
        wrapper = rocrand_discrete_distribution_st.new()
        string.memcpy(wrapper._ptr, &other, sizeof(chiprand.rocrand_discrete_distribution_st))
        return wrapper
   
    def __init__(self,*args,**kwargs):
        """
        """

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
    
    def __int__(self):
        """Returns the data's address as long integer.
        """
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<rocrand_discrete_distribution_st object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`
        """
        return ctypes.c_void_p(int(self))
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
        self._ptr[i].alias = <unsigned int *>cpython.long.PyLong_AsVoidPtr(int(hip._util.types.ListOfUnsigned.from_pyobj(value)))
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
        return hip._util.types.Pointer.from_ptr(self._ptr[i].probability)
    def set_probability(self, i, object value):
        """Set value ``probability`` of ``self._ptr[i]``.

        Note:
            This can be dangerous if the pointer is from a python object
            that is later on garbage collected.
        """
        self._ptr[i].probability = <double *>cpython.long.PyLong_AsVoidPtr(int(hip._util.types.Pointer.from_pyobj(value)))
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
        return hip._util.types.Pointer.from_ptr(self._ptr[i].cdf)
    def set_cdf(self, i, object value):
        """Set value ``cdf`` of ``self._ptr[i]``.

        Note:
            This can be dangerous if the pointer is from a python object
            that is later on garbage collected.
        """
        self._ptr[i].cdf = <double *>cpython.long.PyLong_AsVoidPtr(int(hip._util.types.Pointer.from_pyobj(value)))
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
    """Python wrapper type.
    
    Python wrapper for C type chiprand.rocrand_generator_base_type.

    If this type is initialized via its `__init__` method, it allocates a member of the underlying C type and
    destroys it again if the wrapper type is deallocted.

    This type also serves as adapter when appearing as argument type in a function signature.
    In this case, the type can further be initialized from the following Python objects
    that you can pass as argument instead:
    
    * `None`:
        This will set the ``self._ptr`` attribute to ``NULL`.
    * `~.Pointer` and its subclasses:
        Copies ``pyobj._ptr`` to ``self._ptr``.
    `~.Py_buffer` object ownership is not transferred!
    * `int`:
        Interprets the integer value as pointer address and writes it to ``self._ptr``.
    * `ctypes.c_void_p`:
        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
    * `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol:
        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple 
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.
    * `object` that implements the Python buffer protocol:
        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.
    
    Type checks are performed in the above order.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _ptr_owner (C type ``bint``, protected):
            If this wrapper is the owner of the underlying data.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    """
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
            pyobj (object): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                            or of type `rocrand_generator_base_type`, `int`, or `ctypes.c_void_p`

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
                pyobj,
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
    
    def __int__(self):
        """Returns the data's address as long integer.
        """
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<rocrand_generator_base_type object, self.ptr={int(self)}>"
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


rocrand_generator = rocrand_generator_base_type

class _rocrand_status__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class rocrand_status(_rocrand_status__Base):
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


class _rocrand_rng_type__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class rocrand_rng_type(_rocrand_rng_type__Base):
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


hiprandGenerator_st = rocrand_generator_base_type

hiprandDiscreteDistribution_st = rocrand_discrete_distribution_st

hiprandGenerator_t = rocrand_generator_base_type

hiprandDiscreteDistribution_t = rocrand_discrete_distribution_st

class _hiprandStatus__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hiprandStatus(_hiprandStatus__Base):
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


hiprandStatus_t = hiprandStatus

class _hiprandRngType__Base(enum.IntEnum):
    """Empty enum base class that allows subclassing.
    """
    pass
class hiprandRngType(_hiprandRngType__Base):
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


hiprandRngType_t = hiprandRngType

@cython.embedsignature(True)
def hiprandCreateGenerator(object rng_type):
    r"""Creates a new random number generator.

    Creates a new random number generator of type ``rng_type,``
    and returns it in ``generator.`` That generator will use
    GPU to create random numbers.

    Values for ``rng_type`` are:
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

    Args:
        rng_type (`~.hiprandRngType`):  Type of random number generator to create

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed 

            - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU 

            - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
              dynamically linked library version 

            - HIPRAND_STATUS_TYPE_ERROR if the value for ``rng_type`` is invalid 

            - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type ``rng_type`` is not implemented yet 

            - HIPRAND_STATUS_SUCCESS if generator was created successfully
        * `~.rocrand_generator_base_type`:  Pointer to generator
    """
    generator = rocrand_generator_base_type.from_ptr(NULL)
    if not isinstance(rng_type,_hiprandRngType__Base):
        raise TypeError("argument 'rng_type' must be of type '_hiprandRngType__Base'")
    _hiprandCreateGenerator__retval = hiprandStatus(chiprand.hiprandCreateGenerator(&generator._ptr,rng_type.value))    # fully specified
    return (_hiprandCreateGenerator__retval,generator)


@cython.embedsignature(True)
def hiprandCreateGeneratorHost(object rng_type):
    r"""Creates a new random number generator on host.

    Creates a new host random number generator of type ``rng_type``
    and returns it in ``generator.`` Created generator will use
    host CPU to generate random numbers.

    Values for ``rng_type`` are:
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

    Args:
        rng_type (`~.hiprandRngType`):  Type of random number generator to create

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed 

            - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
              dynamically linked library version 

            - HIPRAND_STATUS_TYPE_ERROR if the value for ``rng_type`` is invalid 

            - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type ``rng_type`` is not implemented yet 

            - HIPRAND_STATUS_SUCCESS if generator was created successfully
        * `~.rocrand_generator_base_type`:  Pointer to generator
    """
    generator = rocrand_generator_base_type.from_ptr(NULL)
    if not isinstance(rng_type,_hiprandRngType__Base):
        raise TypeError("argument 'rng_type' must be of type '_hiprandRngType__Base'")
    _hiprandCreateGeneratorHost__retval = hiprandStatus(chiprand.hiprandCreateGeneratorHost(&generator._ptr,rng_type.value))    # fully specified
    return (_hiprandCreateGeneratorHost__retval,generator)


@cython.embedsignature(True)
def hiprandDestroyGenerator(object generator):
    r"""Destroys random number generator.

    Destroys random number generator and frees related memory.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to be destroyed

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully
    """
    _hiprandDestroyGenerator__retval = hiprandStatus(chiprand.hiprandDestroyGenerator(
        rocrand_generator_base_type.from_pyobj(generator)._ptr))    # fully specified
    return (_hiprandDestroyGenerator__retval,)


@cython.embedsignature(True)
def hiprandGenerate(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed 32-bit unsigned integers.

    Generates ``n`` uniformly distributed 32-bit unsigned integers and
    saves them to ``output_data.``

    Generated numbers are between ``0`` and ``2^32,`` including ``0`` and
    excluding ``2^32.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of 32-bit unsigned integers to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerate__retval = hiprandStatus(chiprand.hiprandGenerate(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned int *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerate__retval,)


@cython.embedsignature(True)
def hiprandGenerateChar(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed 8-bit unsigned integers.

    Generates ``n`` uniformly distributed 8-bit unsigned integers and
    saves them to ``output_data.``

    Generated numbers are between ``0`` and ``2^8,`` including ``0`` and
    excluding ``2^8.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of 8-bit unsigned integers to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateChar__retval = hiprandStatus(chiprand.hiprandGenerateChar(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned char *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateChar__retval,)


@cython.embedsignature(True)
def hiprandGenerateShort(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed 16-bit unsigned integers.

    Generates ``n`` uniformly distributed 16-bit unsigned integers and
    saves them to ``output_data.``

    Generated numbers are between ``0`` and ``2^16,`` including ``0`` and
    excluding ``2^16.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of 16-bit unsigned integers to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateShort__retval = hiprandStatus(chiprand.hiprandGenerateShort(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned short *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateShort__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniform(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed floats.

    Generates ``n`` uniformly distributed 32-bit floating-point values
    and saves them to ``output_data.``

    Generated numbers are between ``0.0f`` and ``1.0f,`` excluding ``0.0f`` and
    including ``1.0f.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of floats to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateUniform__retval = hiprandStatus(chiprand.hiprandGenerateUniform(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniform__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniformDouble(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed double-precision floating-point values.

    Generates ``n`` uniformly distributed 64-bit double-precision floating-point
    values and saves them to ``output_data.``

    Generated numbers are between ``0.0`` and ``1.0,`` excluding ``0.0`` and
    including ``1.0.``

    Note: When ``generator`` is of type: ``HIPRAND_RNG_PSEUDO_MRG32K3A,``
    ``HIPRAND_RNG_PSEUDO_MTGP32,`` or ``HIPRAND_RNG_QUASI_SOBOL32,``
    then the returned ``double`` values are generated from only 32 random bits
    each (one <tt>unsigned int</tt> value per one generated ``double).``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of floats to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateUniformDouble__retval = hiprandStatus(chiprand.hiprandGenerateUniformDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniformDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateUniformHalf(object generator, object output_data, unsigned long n):
    r"""Generates uniformly distributed half-precision floating-point values.

    Generates ``n`` uniformly distributed 16-bit half-precision floating-point
    values and saves them to ``output_data.``

    Generated numbers are between ``0.0`` and ``1.0,`` excluding ``0.0`` and
    including ``1.0.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of halfs to generate

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateUniformHalf__retval = hiprandStatus(chiprand.hiprandGenerateUniformHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n))    # fully specified
    return (_hiprandGenerateUniformHalf__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormal(object generator, object output_data, unsigned long n, float mean, float stddev):
    r"""Generates normally distributed floats.

    Generates ``n`` normally distributed 32-bit floating-point
    values and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of floats to generate

        mean (`~.float`/`~.int`):  Mean value of normal distribution

        stddev (`~.float`/`~.int`):  Standard deviation value of normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(float2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateNormal__retval = hiprandStatus(chiprand.hiprandGenerateNormal(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormal__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormalDouble(object generator, object output_data, unsigned long n, double mean, double stddev):
    r"""Generates normally distributed doubles.

    Generates ``n`` normally distributed 64-bit double-precision floating-point
    numbers and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of doubles to generate

        mean (`~.float`/`~.int`):  Mean value of normal distribution

        stddev (`~.float`/`~.int`):  Standard deviation value of normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(double2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateNormalDouble__retval = hiprandStatus(chiprand.hiprandGenerateNormalDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormalDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateNormalHalf(object generator, object output_data, unsigned long n, int mean, int stddev):
    r"""Generates normally distributed halfs.

    Generates ``n`` normally distributed 16-bit half-precision floating-point
    numbers and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of halfs to generate

        mean (`~.int`):  Mean value of normal distribution

        stddev (`~.int`):  Standard deviation value of normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(half2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateNormalHalf__retval = hiprandStatus(chiprand.hiprandGenerateNormalHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateNormalHalf__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormal(object generator, object output_data, unsigned long n, float mean, float stddev):
    r"""Generates log-normally distributed floats.

    Generates ``n`` log-normally distributed 32-bit floating-point values
    and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of floats to generate

        mean (`~.float`/`~.int`):  Mean value of log normal distribution

        stddev (`~.float`/`~.int`):  Standard deviation value of log normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(float2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateLogNormal__retval = hiprandStatus(chiprand.hiprandGenerateLogNormal(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <float *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormal__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormalDouble(object generator, object output_data, unsigned long n, double mean, double stddev):
    r"""Generates log-normally distributed doubles.

    Generates ``n`` log-normally distributed 64-bit double-precision floating-point
    values and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of doubles to generate

        mean (`~.float`/`~.int`):  Mean value of log normal distribution

        stddev (`~.float`/`~.int`):  Standard deviation value of log normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(double2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateLogNormalDouble__retval = hiprandStatus(chiprand.hiprandGenerateLogNormalDouble(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <double *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormalDouble__retval,)


@cython.embedsignature(True)
def hiprandGenerateLogNormalHalf(object generator, object output_data, unsigned long n, int mean, int stddev):
    r"""Generates log-normally distributed halfs.

    Generates ``n`` log-normally distributed 16-bit half-precision floating-point
    values and saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of halfs to generate

        mean (`~.int`):  Mean value of log normal distribution

        stddev (`~.int`):  Standard deviation value of log normal distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not even, ``output_data`` is not
            aligned to ``sizeof(half2)`` bytes, or ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGenerateLogNormalHalf__retval = hiprandStatus(chiprand.hiprandGenerateLogNormalHalf(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <int *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,mean,stddev))    # fully specified
    return (_hiprandGenerateLogNormalHalf__retval,)


@cython.embedsignature(True)
def hiprandGeneratePoisson(object generator, object output_data, unsigned long n, double lambda_):
    r"""Generates Poisson-distributed 32-bit unsigned integers.

    Generates ``n`` Poisson-distributed 32-bit unsigned integers and
    saves them to ``output_data.``

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to use

        output_data (`~.hip._util.types.Pointer`/`~.object`):  Pointer to memory to store generated numbers

        n (`~.int`):  Number of 32-bit unsigned integers to generate

        lambda (`~.float`/`~.int`):  lambda for the Poisson distribution

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel 

            - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive 

            - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if ``n`` is not a multiple of the dimension
            of used quasi-random generator 

            - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
    """
    _hiprandGeneratePoisson__retval = hiprandStatus(chiprand.hiprandGeneratePoisson(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        <unsigned int *>hip._util.types.Pointer.from_pyobj(output_data)._ptr,n,lambda_))    # fully specified
    return (_hiprandGeneratePoisson__retval,)


@cython.embedsignature(True)
def hiprandGenerateSeeds(object generator):
    r"""Initializes the generator's state on GPU or host.

    Initializes the generator's state on GPU or host.

    If hiprandGenerateSeeds() was not called for a generator, it will be
    automatically called by functions which generates random numbers like
    hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to initialize

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created 

            - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
              a previous kernel launch 

            - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason 

            - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully
    """
    _hiprandGenerateSeeds__retval = hiprandStatus(chiprand.hiprandGenerateSeeds(
        rocrand_generator_base_type.from_pyobj(generator)._ptr))    # fully specified
    return (_hiprandGenerateSeeds__retval,)


@cython.embedsignature(True)
def hiprandSetStream(object generator, object stream):
    r"""Sets the current stream for kernel launches.

    Sets the current stream for all kernel launches of the generator.
    All functions will use this stream.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Generator to modify

        stream (`~.ihipStream_t`/`~.object`):  Stream to use or NULL for default stream

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_SUCCESS if stream was set successfully
    """
    _hiprandSetStream__retval = hiprandStatus(chiprand.hiprandSetStream(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hiprandSetStream__retval,)


@cython.embedsignature(True)
def hiprandSetPseudoRandomGeneratorSeed(object generator, unsigned long long seed):
    r"""Sets the seed of a pseudo-random number generator.

    Sets the seed of the pseudo-random number generator.

    - This operation resets the generator's internal state.
    - This operation does not change the generator's offset.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Pseudo-random number generator

        seed (`~.int`):  New seed value

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator 

            - HIPRAND_STATUS_SUCCESS if seed was set successfully
    """
    _hiprandSetPseudoRandomGeneratorSeed__retval = hiprandStatus(chiprand.hiprandSetPseudoRandomGeneratorSeed(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,seed))    # fully specified
    return (_hiprandSetPseudoRandomGeneratorSeed__retval,)


@cython.embedsignature(True)
def hiprandSetGeneratorOffset(object generator, unsigned long long offset):
    r"""Sets the offset of a random number generator.

    Sets the absolute offset of the random number generator.

    - This operation resets the generator's internal state.
    - This operation does not change the generator's seed.

    Absolute offset cannot be set if generator's type is
    HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Random number generator

        offset (`~.int`):  New absolute offset

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized 

            - HIPRAND_STATUS_SUCCESS if offset was successfully set 

            - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
            or HIPRAND_RNG_PSEUDO_MT19937
    """
    _hiprandSetGeneratorOffset__retval = hiprandStatus(chiprand.hiprandSetGeneratorOffset(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,offset))    # fully specified
    return (_hiprandSetGeneratorOffset__retval,)


@cython.embedsignature(True)
def hiprandSetQuasiRandomGeneratorDimensions(object generator, unsigned int dimensions):
    r"""Set the number of dimensions of a quasi-random number generator.

    Set the number of dimensions of a quasi-random number generator.
    Supported values of ``dimensions`` are 1 to 20000.

    - This operation resets the generator's internal state.
    - This operation does not change the generator's offset.

    Args:
        generator (`~.rocrand_generator_base_type`/`~.object`):  Quasi-random number generator

        dimensions (`~.int`):  Number of dimensions

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_NOT_CREATED if the generator wasn't created 

            - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator 

            - HIPRAND_STATUS_OUT_OF_RANGE if ``dimensions`` is out of range 

            - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully
    """
    _hiprandSetQuasiRandomGeneratorDimensions__retval = hiprandStatus(chiprand.hiprandSetQuasiRandomGeneratorDimensions(
        rocrand_generator_base_type.from_pyobj(generator)._ptr,dimensions))    # fully specified
    return (_hiprandSetQuasiRandomGeneratorDimensions__retval,)


@cython.embedsignature(True)
def hiprandGetVersion():
    r"""Returns the version number of the cuRAND or rocRAND library.

    Returns in ``version`` the version number of the underlying cuRAND or
    rocRAND library.

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``version`` is NULL 

            - HIPRAND_STATUS_SUCCESS if the version number was successfully returned
        * `~.int`:  Version of the library
    """
    cdef int version
    _hiprandGetVersion__retval = hiprandStatus(chiprand.hiprandGetVersion(&version))    # fully specified
    return (_hiprandGetVersion__retval,version)


@cython.embedsignature(True)
def hiprandCreatePoissonDistribution(double lambda_):
    r"""Construct the histogram for a Poisson distribution.

    Construct the histogram for the Poisson distribution with lambda ``lambda.``

    Args:
        lambda (`~.float`/`~.int`):  lambda for the Poisson distribution

    Returns:
        A `~.tuple` of size 2 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated 

            - HIPRAND_STATUS_OUT_OF_RANGE if ``discrete_distribution`` pointer was null 

            - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive 

            - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully
        * `~.rocrand_discrete_distribution_st`:  pointer to the histogram in device memory
    """
    discrete_distribution = rocrand_discrete_distribution_st.from_ptr(NULL)
    _hiprandCreatePoissonDistribution__retval = hiprandStatus(chiprand.hiprandCreatePoissonDistribution(lambda_,&discrete_distribution._ptr))    # fully specified
    return (_hiprandCreatePoissonDistribution__retval,discrete_distribution)


@cython.embedsignature(True)
def hiprandDestroyDistribution(object discrete_distribution):
    r"""Destroy the histogram array for a discrete distribution.

    Destroy the histogram array for a discrete distribution created by
    hiprandCreatePoissonDistribution.

    Args:
        discrete_distribution (`~.rocrand_discrete_distribution_st`/`~.object`):  pointer to the histogram in device memory

    Returns:
        A `~.tuple` of size 1 that contains (in that order):

        * `~.hiprandStatus`: HIPRAND_STATUS_OUT_OF_RANGE if ``discrete_distribution`` was null 

            - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully
    """
    _hiprandDestroyDistribution__retval = hiprandStatus(chiprand.hiprandDestroyDistribution(
        rocrand_discrete_distribution_st.from_pyobj(discrete_distribution)._ptr))    # fully specified
    return (_hiprandDestroyDistribution__retval,)
