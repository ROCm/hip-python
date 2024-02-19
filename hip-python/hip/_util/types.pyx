# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

cimport cpython.bool
cimport cpython.long
cimport cpython.buffer
cimport cpython.bytes
cimport cpython.string
cimport cpython.ref

cimport libc.stdlib
cimport libc.string
cimport libc.stdint

import ctypes
import math

__all__ = [
    # __all__ is important for generating the API documentation in source order
    "Pointer",
    "CStr",
    "ImmortalCStr",
    "DeviceArray",
    "ListOfBytes",
    "ListOfPointer",
    "ListOfInt",
    "ListOfUnsigned",
    "ListOfUnsignedLong",
]

cdef class Pointer:
    """Datatype for handling Python arguments that need to be converted to a pointer type.

    Datatype for handling Python arguments that need to be converted to a pointer type
    when passed to an underlying C function.

    This type stores a C ``void *`` pointer to the original Python object's data
    plus an additional `Py_buffer` object if the pointer has ben acquired from a
    Python object that implements the `Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

    This type can be constructed from input objects that are implementors of the
    `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.

    In summary, the type can be initialized from the following Python objects:

    * `None`:

        This will set the ``self._ptr`` attribute to ``NULL``.

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown reasons.
        Therefore, it must be checked for this type first.

    * `object` that implements the Python buffer protocol:

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `object` that implements the CUDA Array Interface protocol:

        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.

    * `~.Pointer`:

        Copies ``pyobj._ptr`` to ``self._ptr``.
        `~.Py_buffer` object ownership is not transferred!

    * `int`:

        Interprets the integer value as pointer address and writes it to ``self._ptr``.

    * `object` that has `as_c_void_p(self)` method:

        Takes the pointer address ``pyobj.as_c_void_p().value`` and writes it to ``self._ptr``.

    Type checks are performed in the above order.

    Note:
        When initializing `~.Pointer` instances from a Python input object,
        buffer types are checked first by purpose.
        Acquiring/releasing a buffer typically implies that the reference count
        of the buffer is incremented/decremented.
        If the Python input object releases a buffer but a
        `~.Pointer` instance still has acquired it,
        the buffer data will not be freed until the `~.Pointer` instance is deleted.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    """
    # C members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False

    cdef void* getPtr(self):
        return self._ptr

    @staticmethod
    cdef Pointer fromPtr(void* ptr):
        cdef Pointer wrapper = Pointer.__new__(Pointer)
        wrapper._ptr = ptr
        return wrapper

    cpdef Pointer createRef(self):
        """Creates are reference to this pointer.

        Returns a `~.Pointer` that stores the address of this `~.Pointer's data pointer.

        Note:
            No ownership information is transferred.
        """
        return Pointer.fromPtr(<void*>&self._ptr)

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of Pointer, only the pointer is copied.
            Releasing an acquired Py_buffer handles is still an obligation of the original object.
        """
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        self._py_buffer_acquired = False
        if pyobj is None:
            self._ptr = NULL
        elif isinstance(pyobj,ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
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
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            self._ptr = cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif isinstance(pyobj,Pointer):
            self._ptr = (<Pointer>pyobj)._ptr
        elif isinstance(pyobj,int):
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj)
        elif hasattr(pyobj,"as_c_void_p"):
            # NOTE: This must stay down here because 'as_c_void_p' is
            #       an interface provided by all the *_util.types types
            #       that should only be used if the type couldn't
            #       be identified as a Python buffer.
            c_void_p_value = pyobj.as_c_void_p().value
            self._ptr = cpython.long.PyLong_AsVoidPtr(c_void_p_value) if c_void_p_value != None else NULL
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")

    @staticmethod
    def fromObj(pyobj):
        """Creates a Pointer from the given object.

        In case ``pyobj`` is itself a ``Pointer`` instance, this method
        returns it directly. No new ``Pointer`` is created.
        """
        return Pointer.fromPyobj(pyobj)

    @staticmethod
    cdef Pointer fromPyobj(object pyobj):
        """Creates a Pointer from the given object.

        In case ``pyobj`` is itself an ``Pointer`` instance, this method
        returns it directly. No new ``Pointer`` is created.

        Args:
            pyobj (`object`):
                Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                or of type `~.Pointer`, `int`, or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `~.Pointer`.
        """
        cdef Pointer wrapper

        if isinstance(pyobj,Pointer):
            return pyobj
        else:
            wrapper = Pointer.__new__(Pointer)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    @property
    def is_ptr_null(self):
        """If data pointer is NULL.
        """
        return self._ptr == NULL

    def __nonzero__(self):
        """If this object points to meaningful data.
        """
        return self._ptr != NULL

    def __int__(self):
        """Integer representation of the data pointer.
        """
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<Pointer object, _ptr={int(self)}>"
    def as_c_void_p(self):
        """Data pointer as ``ctypes.c_void_p``.
        """
        return ctypes.c_void_p(int(self))

    def __getitem__(self,offset):
        """Returns a new Pointer whose pointer is this instance's pointer offsetted by ``offset``.

        Args:
            offset (`int`): Offset (in bytes) to add to this instance's pointer.
        """
        cdef Pointer result
        if isinstance(offset,int):
            if offset < 0:
                raise ValueError("offset='{offset}' must be non-negative")
            return Pointer.fromPtr(<void*>(<unsigned long>self._ptr + cpython.long.PyLong_AsUnsignedLong(offset)))
        raise NotImplementedError("'__getitem__': not implemented for other 'offset' types than 'int'")

    def __init__(self,object pyobj = None):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.Pointer` for information
                about accepted types for ``pyobj``.
                Defaults to None.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """

        Pointer.init_from_pyobj(self,pyobj)


cdef class CStr(Pointer):
    """Datatype for handling C strings (`char *` and related).

    Datatype for handling C strings (`char *`). Cython's parameter
    autoconversion creates duplicates of C strings and hence loses
    the original data's address, which can be an issue.

    This implementation assumes that this type is mainly used like a Python
    `str` in cases where it is returned by a function.
    Hence, the `__getitem__`, `__repr__` and `__str__` implementation of this class
    decode the underlying data as `UTF-8` string. ASCII is a subset of UTF-8.
    Note that this design choice is irrelevant for the case where the type is
    used as adapter to convert Python arguments to a C string.

    This datatype implements the Python buffer protocol. Therefore, different
    decoding of the underlying data can be achieved by passing this type
    to the constructor of `bytes` or to other array types or memory views that can deal
    with Python buffers.

    Warning:
        When using this type as adapter, be aware that `bytes` and `str`
        objects passed to the constructor of this class might get garbage collected.
        If the called C library stores pointers to the data of these Python objects
        into a library-managed data structure and the latter is then used
        outside of the original scope, you might experience memory errors.

    Limitation:
        This class is only designed for handling strings that encode each
        character with 8 bits (ASCII and UTF-8). Smaller or larger symbols are not supported.

    The type can be initialized from the following Python objects:

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        Length information is obtained via ``strlen`` in this case.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown reasons.
        Therefore, it must be checked for this type first.

    * `object` that implements the Python buffer protocol:

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `object` that is accepted as input by `~.Pointer.__init__`.

    Type checks are performed in the above order.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _shape (`Py_size_t[1]`, protected):
            Size of the wrapped zero-terminated C char,
            stored into first array element.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
    """
    # C members declared in declaration part ``types.pxd``

    def __cinit__(self):
        self._is_ptr_owner = False
        self._shape[0] = 0 # must be zero

    cdef const char* getElementPtr(self):
        return <const char*>self._ptr

    cdef Py_ssize_t get_or_determine_len(self):
        """Get/Determine the length of the C string.

        Returns 0 in case of `self._ptr` being 0.
        """
        if self._ptr == NULL:
            self._shape[0] = 0
        elif self._shape[0] == 0:
            self._shape[0] = libc.string.strlen(<const char*>self._ptr)
        return self._shape[0]

    @staticmethod
    cdef CStr fromPtr(void* ptr):
        """Initialize a new CStr instance from a pointer.

        Note:
            For output arguments (char**), there will be a NULL pointer
            passed here. In this case, the self._shape array
            cannot be initialized. As the length of the C char depends on the location of the 0-char,
            we postpone length calculations to a later stage whenenver this
            information is required.
        """
        cdef CStr wrapper = CStr.__new__(CStr)
        wrapper._ptr = ptr
        wrapper.get_or_determine_len()
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `CStr`, only the pointer and
            length information is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        if isinstance(pyobj,CStr):
            self._ptr = (<CStr>pyobj)._ptr
            self._shape[0] = (<CStr>pyobj)._shape[0]
        elif isinstance(pyobj,ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
            self.get_or_determine_len()
        elif isinstance(pyobj,str):
            raise RuntimeError("CStr.init_from_pyobj: currently no support for Python `str` objects.")
            # self._ptr = <void*>cpython.string.PyString_AsString(pyobj) # caused 'undefined reference' at runtime
            # self._shape[0] = cpython.string.PyString_Size(pyobj)       # caused 'undefined reference' at runtime
            # self._shape[0] = len(pyobj)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj): # handles 'bytes' too
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &self._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            self._py_buffer_acquired = True
            self._ptr = self._py_buffer.buf
            self._shape[0] = self._py_buffer.len
        else:
            Pointer.init_from_pyobj(self,pyobj)
            self.get_or_determine_len()

    @staticmethod
    def fromObj(pyobj):
        """Creates a CStr from the given object.

        In case ``pyobj`` is itself a ``CStr`` instance, this method
        returns it directly. No new ``CStr`` is created.
        """
        return CStr.fromPyobj(pyobj)

    @staticmethod
    cdef CStr fromPyobj(object pyobj):
        """Derives a CStr from the given object.

        In case ``pyobj`` is itself an ``CStr`` instance, this method
        returns it directly. No new ``CStr`` is created.

        Args:
            pyobj (`object`): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                or of type `CStr`, `int`, or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of CStr.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef CStr wrapper

        if isinstance(pyobj,CStr):
            return pyobj
        else:
            wrapper = CStr.__new__(CStr)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    cpdef void malloc(self,Py_ssize_t size_bytes):
        """Dynamically allocate a buffer of bytes for this CStr.

        Args:
            size_bytes (`Py_ssize_t`): The number of bytes to allocate.
        Note:
            Throws `~.RuntimeError` if the data pointer is not NULL as this
            indicates that this instance handles external data.
        Note:
            Sets the _is_ptr_owner flag.
        """
        if self._ptr != NULL:
            raise RuntimeError("Data pointer must be NULL.")
        self._ptr = libc.stdlib.malloc(size_bytes)
        libc.string.memset(<void*>self._ptr, 0, size_bytes)
        self._is_ptr_owner = True

    cpdef void free(self):
        """Free dynamically allocated data.

        Note:
            Simply returns if the data pointer is NULL.
        Note:
            Throws `~.RuntimeError` if this instance does not own the data that ought to be freed.
        Note:
            Unsets the _is_ptr_owner flag.
        """
        if self._is_ptr_owner == False:
            raise RuntimeError("Attempt to free that is not owned by this instance.")
        if self._ptr == NULL:
            return # do nothing
        libc.stdlib.free(self._ptr)
        self._is_ptr_owner = False

    def __dealloc__(self):
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            self.free()

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.CStr` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        CStr.init_from_pyobj(self,pyobj)

    def __len__(self) -> int:
        """The number of chars/bytes of the C string.
        """
        return self.get_or_determine_len()

    def __getitem__(self, subscript):
        """Get individual chars or slice the underlying chars.

        Note:
            Copies into a temporary str object
            if `subscript` is a slice.
        """
        if self._ptr == NULL:
            raise RuntimeError("__getitem__: data pointer `_ptr` is `NULL`.")
        return str(self)[subscript]

    def __str__(self) -> str:
        """Decodes the bytes representation of this C string as UTF-8 string.

        Decodes the bytes representation of this C string as UTF-8 string.
        Returns None if the underlying pointer is None.

        Note:
            See the decode routine for representing this object's
            data in different formats.
        """
        return bytes(self).decode("utf-8")

    def __nonzero__(self):
        """Implements Python `str` like behavior.
        """
        return self.get_or_determine_len() > 0

    def __repr__(self):
        return self.__str__()

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """Buffer protocol routine for acquiring a view on this `CStr`'s data.

        Note:
            `__getbuffer__` and `__releasebuffer__` allow to convert this
            object to bytes.
        Note:
            `buffer.len` and `buffer.shape` are computed on-the-fly (if not set already)
            via `CStr.get_or_determine_len(self)`.
        See:
            For details on the Python buffer protocol see https://peps.python.org/pep-3118/.
        """
        buffer.buf = <char *>(self._ptr)
        buffer.format = NULL # NULL implies bytes, 'B'
        buffer.internal = NULL # for storing context
                               # for the implementor at dealloc time
        buffer.itemsize = 1
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.len = self.get_or_determine_len() # product(_shape) * itemsize
        buffer.shape = self._shape # must follow buffer.len
        buffer.strides = NULL # contiguous
        buffer.suboffsets = NULL # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        """Buffer protocol routine for releasing a view on this `CStr`'s data.
        """
        pass

    def encode(self, /, encoding="utf-8", errors="strict"):
        """Return a `bytes` object with respect to the encoding.

        See:
            `str.encode`
        """
        return self.decode(encoding=encoding,errors=errors).encode(
            encoding=encoding,errors=errors)

    def decode(self, /, encoding="utf-8", errors="strict"):
        """Return a `str` object with respect to the enconding.

        See:
            `bytes.decode`
        """
        return bytes(self).decode(encoding=encoding,errors=errors)


cdef class ImmortalCStr(CStr):
    """Immortal version of `CStr` that sets
    the reference count of itself `1` initially,
    which prevents it from getting garbage collected.
    Furthermore, increases the reference count
    of wrapped bytes

    Note:
        Class name and implementation inspired from:
        https://peps.python.org/pep-0683
    """

    def __cinit__(self):
        CStr.__cinit__(self)
        cpython.ref.Py_INCREF(self)

    cdef void init_from_pyobj(self, object pyobj):
        CStr.init_from_pyobj(self, pyobj)
        cpython.ref.Py_INCREF(pyobj)

    @staticmethod
    cdef ImmortalCStr fromPtr(void* ptr):
        cdef ImmortalCStr wrapper = ImmortalCStr.__new__(CStr)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates an ImmortalCStr from the given object.

        In case ``pyobj`` is itself a ``ImmortalCStr`` instance, this method
        returns it directly. No new ``ImmortalCStr`` is created.
        """
        return ImmortalCStr.fromPyobj(pyobj)

    @staticmethod
    cdef ImmortalCStr fromPyobj(object pyobj):
        cdef ImmortalCStr wrapper

        if isinstance(pyobj,ImmortalCStr):
            return pyobj
        else:
            wrapper = ImmortalCStr.__new__(ImmortalCStr)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __init__(self,object pyobj):
        """Constructor.
        """
        CStr.init_from_pyobj(self,pyobj)

cdef class NDBuffer(Pointer):
    """Datatype for handling contiguous n-dimensional buffers of various element types.

    Datatype for handling contiguous n-dimensional buffers of various element types.
    The buffer can be reshaped via its ``configure`` method.

    Note:
        This buffer does not provide any routines to read or write
        elements of the buffer. Instead, its ``__getitem__`` operator is overloaded
        to return ``NDBuffer`` instances pointing to contiguous subregions
        or single elements of the original buffer. If this buffer is wrapped around host data,
        users can convert it to types that allow access to the underlying
        data such as `bytes`, `bytearray` or numpy array types as this
        type implements the Python buffer protocol.

    This type implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
    protocol. Note, however that it is the user's obligation to only pass this type to consumers of the
    CUDA array interface if and only if the underlying data is device data.

    It can be initialized from the following Python objects:

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        No length information can be obtained in this case.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown reasons.
        Therefore, it must be checked for this type first.

    * `object` with ``__cuda_array_interface__`` member:
        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.
        Copies shape and type information.

    * `object` that implements the Python buffer protocol:

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer.__init__` for more information.

    Note:
        Type checks are performed in the above order.

    Note:
        Shape and type information and other metadata can be modified or overwritten after creation via the `~.configure`
        member function. be aware that you might need to pass the ``_force=True`` keyword argument ---
        in particular if your instance was created from a type that does not implement the
        `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.
    See:
        `~.configure`

    C Attributes:
        _ptr (``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer (`~.Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (`bool`, protected):
            Stores a pointer to the data of the original Python object.
        __dict__ (`dict`, protected):
            Dict with member ``__cuda_array_interface__``.
        _itemsize (``size_t``, protected):
            Stores the itemsize. The item size is not member of
            ``__cuda_array_interface__``.
        _py_buffer_shape (``Py_Ssize_t*``, private):
            A buffer to pass shape information to consumers
            of this Python buffer.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<NDBuffer object, _ptr={int(self)}, typestr={self.typestr}, itemsize={self.itemsize}, shape={str(self.shape)}, is_read_only={self.is_read_only}, stream={self.stream_as_int}>"

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

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False
        self.__view_count = 0
        self._py_buffer_shape = NULL
        self._itemsize = 1
        self.__dict__ = dict(
            __cuda_array_interface__ = dict(
               shape=(1,), # by default assume a single byte
               typestr='B', # See: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
               data=(None,False), # 1: data pointer as int (long int), 2: read-only
               strides=None,
               offset=0,
               mask=None,
               version=3,
               # numba
               stream=None, #
           )
        )

    cdef _set_ptr(self,void* ptr):
        """Sets the `self._ptr` C member and the 'data' field in the CUDA array interface.
        """
        cdef tuple old_data = self.__dict__["__cuda_array_interface__"]["data"]
        self._ptr = ptr
        self.__dict__["__cuda_array_interface__"]["data"] = (cpython.long.PyLong_FromVoidPtr(ptr),old_data[1])

    @staticmethod
    cdef NDBuffer fromPtr(void* ptr):
        cdef NDBuffer wrapper = NDBuffer.__new__(NDBuffer)
        wrapper._set_ptr(ptr)
        return wrapper

    @property
    def rank(self):
        """Rank of the underlying data.

        See:
            set_bounds
        """
        cdef size_t rank = 0
        for r in self.__dict__["__cuda_array_interface__"]["shape"]:
            if r > 1:
                rank += 1
        return rank

    def configure(self, **kwargs):
        """(Re-)configure this contiguous n-dimensional buffer.

        Warning:
            When you reconfigure the buffer shape, previously acquired
            views on this NDBuffer via the Python buffer protocol
            might become invalid. Therefore, a `RuntimeException`
            is thrown if this method is called while the view count
            is greater than zero.

        Keyword arguments:
            shape (`tuple`):
                A tuple that describes the extent per dimension.
                The length of the tuple is the number of dimensions.
            typestr (`str`):
                A numpy typestr, see the notes for more details.
            stream (`int` or `None`):
                The stream to synchronize before consuming
                this array. See first note for more details.
                Only makes sense if this buffer wraps device data.
            itemsize (`int`):
                Size in bytes of each item. Defaults to 1. See the notes.
            read_only (`bool`):
                `NDBuffer` is read_only. Second entry of the
                CUDA array interface 'data' tuple. Defaults to False.
            _force(`bool`):
                Ignore changes in the total number of bytes when
                overriding shape, typestr, and/or itemsize.

        Note:
            More details on the keyword arguments can be found here:
            https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html

        Note:
            This method does not automatically map all existing numpy/numba typestr to appropriate number
            of bytes, i.e. `itemsize`. Hence, you need to specify itemsize additionally
            when dealing with other datatypes than bytes (typestr: ``'b'``).
        """
        cdef list supported_keys = ["shape","typestr","stream"]
        cdef list extra_keys = ["itemsize","read_only","_force"]
        cdef str allowed_keys_str =", ".join([f"'{e}'" for e in supported_keys + extra_keys])
        cdef bint force_new_shape
        cdef tuple shape
        cdef tuple old_shape
        cdef tuple old_data
        cdef bint read_only
        cdef int itemsize = -1
        cdef str typestr = None

        if self.__view_count > 0:
            raise RuntimeError("cannot re-configure this NDBuffer while it is viewed by other objects via the Python buffer protocol")

        for k in kwargs:
            if k not in (supported_keys + extra_keys):
                raise KeyError(f"allowed keyword arguments are: {allowed_keys_str}")

        force_new_shape = kwargs.get("_force",False)
        shape = old_shape = self.__dict__["__cuda_array_interface__"]["shape"]
        if "shape" in kwargs:
            shape = kwargs["shape"]
            if not len(shape):
                raise ValueError("'shape': must have at least one entry")
            for i in shape:
                if not isinstance(i,int):
                    raise TypeError("'shape': entries must be int")
            #self.__dict__["__cuda_array_interface__"]["shape"] = shape
        if "typestr" in kwargs:
            typestr = kwargs["typestr"]
            self.__dict__["__cuda_array_interface__"]["typestr"] = typestr
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
                self.__dict__["__cuda_array_interface__"]["stream"] = stream
            else:
                self.__dict__["__cuda_array_interface__"]["stream"] = int(Pointer.fromPyobj(stream))
        if "read_only" in kwargs:
            read_only = kwargs["read_only"]
            if not isinstance(read_only,bool):
                raise ValueError("'read_only:' expected bool")
            old_data = self.__dict__["__cuda_array_interface__"]["data"]
            self.__dict__["__cuda_array_interface__"]["data"] = (old_data[0],read_only)

        if itemsize > 0 or shape != old_shape:
            old_num_bytes = self._itemsize * math.prod(old_shape)
            if itemsize < 0:
                itemsize = self._itemsize
            new_num_bytes = itemsize * math.prod(shape)
            if old_num_bytes == new_num_bytes or force_new_shape:
                self._itemsize = itemsize
                self.__dict__["__cuda_array_interface__"]["shape"] = shape
            else:
                raise ValueError(f"new shape would change buffer size information: {old_num_bytes} B -> {new_num_bytes} B. Additionaly specify `_force=True` if this is intended.")

        return self

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of NDBuffer, only the pointer is copied.
            Releasing an acquired Py_buffer handles is still an obligation of the original object.
        """
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        self._py_buffer_acquired = False
        if pyobj is None:
            self._set_ptr(NULL)
        elif isinstance(pyobj,ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj): # handles 'bytes' too
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &self._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            self._py_buffer_acquired = True
            self._set_ptr(self._py_buffer.buf)

            shape = [cpython.long.PyLong_FromSsize_t(self._py_buffer.shape[i])
                    for i in range(0,self._py_buffer.ndim)]
            if self._py_buffer.format == NULL:
                typestr = 'B' # see https://peps.python.org/pep-3118/#the-py-buffer-struct
            else:
                typestr = cpython.bytes.PyBytes_FromString(self._py_buffer.format).decode("utf-8")
            itemsize = cpython.long.PyLong_FromSsize_t(self._py_buffer.itemsize)
            read_only = cpython.bool.PyBool_FromLong(<long>self._py_buffer.readonly)
            self.configure(
                _force=True,
                typestr=typestr,
                itemsize=itemsize,
                shape=tuple(shape),
                read_only=read_only,
            )
            self.__dict__["__pybuffer_obj"] = self._py_buffer.obj
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            if cuda_array_interface["strides"] != None:
                raise RuntimeError("CUDA array interface is not contiguous")
            ptr_as_int = cuda_array_interface["data"][0]
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(ptr_as_int))
            self.configure(cuda_array_interface)
            if isinstance(pyobj,NDBuffer):
                self._itemsize = pyobj._itemsize
        else:
            pointer = Pointer.fromPyobj(pyobj)
            self._set_ptr(pointer._ptr)

    @staticmethod
    def fromObj(pyobj):
        """Creates a NDBuffer from the given object.

        In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
        returns it directly. No new ``NDBuffer`` is created.
        """
        return NDBuffer.fromPyobj(pyobj)

    @staticmethod
    cdef NDBuffer fromPyobj(object pyobj):
        """Creates a NDBuffer from the given object.

        In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
        returns it directly. No new ``NDBuffer`` is created.

        Args:
            pyobj (`object`):
                Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                an `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
                protocol, or an instance of `Pointer`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `NDBuffer`.
        """
        cdef NDBuffer wrapper = NDBuffer.__new__(NDBuffer)

        if isinstance(pyobj,NDBuffer):
            return pyobj
        else:
            wrapper = NDBuffer.__new__(NDBuffer)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    cdef tuple _handle_int(self,size_t subscript, size_t shape_dim):
        if subscript < 0:
            raise ValueError(f"subscript='{subscript}' must be non-negative.")
        if subscript >= shape_dim:
            raise ValueError(f"subscript='{subscript}' must be smaller than axis' exclusive upper bound ('{shape_dim}')")
        return (subscript,subscript+1)


    cdef tuple _handle_slice(self,slice subscript,size_t shape_dim):
        cdef size_t start = -1
        cdef size_t stop = -1
        cdef bint extract_full_dim = False

        if subscript.step not in (None,1):
            raise ValueError(f"subscript's step='{subscript.step}' must be 'None' or '1'.")
        if subscript.stop != None:
            if subscript.stop <= 0:
                raise ValueError(f"subscript's stop='{subscript.stop}' must be greater than zero.")
            if subscript.stop > shape_dim:
                raise ValueError(f"subscript's stop='{subscript.stop}' must not be greater than axis' exclusive upper bound ({shape_dim}).")
            stop = subscript.stop
        else:
            stop = shape_dim
        if subscript.start != None:
            if subscript.start < 0:
                raise ValueError(f"subscript's start='{subscript.start}' must be non-negative.")
            if subscript.start >= shape_dim:
                raise ValueError(f"subscript's start='{subscript.start}' must be smaller than axis' exclusive upper bound ({shape_dim}).")
            start = subscript.start
        else:
            start = 0

        if start >= stop:
            raise ValueError(f"subscript's stop='{subscript.stop}' must be greater than subscript's start='{subscript.start}'")

        extract_full_dim = (
            start == 0
            and stop == shape_dim
        )
        return (start,stop,extract_full_dim)


    def __getitem__(self,subscript):
        """Returns a contiguous subarray according to the subscript expression.

        Returns a contiguous subarray according to the subscript expression.

        Args:
            subscript (`int`/`slice`/`tuple`):
                Either an integer, a slice, or a tuple of slices and integers.

        Note:
            If the subscript is a single integer, e.g. `[i]`, the subarray `[i,:,:,...,:]` is returned.
            A `KeyError` is raised if the extent of axis 0 is surpassed. This behavior is identical to that of numpy.

        Raises:
            `TypeError`: If the subscript types are not 'int', 'slice' or a 'tuple' thereof.
            `ValueError`: If the subscripts do not yield an contiguous subarray. A single array element is regarded as contiguous array of size 1.
        """
        cdef size_t stride = 1
        cdef size_t offset = 0
        cdef bint next_slice_yields_contiguous = True
        cdef tuple shape = self.__dict__["__cuda_array_interface__"]["shape"]
        cdef size_t len_shape = len(shape)
        cdef list result_shape = list() # elements will be appended
        cdef list expanded_subscript = list()
        cdef size_t len_subscript

        if isinstance(subscript,tuple):
            expanded_subscript += subscript[:]
            len_subscript = len(subscript)
        elif isinstance(subscript,(slice,int)):
            expanded_subscript = [subscript]
            len_subscript = 1
        else:
            raise TypeError(f"subscript type='{type(subscript)}' is none of: 'slice', 'int', 'tuple'")
        # check len and pad ':' slices if the subscript tuple's size is smaller than the array's shape dimensions.
        if len_shape < len_subscript:
            raise IndexError(f"too many indices specified, maximum number of indices that can be specified is {len_shape}")
        if len_shape > len_subscript:
            expanded_subscript += [slice(None)]*(len_shape-len_subscript)
        for _i,spec in enumerate(reversed(expanded_subscript)): # row major
            i = len_shape-_i-1
            if isinstance(spec,int):
                (start,stop) = self._handle_int(spec,shape[i])
                next_slice_yields_contiguous = False
            elif isinstance(spec,slice):
                if not next_slice_yields_contiguous:
                    raise ValueError(f"subscript='{expanded_subscript}' yields no contiguous subarray")
                (start,stop,extract_full_dim) = self._handle_slice(spec,shape[i])
                next_slice_yields_contiguous = extract_full_dim
                # extract_full_dim => start == 0
            else:
                raise TypeError(f"subscript tuple entry type='{type(spec)}' is none of: 'slice', 'int'")
            result_shape.append(stop-start)
            offset += start*stride
            stride *= <size_t>shape[i]
        offset *= self._itemsize # scale offset with itemsize
        return NDBuffer.fromPtr(<void*>(<unsigned long>self._ptr + offset)).configure(
            _force=True,
            typestr=self.typestr,
            itemsize=self.itemsize,
            shape=tuple(result_shape),
            read_only=self.is_read_only,
            stream=self.stream_as_int,
        )

    def __getattribute__(self,key):
        """Synchronize interface data whenever it is accessed.
        """
        if key == "__cuda_array_interface__":
            self._set_ptr(self._ptr)
        return super().__getattribute__(key)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """Buffer protocol routine for acquiring a view on this NDBuffer's data.

        Note:
            `__getbuffer__` and `__releasebuffer__` allow to convert this
            object to bytes.
        Note:
            The caller is responsible for keeping a reference to obj until releasebuffer is called.
        See:
            For details on the Python buffer protocol see https://peps.python.org/pep-3118/.
        """
        cdef Py_ssize_t ndim = cpython.long.PyLong_AsSsize_t(len(self.shape))
        cdef Py_ssize_t size = cpython.long.PyLong_AsSsize_t(self.size)
        # reallocate the shape buffer
        if self._py_buffer_shape != NULL:
            libc.stdlib.free(self._py_buffer_shape)
        self._py_buffer_shape = <Py_ssize_t*>libc.stdlib.malloc(ndim*sizeof(Py_ssize_t))
        shape = self.shape
        for i in range(0,ndim):
            self._py_buffer_shape[i] = cpython.long.PyLong_AsSsize_t(shape[i])

        buffer.buf = <char *>(self._ptr)
        self.__dict__["__typestr_bytes"] = self.typestr.encode(
            "utf-8")+b"\x00"  # NUL-terminated, reference must stay alive
        buffer.format = cpython.bytes.PyBytes_AsString(self.__dict__["__typestr_bytes"])
        buffer.internal = NULL # for storing context
                               # for the implementor at dealloc time
        buffer.itemsize = self._itemsize
        buffer.ndim = ndim
        buffer.obj = self
        buffer.readonly = self.is_read_only
        buffer.len = size*self._itemsize
        buffer.shape = self._py_buffer_shape
        buffer.strides = NULL # contiguous
        buffer.suboffsets = NULL # for pointer arrays only

        self.__view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        """Buffer protocol routine for releasing a view on this NDBuffer's data.

        Decrements the view count.
        """
        self.__view_count -= 1

    @property
    def typestr(self):
        """The type string (see `CUDA Array Interface specification <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#python-interface-specification>`_).
        """
        return self.__dict__["__cuda_array_interface__"]["typestr"]
    @property
    def shape(self):
        """A tuple of int (or long) representing the size of each dimension.
        """
        return self.__dict__["__cuda_array_interface__"]["shape"]
    @property
    def size(self):
        """Product of the `~.shape` entries.
        """
        return math.prod(self.__dict__["__cuda_array_interface__"]["shape"])
    @property
    def itemsize(self):
        """Number of bytes required to store a single element of the array.
        """
        return self._itemsize
    @property
    def is_read_only(self):
        """If the data is read only, i.e. must not be modified.
        """
        return self.__dict__["__cuda_array_interface__"]["data"][1]
    @property
    def stream_as_int(self):
        """Returns the stream address as integer value.
        """
        return self.__dict__["__cuda_array_interface__"]["stream"]

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.NDBuffer` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.

        Note:
            Shape and type information and other metadata can be modified or overwritten after creation via the `~.configure`
            member function. be aware that you might need to pass the ``_force=True`` keyword argument ---
            in particular if your instance was created from a type that does not implement the
            `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.
        See:
            `~.configure`
        """
        NDBuffer.init_from_pyobj(self,pyobj)

    def __dealloc__(self):
        if self._py_buffer_shape != NULL:
            libc.stdlib.free(self._py_buffer_shape)
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)

cdef class DeviceArray(NDBuffer):
    """Datatype for handling device buffers.

    Datatype for handling device buffers returned by `~.hipMalloc` and related device
    memory allocation routines.

    This type implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.

    It can be initialized from the following Python objects:

    * `None`:
        This will set the ``self._ptr`` attribute to ``NULL``.
        No shape and type information is available in this case!
    * `object` that is accepted as input by `~.Pointer.__init__`:
        In this case, init code from `~.Pointer` is used.
        `~.Py_buffer` object ownership is not transferred
        See `~.Pointer.__init__` for more information.
        No shape and type information is available in this case!
    * `int`:
        Interprets the integer value as pointer address and writes it to ``self._ptr``.
        No shape and type information is available in this case!
    * `ctypes.c_void_p`:
        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        No shape and type information is available in this case!
    * `object` with ``__cuda_array_interface__`` member:
        Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple
        from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.
        Copies shape and type information.

    Note:
        Type checks are performed in the above order.

    Note:
        Shape and type information and other metadata can be modified or overwritten after creation via the `~.configure`
        member function. be aware that you might need to pass the ``_force=True`` keyword argument ---
        in particular if your instance was created from a type that does not implement the
        `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.
    See:
        `~.configure`

    C Attributes:
        _ptr (``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer (`~.Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (`bool`, protected):
            Stores a pointer to the data of the original Python object.
        _itemsize (``size_t``, protected):
            Stores the itemsize.
        __dict__ (`dict`, protected):
            Dict with member ``__cuda_array_interface__``.
    """
    # C members declared in declaration part ``types.pxd``

    @staticmethod
    def DeviceArray(pyobj):
        """Creates a NDBuffer from the given object.

        In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
        returns it directly. No new ``NDBuffer`` is created.
        """
        return DeviceArray.fromPyobj(pyobj)

    @staticmethod
    cdef DeviceArray fromPyobj(object pyobj):
        """Creates a NDBuffer from the given object.

        In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
        returns it directly. No new ``NDBuffer`` is created.

        Args:
            pyobj (`object`):
                Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                an `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
                protocol, or an instance of `Pointer`, `int`, or `ctypes.c_void_p`

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `NDBuffer`.
        """
        cdef DeviceArray wrapper

        if isinstance(pyobj,DeviceArray):
            return pyobj
        else:
            wrapper = DeviceArray.__new__(DeviceArray)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    @staticmethod
    cdef DeviceArray fromPtr(void* ptr):
        cdef DeviceArray wrapper = DeviceArray.__new__(DeviceArray)
        wrapper._set_ptr(ptr)
        return wrapper

cdef class ListOfBytes(Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries of type `bytes` or `~.CStr`.

    Datatype for handling Python `list` and `tuple` objects with entries of type `bytes`
    that need to be converted to a pointer type when passed to the underlying C function.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of `bytes / `~.CStr`:

        A `list` or `tuple` of `bytes` or `~.CStr` objects.
        In this case, this type allocates an array of ``const char*`` pointers wherein it stores the addresses from the `list`/ `tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer.__init__` for more information.

    Note:
        Type checks are performed in the above order.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
        _is_ptr_owner (`bint`, protected):
            If this object is the owner of the allocated buffer. Defaults to `False`.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<ListOfBytes object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False

    @staticmethod
    cdef ListOfBytes fromPtr(void* ptr):
        cdef ListOfBytes wrapper = ListOfBytes.__new__(ListOfBytes)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfBytes`, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        cdef const char* entry_as_cstr = NULL

        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(void*))
            libc.string.memset(self._ptr, 0, len(pyobj)*sizeof(void*))
            for i,entry in enumerate(pyobj):
                if isinstance(entry,bytes):
                    entry_as_cstr = entry # assumes pyobj/pyobj's entries won't be garbage collected
                    # More details: https://cython.readthedocs.io/en/latest/src/tutorial/strings.html
                    (<void**>self._ptr)[i] = <void*>entry_as_cstr
                elif isinstance(entry,CStr):
                    (<void**>self._ptr)[i] = (<CStr>entry)._ptr
                else:
                    raise ValueError("elements of list/tuple input must be of type 'bytes'")
        elif isinstance(pyobj,ListOfBytes):
            self._ptr = (<ListOfBytes>pyobj)._ptr
        else:
            Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfBytes from the given object.

        In case ``pyobj`` is itself an ``ListOfBytes`` instance, this method
        returns it directly. No new ``ListOfBytes`` is created.
        """
        return ListOfBytes.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfBytes fromPyobj(object pyobj):
        """Derives a ListOfBytes from the given object.

        In case ``pyobj`` is itself an ``ListOfBytes`` instance, this method
        returns it directly. No new ``ListOfBytes`` is created.

        Args:
            pyobj (`object`): Must be either `None`, a simple, contiguous buffer according to the buffer protocol,
                or of type `ListOfBytes`, `int`, or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfBytes.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfBytes wrapper

        if isinstance(pyobj,ListOfBytes):
            return pyobj
        else:
            wrapper = ListOfBytes.__new__(ListOfBytes)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfBytes` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfBytes.init_from_pyobj(self,pyobj)

cdef class ListOfPointer(Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries that can be converted to type `~.Pointer`.

    Datatype for handling Python `list` and `tuple` objects with entries that can be converted to type `~.Pointer`.
    Such entries might be of type `None`, `int`, `ctypes.c_void_p`, Python buffer interface implementors, CUDA Array Interface
    implementors, `~.Pointer`, subclasses of Pointer.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of `bytes`:

        A `list` or `tuple` of types that can be converted to `~.Pointer`.
        In this case, this type allocates an array of ``void *`` pointers wherein it stores the addresses obtained from the `list`/`tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer.__init__` for more information.

    Note:
        Type checks are performed in the above order.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
        _is_ptr_owner (`bint`, protected):
            If this object is the owner of the allocated buffer. Defaults to `False`.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<ListOfPointer object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False

    @staticmethod
    cdef ListOfPointer fromPtr(void* ptr):
        cdef ListOfPointer wrapper = ListOfPointer.__new__(ListOfPointer)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfPointer`, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj,ListOfPointer):
            self._ptr = (<ListOfPointer>pyobj)._ptr

        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(void *))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(void *))
            for i,entry in enumerate(pyobj):
                (<void**>self._ptr)[i] = cpython.long.PyLong_AsVoidPtr(int(Pointer.fromPyobj(entry)))
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfPointer from the given object.

        In case ``pyobj`` is itself a ``ListOfPointer`` instance, this method
        returns it directly. No new ``ListOfPointer`` is created.
        """
        return ListOfPointer.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfPointer fromPyobj(object pyobj):
        """Creates a ListOfPointer from the given object.

        In case ``pyobj`` is itself an ``ListOfPointer`` instance, this method
        returns it directly. No new ``ListOfPointer`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to `~.Pointer`, or any other `object` that is accepted as input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if `pyobj` is an instance of ListOfPointer.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfPointer wrapper

        if isinstance(pyobj,ListOfPointer):
            return pyobj
        else:
            wrapper = ListOfPointer.__new__(ListOfPointer)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfPointer` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfPointer.init_from_pyobj(self,pyobj)

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

cdef class ListOfInt(Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries that can be converted to C type ``int``.

    Datatype for handling Python `list` and `tuple` objects with entries that can be converted to C type ``int``.
    Such entries might be of Python type `None`, `int`, or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``int``:

        A `list` or `tuple` of types that can be converted to C type ``int``.
        In this case, this type allocates an array of C ``int`` values wherein it stores the values obtained from the `list`/`tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer` for more information.

    Note:
        Type checks are performed in the above order.

    Note:
        Simple, contiguous numpy and Python 3 array types can be passed
        directly to this routine as they implement the Python buffer protocol.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
        _is_ptr_owner (`bint`, protected):
            If this object is the owner of the allocated buffer. Defaults to `False`.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<ListOfDataInt object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False

    @staticmethod
    cdef ListOfInt fromPtr(void* ptr):
        cdef ListOfInt wrapper = ListOfInt.__new__(ListOfInt)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of ListOfInt, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj,ListOfInt):
            self._ptr = (<ListOfInt>pyobj)._ptr

        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(int))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(int))
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
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfInt from the given object.

        In case ``pyobj`` is itself a ``ListOfInt`` instance, this method
        returns it directly. No new ``ListOfInt`` is created.
        """
        return ListOfInt.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfInt fromPyobj(object pyobj):
        """Derives a ListOfInt from the given object.

        In case ``pyobj`` is itself an ``ListOfInt`` instance, this method
        returns it directly. No new ``ListOfInt`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``int``, or any other `object` that is accepted as input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of `ListOfInt`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfInt wrapper

        if isinstance(pyobj,ListOfInt):
            return pyobj
        else:
            wrapper = ListOfInt.__new__(ListOfInt)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfInt` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfInt.init_from_pyobj(self,pyobj)

cdef class ListOfUnsigned(Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries that can be converted to C type ``unsigned``.

    Datatype for handling Python `list` and `tuple` objects with entries that can be converted to C type ``unsigned``.
    Such entries might be of Python type `None`, `int`, or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``unsigned``:

        A `list` or `tuple` of types that can be converted to C type ``unsigned``.
        In this case, this type allocates an array of C ``unsigned`` values wherein it stores the values obtained from the `list`/`tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer` for more information.

    Note:
        Type checks are performed in the above order.

    Note:
        Simple, contiguous numpy and Python 3 array types can be passed
        directly to this routine as they implement the Python buffer protocol.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
        _is_ptr_owner (`bint`, protected):
            If this object is the owner of the allocated buffer. Defaults to `False`.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<ListOfUnsigned object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False

    @staticmethod
    cdef ListOfUnsigned fromPtr(void* ptr):
        cdef ListOfUnsigned wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfUnsigned`, only the pointer is copied.
            Releasing an acquired `Py_buffer` and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj,ListOfUnsigned):
            self._ptr = (<ListOfUnsigned>pyobj)._ptr

        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned int))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(unsigned int))
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
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfUnsigned from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsigned`` instance, this method
        returns it directly. No new ``ListOfUnsigned`` is created.
        """
        return ListOfUnsigned.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfUnsigned fromPyobj(object pyobj):
        """Creates a ListOfUnsigned from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsigned`` instance, this method
        returns it directly. No new ``ListOfUnsigned`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``unsigned``, or any other `object` that is accepted as input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `ListOfUnsigned`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsigned wrapper

        if isinstance(pyobj,ListOfUnsigned):
            return pyobj
        else:
            wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfUnsigned` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfUnsigned.init_from_pyobj(self,pyobj)

cdef class ListOfUnsignedLong(Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries that can be converted to C type ``unsigned long``.

    Datatype for handling Python `list` and `tuple` objects with entries that can be converted to C type ``unsigned long``.
    Such entries might be of Python type `None`, `int`, or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``unsigned long``:

        A `list` or `tuple` of types that can be converted to C type ``unsigned long``.
        In this case, this type allocates an array of C ``unsigned long`` values wherein it stores the values obtained from the `list`/`tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner` remains unchanged.
        See `~.Pointer` for more information.

    Note:
        Type checks are performed in the above order.

    Note:
        Simple, contiguous numpy and Python 3 array types can be passed
        directly to this routine as they implement the Python buffer protocol.

    C Attributes:
        _ptr (``void *``, protected):
            See `~.Pointer` for more information.
        _py_buffer (`~.Py_buffer`, protected):
            See `~.Pointer` for more information.
        _py_buffer_acquired (`bool`, protected):
            See `~.Pointer` for more information.
        _is_ptr_owner (`bint`, protected):
            If this object is the owner of the allocated buffer. Defaults to `False`.
    """
    # C members declared in declaration part ``types.pxd``

    def __repr__(self):
        return f"<ListOfUnsigned object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False

    @staticmethod
    cdef ListOfUnsignedLong fromPtr(void* ptr):
        cdef ListOfUnsignedLong wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfUnsignedLong`, only the pointer is copied.
            Releasing an acquired `Py_buffer` and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj,ListOfUnsignedLong):
            self._ptr = (<ListOfUnsignedLong>pyobj)._ptr

        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner = True
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned long))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(unsigned long))
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
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfUnsignedLong from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsignedLong`` instance, this method
        returns it directly. No new ``ListOfUnsignedLong`` is created.
        """
        return ListOfUnsignedLong.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfUnsignedLong fromPyobj(object pyobj):
        """Creates a ListOfUnsignedLong from the given object.

        In case ``pyobj`` is itself an ``ListOfUnsignedLong`` instance, this method
        returns it directly. No new ``ListOfUnsignedLong`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``unsigned long``, or any other `object` that is accepted as input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of ListOfUnsignedLong.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsignedLong wrapper

        if isinstance(pyobj,ListOfUnsignedLong):
            return pyobj
        else:
            wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __init__(self,object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfUnsigned` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfUnsignedLong.init_from_pyobj(self,pyobj)

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)
