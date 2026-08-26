# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
cimport cpython.buffer
cimport cpython.bytes
cimport cpython.long
cimport cpython.ref
cimport libc.stdint
cimport libc.stdlib
cimport libc.string

import ctypes
import math

__all__ = [
    # __all__ is important for generating the API documentation in source order
    "Pointer",
    "CStr",
    "DeviceArray",
    "ListOfBytes",
    "ListOfPointer",
    "ListOfInt",
    "ListOfLong",
    "ListOfUnsigned",
    "ListOfUnsignedLong",
    "ListOfInt64",
    "ListOfUInt64",
    "PointerToInt",
    "PointerToLong",
    "PointerToUnsigned",
    "PointerToUnsignedLong",
    "PointerToInt64",
    "PointerToUInt64",
]

cdef class Pointer:
    """Handler for Python arguments that need to be converted to a pointer type

    Datatype for handling Python arguments that need to be converted to a pointer type
    when passed to an underlying C function.

    This type stores a C ``void *`` pointer to the original Python object's data plus
    an additional `Py_buffer` object if the pointer has ben acquired from a Python
    object that implements the
    `Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

    This type can be constructed from input objects that are implementors of the
    CUDA array interface protocol.

    In summary, the type can be initialized from the following Python objects:

    * `None`:

        This will set the ``self._ptr`` attribute to ``NULL``.

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown
        reasons. Therefore, it must be checked for this type first.

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown
        reasons. Therefore, it must be checked for this type first.

    * `object` that implements the Python buffer protocol:

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `object` that implements the CUDA array interface protocol:

        Takes the integer-valued pointer address, i.e. the first entry of the ``data``
        tuple from ``pyobj``'s member ``__cuda_array_interface__``  and writes it to
        ``self._ptr``.

    * `~.Pointer`:

        Copies ``pyobj._ptr`` to ``self._ptr``.
        `~.Py_buffer` object ownership is not transferred!

    * `int`:

        Interprets the integer value as pointer address and writes it to ``self._ptr``.

    * `object` that has `as_c_void_p(self)` method:

        Takes the pointer address ``pyobj.as_c_void_p().value`` and writes it to
        ``self._ptr``.

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
            Releasing an acquired Py_buffer handles is still an obligation of the
            original object.
        """
        cdef dict cuda_array_interface = getattr(
            pyobj, "__cuda_array_interface__", None)

        self._py_buffer_acquired = False
        if pyobj is None:
            self._ptr = NULL
        elif isinstance(pyobj, ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._ptr = cpython.long.PyLong_AsVoidPtr(
                pyobj.value) if pyobj.value is not None else NULL
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &self._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from"
                                   + " Python object")
            self._py_buffer_acquired = True
            self._ptr = self._py_buffer.buf
        elif cuda_array_interface is not None:
            if "data" not in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute"
                                 + " but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            self._ptr = cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif isinstance(pyobj, Pointer):
            self._ptr = (<Pointer>pyobj)._ptr
        elif isinstance(pyobj, int):
            self._ptr = cpython.long.PyLong_AsVoidPtr(pyobj)
        elif hasattr(pyobj, "as_c_void_p"):
            # NOTE: This must stay down here because 'as_c_void_p' is
            #       an interface provided by all the *_util.types types
            #       that should only be used if the type couldn't
            #       be identified as a Python buffer.
            c_void_p_value = pyobj.as_c_void_p().value
            self._ptr = cpython.long.PyLong_AsVoidPtr(
                c_void_p_value) if c_void_p_value is not None else NULL
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
                Must be either `None`, a simple, contiguous buffer according to the
                buffer protocol, or of type `~.Pointer`, `int`, or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `~.Pointer`.
        """
        cdef Pointer wrapper

        if isinstance(pyobj, Pointer):
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

    def __getitem__(self, offset):
        """Returns new `Pointer` whose ``_ptr`` is offsetted by ``offset``

        Args:
            offset (`int`): Offset (in bytes) to add to this instance's pointer.
        """
        if isinstance(offset, int):
            if offset < 0:
                raise ValueError("offset='{offset}' must be non-negative")
            # uintptr_t, not unsigned long: the latter is 32-bit on Windows
            # (LLP64), which would truncate the pointer.
            return Pointer.fromPtr(<void*>(<libc.stdint.uintptr_t>self._ptr
                                   + <libc.stdint.uintptr_t>cpython.long.PyLong_AsUnsignedLongLong(offset)))
        raise NotImplementedError("'__getitem__': not implemented for other"
                                  + " 'offset' types than 'int'")

    def __init__(self, object pyobj = None):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.Pointer` for information
                about accepted types for ``pyobj``.
                Defaults to None.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """

        Pointer.init_from_pyobj(self, pyobj)


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

    Pinning of `str` and `bytes` inputs:
        `str` and `bytes` arguments are interned in the
        `~.CStr._retained_inputs` class dict for the lifetime of the
        program, so the C pointer handed to the backend remains valid
        even after the wrapper instance is collected and even if the
        backend retains the pointer past the call's return (the COMGR
        compile cache reached via `~.bindings.hiprtc` is one such
        backend). Repeated calls with logically equal content reuse the
        same canonical bytes object and therefore the same C pointer,
        which preserves backend caches keyed on pointer identity. The
        intern table only grows with the number of *distinct* string
        contents ever passed.

    Warning:
        The program-lifetime pin above only applies to `str` and `bytes`
        inputs. When the wrapper is constructed from another
        buffer-protocol object (`bytearray`, `numpy.ndarray`,
        `memoryview`, …) the source is pinned only for the wrapper
        instance's lifetime via `Py_buffer` acquisition. If the called
        C library stores the pointer into a library-managed structure
        and the wrapper then goes out of scope, memory errors are
        possible — pass a `bytes`/`str` (which is interned) or hold a
        Python-side reference to the source for as long as the backend
        may dereference it.

    Limitation:
        This class is only designed for handling strings that encode each
        character with 8 bits (ASCII and UTF-8). Smaller or larger symbols are not
        supported.

    The type can be initialized from the following Python objects:

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        If needed, length information must be obtained via ``strlen`` in this case.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for unknown
        reasons. Therefore, it must be checked for this type first.

    * `str`:

        UTF-8 encoded, then interned via
        ``CStr._retained_inputs.setdefault(b, b)``. ``self._ptr`` points
        into the canonical bytes object and is valid for the program's
        lifetime.

    * `bytes`:

        Interned via ``CStr._retained_inputs.setdefault(pyobj, pyobj)``.
        ``self._ptr`` points into the canonical bytes object and is
        valid for the program's lifetime.

    * `object` that implements the Python buffer protocol:

        Note that `bytes` also implements the buffer protocol but is
        intercepted by the dedicated branch above so it takes the
        intern path instead of `Py_buffer` acquisition. This branch
        therefore handles `bytearray`, `numpy.ndarray`, `memoryview`,
        and similar mutable / typed buffers.

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.
        The source is pinned only for the wrapper's lifetime (see
        Warning above).

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

    # Class-level intern table for string inputs that the wrapper
    # hands a `const char*` to the backend. Keys = canonical bytes
    # (the encoded form). Values = the same bytes object —
    # ``setdefault(b, b)`` returns the existing canonical entry if
    # already present, else inserts and returns the new one.
    #
    # Why: many ROCm/HIP backend functions (e.g. hiprtcCompileProgram
    # via COMGR's compile cache) retain pointers to caller-supplied
    # strings well past the call's return — a wrapper instance pin
    # alone (or a transient Py_buffer acquisition) is not enough,
    # because the bytes can still be collected once the wrapper goes
    # out of scope. Interning the canonical bytes in this dict
    # guarantees a program-lifetime address. Repeated calls with the
    # same logical content reuse the same canonical bytes (and thus
    # the same C pointer), so a backend that caches by pointer
    # identity sees the expected hits. The high-water mark is bounded
    # by the number of *unique* string contents ever passed (typical
    # hip-python use: a handful of compile flags + kernel sources;
    # not a leak that grows with call count).
    #
    # Mutating this dict happens with the GIL held (``init_from_pyobj``
    # is called from python-context wrappers), so there is no race
    # despite the cdef class being usable from many threads.
    _retained_inputs = {}

    def __cinit__(self):
        self._is_ptr_owner = False
        self._shape[0] = 0  # must be zero

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
            cannot be initialized. As the length of the C char depends on the location
            of the 0-char, we postpone length calculations to a later stage whenenver
            this information is required.
        """
        cdef CStr wrapper = CStr.__new__(CStr)
        wrapper._ptr = ptr
        wrapper.get_or_determine_len()
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            ``str`` and ``bytes`` inputs are interned in the
            ``CStr._retained_inputs`` class dict for the program's
            lifetime so the C pointer remains valid even after the
            wrapper instance is collected and even if the backend
            retains the pointer. ``str`` inputs are UTF-8 encoded.
            See the ``_retained_inputs`` docstring for the
            program-lifetime contract.

            If ``pyobj`` is an instance of `CStr`, only the pointer and
            length information is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        cdef bytes b
        cdef bytes canonical
        self._py_buffer_acquired = False
        if isinstance(pyobj, CStr):
            self._ptr = (<CStr>pyobj)._ptr
            self._shape[0] = (<CStr>pyobj)._shape[0]
        elif isinstance(pyobj, ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._ptr = (cpython.long.PyLong_AsVoidPtr(pyobj.value)
                         if pyobj.value is not None else NULL)
            self.get_or_determine_len()
        elif isinstance(pyobj, str):
            # NEW path: UTF-8 encode + intern. The wrapper hands the
            # canonical bytes' internal char buffer to the backend;
            # the canonical bytes lives for the program's lifetime via
            # the ``_retained_inputs`` class dict.
            b = (<str>pyobj).encode("utf-8")
            canonical = CStr._retained_inputs.setdefault(b, b)
            self._ptr = <void*><const char*>canonical
            self._shape[0] = len(canonical)
        elif isinstance(pyobj, bytes):
            # NEW path: intern bytes inputs as well. Previously this
            # case fell through to PyObject_CheckBuffer which acquires
            # a Py_buffer — that pins the source only while the wrapper
            # is alive. Interning here pins the canonical bytes for the
            # program's lifetime, which the backend may need.
            canonical = CStr._retained_inputs.setdefault(pyobj, pyobj)
            self._ptr = <void*><const char*>canonical
            self._shape[0] = len(canonical)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            # Other buffer-protocol objects (numpy arrays, bytearray,
            # memoryview, …). Intern doesn't make sense here — the
            # source is typically a mutable buffer with content the
            # caller wants to control. Stay on the Py_buffer path,
            # which pins the source for the wrapper's lifetime via
            # PyBuffer_Release in __dealloc__.
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &self._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from"
                                   + " Python object")
            self._py_buffer_acquired = True
            self._ptr = self._py_buffer.buf
            self._shape[0] = self._py_buffer.len
        else:
            Pointer.init_from_pyobj(self, pyobj)
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
            pyobj (`object`): Must be either `None`, a simple, contiguous buffer
                according to the buffer protocol, or of type `CStr`, `int`, or
                `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of CStr.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef CStr wrapper

        if isinstance(pyobj, CStr):
            return pyobj
        else:
            wrapper = CStr.__new__(CStr)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    cpdef void malloc(self, Py_ssize_t content_len):
        """Allocate a zeroed buffer with room for ``content_len`` chars plus a
        NUL terminator.

        Allocates ``content_len + 1`` bytes and zero-fills them, so the buffer
        is always NUL-terminated: even a C callee that writes all
        ``content_len`` bytes without terminating leaves the appended trailing
        byte as the terminator, keeping ``strlen``-based length reporting
        in-bounds.

        Args:
            content_len (`Py_ssize_t`): Number of content (non-terminator)
                bytes to reserve. The actual allocation is ``content_len + 1``
                to hold the appended NUL terminator.
        Note:
            ``malloc`` appends a NUL terminator byte; pass the buffer capacity
            the C callee will be told about (e.g. HIP's ``len``), not
            ``len + 1``.
        Note:
            Throws `~.RuntimeError` if the data pointer is not NULL as this
            indicates that this instance handles external data.
        Note:
            Sets the _is_ptr_owner flag.
        """
        if self._ptr != NULL:
            raise RuntimeError("Data pointer must be NULL.")
        self._ptr = libc.stdlib.malloc(content_len + 1)
        libc.string.memset(<void*>self._ptr, 0, content_len + 1)
        self._is_ptr_owner = True

    cpdef void free(self):
        """Free dynamically allocated data.

        Note:
            Simply returns if the data pointer is NULL.
        Note:
            Throws `~.RuntimeError` if this instance does not own the data that ought
            to be freed.
        Note:
            Unsets the _is_ptr_owner flag.
        """
        if self._is_ptr_owner is False:
            raise RuntimeError("Attempt to free that is not owned by this instance.")
        if self._ptr == NULL:
            return  # do nothing
        libc.stdlib.free(self._ptr)
        self._is_ptr_owner = False

    def __dealloc__(self):
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            self.free()

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.CStr` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        CStr.init_from_pyobj(self, pyobj)

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
        """Buffer protocol routine for acquiring a view on this `CStr`'s data

        Note:
            `__getbuffer__` and `__releasebuffer__` allow to convert this
            object to bytes.
        Note:
            `buffer.len` and `buffer.shape` are computed on-the-fly (if not set
            already) via `~.CStr.get_or_determine_len(self)`.
        See:
            For details on the Python buffer protocol,
            see https://peps.python.org/pep-3118/.
        """
        buffer.buf = <char *>(self._ptr)
        buffer.format = NULL  # NULL implies bytes, 'B'
        buffer.internal = NULL  # for storing context for dealloc
        buffer.itemsize = 1
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.len = self.get_or_determine_len()  # product(_shape) * itemsize
        buffer.shape = self._shape  # must follow buffer.len
        buffer.strides = NULL  # contiguous
        buffer.suboffsets = NULL  # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        """Buffer protocol routine for releasing a view on this `CStr`'s data.
        """
        pass

    def encode(self, /, encoding="utf-8", errors="strict"):
        """Return a `bytes` object with respect to the encoding.

        See:
            `str.encode`
        """
        return self.decode(encoding=encoding, errors=errors).encode(
            encoding=encoding, errors=errors)

    def decode(self, /, encoding="utf-8", errors="strict"):
        """Return a `str` object with respect to the enconding.

        See:
            `bytes.decode`
        """
        return bytes(self).decode(encoding=encoding, errors=errors)


cdef class NDBuffer(Pointer):
    """Handler for contiguous n-dimensional buffers of various element types

    Datatype for handling contiguous n-dimensional buffers of various element types.
    The buffer can be reshaped via its ``configure`` method.

    Note:
        This buffer does not provide any routines to read or write
        elements of the buffer. Instead, its ``__getitem__`` operator is overloaded
        to return ``NDBuffer`` instances pointing to contiguous subregions
        or single elements of the original buffer. If this buffer is wrapped around
        host data, users can convert it to types that allow access to the underlying
        data such as `bytes`, `bytearray` or numpy array types as this type
        implements the Python buffer protocol.

    This type implements the CUDA array interface protocol. Note, however that it is
    the user's obligation to only pass this type to consumers of the CUDA array
    interface if and only if the underlying data is device data.

    It can be initialized from the following Python objects:

    * `ctypes.c_void_p`:

        Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
        No length information can be obtained in this case.
        Note that `ctypes.c_void_p` seems to be identified as Python buffer for
        unknown reasons. Therefore, it must be checked for this type first.

    * `object` with ``__cuda_array_interface__`` member:
        Takes the integer-valued pointer address, i.e. the first entry of the ``data``
        tuple from ``pyobj``'s member ``__cuda_array_interface__``  and writes it to
        ``self._ptr``. Copies shape and type information.

    * `object` that implements the Python buffer protocol:

        If the object represents a simple contiguous array,
        writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
        sets the `self._py_buffer_acquired` flag to `True`, and
        writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

    Note:
        Type checks are performed in the above order.

    Note:
        Shape and type information and other metadata can be modified or overwritten
        after creation via the `~.configure` member function. Be aware that you might
        need to pass the ``_force=True`` keyword argument --- in particular if your
        instance was created from a type that does not implement the CUDA array
        interface protocol.
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
        return (f"<NDBuffer object, _ptr={int(self)},"
                + f" typestr={self.typestr}, itemsize={self.itemsize},"
                + f" shape={str(self.shape)}, is_read_only={self.is_read_only},"
                + f" stream={self.stream_as_int}>")

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

    cdef int _numpy_typestr_to_bytes(self, str typestr):
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
        elif typestr in ("cdouble", "complex", "complex_", "cfloat", "D", "=D", "<D",
                         ">D"):
            return <int>sizeof(double complex)
        elif typestr in ("clongdouble", "clongfloat", "longcomplex", "G", "=G", "<G",
                         ">G"):
            return <int>sizeof(long double complex)
        return -1

    def __cinit__(self):
        self._ptr = NULL
        self._py_buffer_acquired = False
        self.__view_count = 0
        self._py_buffer_shape = NULL
        self._itemsize = 1
        # NOTE: See: https://docs.scipy.org/doc/numpy-1.13.0/reference/
        #       arrays.interface.html for info on `typestr`.
        self.__dict__ = dict(
            __cuda_array_interface__ = dict(
               shape=(1,),  # by default assume a single byte
               typestr="B",
               data=(None, False),  # 1: data pointer as int (long int), 2: read-only
               strides=None,
               offset=0,
               mask=None,
               version=3,
               # numba
               stream=None,
            )
        )

    cdef _set_ptr(self, void* ptr):
        """Set `self._ptr` C member and 'data' field of CUDA array interface
        """
        cdef tuple old_data = self.__dict__["__cuda_array_interface__"]["data"]
        self._ptr = ptr
        self.__dict__["__cuda_array_interface__"]["data"] = (
            cpython.long.PyLong_FromVoidPtr(ptr), old_data[1]
        )

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
            This method does not automatically map all existing numpy/numba typestr to
            appropriate number of bytes, i.e. `itemsize`. Hence, you need to specify
            ``itemsize`` additionally when dealing with other datatypes than bytes
            (typestr: ``'b'``).
        """
        cdef list supported_keys = ["shape", "typestr", "stream"]
        cdef list extra_keys = ["itemsize", "read_only", "_force"]
        cdef str allowed_keys_str =", ".join(
            [f"'{e}'" for e in supported_keys + extra_keys]
        )
        cdef bint force_new_shape
        cdef tuple shape
        cdef tuple old_shape
        cdef tuple old_data
        cdef bint read_only
        cdef int itemsize = -1
        cdef str typestr = None

        if self.__view_count > 0:
            raise RuntimeError("cannot re-configure this NDBuffer while it is viewed"
                               + " by other objects via the Python buffer protocol")

        for k in kwargs:
            if k not in (supported_keys + extra_keys):
                raise KeyError(f"allowed keyword arguments are: {allowed_keys_str}")

        force_new_shape = kwargs.get("_force", False)
        shape = old_shape = self.__dict__["__cuda_array_interface__"]["shape"]
        if "shape" in kwargs:
            shape = kwargs["shape"]
            if not len(shape):
                raise ValueError("'shape': must have at least one entry")
            for i in shape:
                if not isinstance(i, int):
                    raise TypeError("'shape': entries must be int")
            # self.__dict__["__cuda_array_interface__"]["shape"] = shape
        if "typestr" in kwargs:
            typestr = kwargs["typestr"]
            self.__dict__["__cuda_array_interface__"]["typestr"] = typestr
            itemsize = self._numpy_typestr_to_bytes(typestr)
            if itemsize < 0:
                if typestr not in self.NUMPY_CHAR_CODES:
                    raise ValueError(f"typestr: value '{typestr}' is not a valid"
                                     + " numpy char code. See class attribute"
                                     + " 'NUMPY_CHAR_CODES' for valid expressions.")
                elif "itemsize" not in kwargs:
                    raise ValueError(f"'typestr': value '{typestr}' could not be"
                                     + " mapped to a number of bytes. Please also"
                                     + " specify 'itemsize'.")
        if "itemsize" in kwargs:
            itemsize = kwargs["itemsize"]
            if not isinstance(itemsize, int):
                raise TypeError("'itemsize': must be int")
            if itemsize <= 0:
                raise ValueError("'itemsize': must be positive int")
        #
        if "stream" in kwargs:
            stream = kwargs["stream"]
            if isinstance(stream, int):
                if stream == 0:
                    return ValueError("'stream': value '0' is disallowed as it would be"
                                      + " ambiguous between None and the default"
                                      + " stream, more details: https://numba."
                                      + "readthedocs.io/en/stable/cuda/"
                                      +"cuda_array_interface.html")
                elif stream < 0:
                    return ValueError("'stream': expected positive integer")
                self.__dict__["__cuda_array_interface__"]["stream"] = stream
            else:
                self.__dict__["__cuda_array_interface__"]["stream"] =\
                    int(Pointer.fromPyobj(stream))
        if "read_only" in kwargs:
            read_only = kwargs["read_only"]
            if not isinstance(read_only, bool):
                raise ValueError("'read_only:' expected bool")
            old_data = self.__dict__["__cuda_array_interface__"]["data"]
            self.__dict__["__cuda_array_interface__"]["data"] = (old_data[0], read_only)

        if itemsize > 0 or shape != old_shape:
            old_num_bytes = self._itemsize * math.prod(old_shape)
            if itemsize < 0:
                itemsize = self._itemsize
            new_num_bytes = itemsize * math.prod(shape)
            if old_num_bytes == new_num_bytes or force_new_shape:
                self._itemsize = itemsize
                self.__dict__["__cuda_array_interface__"]["shape"] = shape
            else:
                raise ValueError("new shape would change buffer size information:"
                                 + " {old_num_bytes} B -> {new_num_bytes} B."
                                 + " Specify `_force=True` if this is intended.")

        return self

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of NDBuffer, only the pointer is copied.
            Releasing an acquired Py_buffer handles is still an obligation of the
            original object.
        """
        cdef dict cuda_array_interface =\
            getattr(pyobj, "__cuda_array_interface__", None)

        self._py_buffer_acquired = False
        if pyobj is None:
            self._set_ptr(NULL)
        elif isinstance(pyobj, ctypes.c_void_p):
            # NOTE: must come before the PyObject_CheckBuffer check
            #       as it classifies ctypes.c_void_p as Py buffer for some reason.
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(pyobj.value)
                          if pyobj.value is not None else NULL)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):  # handles 'bytes' too
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &self._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from"
                                   + " Python object")
            self._py_buffer_acquired = True
            self._set_ptr(self._py_buffer.buf)

            shape = [cpython.long.PyLong_FromSsize_t(self._py_buffer.shape[i])
                     for i in range(0, self._py_buffer.ndim)]
            if self._py_buffer.format == NULL:
                # see: https://peps.python.org/pep-3118/#the-py-buffer-struct
                typestr = "B"
            else:
                typestr = cpython.bytes.PyBytes_FromString(
                    self._py_buffer.format).decode("utf-8")
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
        elif cuda_array_interface is not None:
            if "data" not in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__'"
                                 + " attribute but the dict has no 'data' key")
            if cuda_array_interface["strides"] is not None:
                raise RuntimeError("CUDA array interface is not contiguous")
            ptr_as_int = cuda_array_interface["data"][0]
            self._set_ptr(cpython.long.PyLong_AsVoidPtr(ptr_as_int))
            self.configure(cuda_array_interface)
            if isinstance(pyobj, NDBuffer):
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
                Must be either `None`, a simple, contiguous buffer according to the
                buffer protocol, an `object` that implements the CUDA array interface
                protocol, or an instance of `Pointer`, `int`, or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `NDBuffer`.
        """
        cdef NDBuffer wrapper = NDBuffer.__new__(NDBuffer)

        if isinstance(pyobj, NDBuffer):
            return pyobj
        else:
            wrapper = NDBuffer.__new__(NDBuffer)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    cdef tuple _handle_int(self, size_t subscript, size_t shape_dim):
        if subscript < 0:
            raise ValueError(f"subscript='{subscript}' must be non-negative.")
        if subscript >= shape_dim:
            raise ValueError(f"subscript='{subscript}' must be smaller than"
                             + " axis' exclusive upper bound ('{shape_dim}')")
        return (subscript, subscript+1)

    cdef tuple _handle_slice(self, slice subscript, size_t shape_dim):
        cdef size_t start = -1
        cdef size_t stop = -1
        cdef bint extract_full_dim = False

        if subscript.step not in (None, 1):
            raise ValueError(f"subscript's step='{subscript.step}'"
                             + " must be 'None' or '1'.")
        if subscript.stop is not None:
            if subscript.stop <= 0:
                raise ValueError(f"subscript's stop='{subscript.stop}'"
                                 + " must be greater than zero.")
            if subscript.stop > shape_dim:
                raise ValueError(f"subscript's stop='{subscript.stop}' must not be"
                                 +" greater than axis' exclusive upper bound"
                                 + f" ({shape_dim}).")
            stop = subscript.stop
        else:
            stop = shape_dim
        if subscript.start is not None:
            if subscript.start < 0:
                raise ValueError(f"subscript's start='{subscript.start}'"
                                 + " must be non-negative.")
            if subscript.start >= shape_dim:
                raise ValueError(f"subscript's start='{subscript.start}' must be"
                                 + " smaller than axis' exclusive upper bound"
                                 + " ({shape_dim}).")
            start = subscript.start
        else:
            start = 0

        if start >= stop:
            raise ValueError(f"subscript's stop='{subscript.stop}' must be greater"
                             + f" than subscript's start='{subscript.start}'")

        extract_full_dim = (
            start == 0
            and stop == shape_dim
        )
        return (start, stop, extract_full_dim)

    def __getitem__(self, subscript):
        """Returns a contiguous subarray according to the subscript expression.

        Returns a contiguous subarray according to the subscript expression.

        Args:
            subscript (`int`/`slice`/`tuple`):
                Either an integer, a slice, or a tuple of slices and integers.

        Note:
            If the subscript is a single integer, e.g. `[i]`, the subarray
            `[i,:,:,...,:]` is returned. A `KeyError` is raised if the extent of axis 0
            is surpassed. This behavior is identical to that of numpy.

        Raises:
            `TypeError`:
                If the subscript types are not 'int', 'slice' or a 'tuple' thereof.
            `ValueError`:
                If the subscripts do not yield an contiguous subarray. A single array
                element is regarded as contiguous array of size 1.
        """
        cdef size_t stride = 1
        cdef size_t offset = 0
        cdef bint next_slice_yields_contiguous = True
        cdef tuple shape = self.__dict__["__cuda_array_interface__"]["shape"]
        cdef size_t len_shape = len(shape)
        cdef list result_shape = list()  # elements will be appended
        cdef list expanded_subscript = list()
        cdef size_t len_subscript

        if isinstance(subscript, tuple):
            expanded_subscript += subscript[:]
            len_subscript = len(subscript)
        elif isinstance(subscript, (slice, int)):
            expanded_subscript = [subscript]
            len_subscript = 1
        else:
            raise TypeError(
                f"subscript type='{type(subscript)}' is none of:"
                + " 'slice', 'int', 'tuple'"
            )
        # check len and pad ':' slices if the subscript tuple's size is smaller than
        # the array's shape dimensions.
        if len_shape < len_subscript:
            raise IndexError("too many indices specified, maximum number of"
                             + " indices that can be specified is {len_shape}")
        if len_shape > len_subscript:
            expanded_subscript += [slice(None)]*(len_shape-len_subscript)
        for _i, spec in enumerate(reversed(expanded_subscript)):  # row major
            i = len_shape-_i-1
            if isinstance(spec, int):
                (start, stop) = self._handle_int(spec, shape[i])
                next_slice_yields_contiguous = False
            elif isinstance(spec, slice):
                if not next_slice_yields_contiguous:
                    raise ValueError(
                        f"subscript='{expanded_subscript}'"
                        + " yields no contiguous subarray"
                    )
                (start, stop, extract_full_dim) = self._handle_slice(spec, shape[i])
                next_slice_yields_contiguous = extract_full_dim
                # extract_full_dim => start == 0
            else:
                raise TypeError(
                    f"subscript tuple entry type='{type(spec)}' is none of:"
                    + " 'slice', 'int'"
                )
            result_shape.append(stop-start)
            offset += start*stride
            stride *= <size_t>shape[i]
        offset *= self._itemsize  # scale offset with itemsize
        # uintptr_t, not unsigned long: the latter is 32-bit on Windows
        # (LLP64), which would truncate the pointer.
        return NDBuffer.fromPtr(<void*>(<libc.stdint.uintptr_t>self._ptr + offset)).configure(
            _force=True,
            typestr=self.typestr,
            itemsize=self.itemsize,
            shape=tuple(result_shape),
            read_only=self.is_read_only,
            stream=self.stream_as_int,
        )

    def __getattribute__(self, key):
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
            The caller is responsible for keeping a reference to obj until
            ``__releasebuffer__`` is called.
        See:
            For details on the Python buffer protocol,
            see https://peps.python.org/pep-3118/ .
        """
        cdef Py_ssize_t ndim = cpython.long.PyLong_AsSsize_t(len(self.shape))
        cdef Py_ssize_t size = cpython.long.PyLong_AsSsize_t(self.size)
        # reallocate the shape buffer
        if self._py_buffer_shape != NULL:
            libc.stdlib.free(self._py_buffer_shape)
        self._py_buffer_shape = <Py_ssize_t*>libc.stdlib.malloc(ndim*sizeof(Py_ssize_t))
        shape = self.shape
        for i in range(0, ndim):
            self._py_buffer_shape[i] = cpython.long.PyLong_AsSsize_t(shape[i])

        buffer.buf = <char *>(self._ptr)
        self.__dict__["__typestr_bytes"] = self.typestr.encode(
            "utf-8")+b"\x00"  # NUL-terminated, reference must stay alive
        buffer.format = cpython.bytes.PyBytes_AsString(self.__dict__["__typestr_bytes"])
        buffer.internal = NULL  # for storing context for dealloc
        buffer.itemsize = self._itemsize
        buffer.ndim = ndim
        buffer.obj = self
        buffer.readonly = self.is_read_only
        buffer.len = size*self._itemsize
        buffer.shape = self._py_buffer_shape
        buffer.strides = NULL  # contiguous
        buffer.suboffsets = NULL  # for pointer arrays only

        self.__view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        """Buffer protocol routine for releasing a view on this NDBuffer's data.

        Decrements the view count.
        """
        self.__view_count -= 1

    @property
    def typestr(self):
        """The type string (see CUDA array interface specification).
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

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.NDBuffer` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.

        Note:
            Shape and type information and other metadata can be modified or
            overwritten after creation via the `~.configure` member function. Be aware
            that you might need to pass the ``_force=True`` keyword argument --- in
            particular if your instance was created from a type that does not implement
            the CUDA array interface  protocol.
        See:
            `~.configure`
        """
        NDBuffer.init_from_pyobj(self, pyobj)

    def __dealloc__(self):
        if self._py_buffer_shape != NULL:
            libc.stdlib.free(self._py_buffer_shape)
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)

cdef class DeviceArray(NDBuffer):
    """Datatype for handling device buffers.

    Datatype for handling device buffers returned by `~.hipMalloc` and related device
    memory allocation routines.

    This type implements the CUDA array interface protocol.

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
        Takes the integer-valued pointer address, i.e. the first entry of the ``data``
        tuple from `pyobj`'s member ``__cuda_array_interface__``  and writes it to
        ``self._ptr``. Copies shape and type information.

    Note:
        Type checks are performed in the above order.

    Note:
        Shape and type information and other metadata can be modified or overwritten
        after creation via the `~.configure` member function. be aware that you might
        need to pass the ``_force=True`` keyword argument --- in particular if your
        instance was created from a type that does not implement the CUDA array
        interface protocol.
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
    """  # no-cython-lint
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
                Must be either `None`, a simple, contiguous buffer according to the
                buffer protocol, an `object` that implements the
                ``CUDA array interface`` protocol, or an instance of `Pointer`, `int`,
                or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `NDBuffer`.
        """
        cdef DeviceArray wrapper

        if isinstance(pyobj, DeviceArray):
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

def _listof_require_len(Py_ssize_t n):
    """Return ``n`` if the length is known, else raise.

    The ``ListOf*`` element/iteration protocol needs a known element
    count. Instances created via ``allocate(count)`` or from a
    ``list``/``tuple`` track it; instances wrapping a raw pointer
    (``fromPtr`` / an ``int`` / a buffer) do not (``_len == -1``).
    """
    if n < 0:
        raise TypeError(
            "length is unknown for this ListOf* instance; create it via "
            "'allocate(count)' or from a 'list'/'tuple' to index, iterate, "
            "or convert it to a Python list/tuple"
        )
    return n


def _listof_norm_index(object subscript, Py_ssize_t n):
    """Normalize a (possibly negative) integer ``subscript`` against a
    known length ``n``, raising ``IndexError`` when out of range."""
    cdef Py_ssize_t i = subscript
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError("ListOf* index out of range")
    return i


cdef class ListOfBytes(Pointer):
    """Handler for `list` / `tuple` whose entries are `bytes`, `str`, or `~.CStr`.

    Datatype for handling Python `list` and `tuple` objects with entries of type
    `bytes`, `str`, or `~.CStr` that need to be converted to a pointer type
    when passed to the underlying C function. ``str`` entries are UTF-8 encoded
    transparently.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of `bytes`, `str`, or `~.CStr`:

        A `list` or `tuple` of `bytes`, `str`, or `~.CStr` objects.
        In this case, this type allocates an array of ``const char*`` pointers wherein
        it stores the addresses from the `list`/`tuple` entries. ``str`` entries are
        UTF-8 encoded; ``bytes`` and the encoded form of ``str`` entries are interned
        in the ``ListOfBytes._retained_inputs`` class dict for the program's lifetime
        so the C pointers remain valid even if the backend retains them. Furthermore,
        the instance's ``self._is_ptr_owner`` C attribute is set to `True`.

        The interning covers the strings, not the array around them: the
        ``const char**`` array lives exactly as long as this instance, which
        frees it in ``__dealloc__``. Generated wrappers bind the adapter to a
        local variable, so the array is alive for the whole C call. If a C
        function retains the array pointer past its return, the caller must
        keep the adapter alive too — pass ``ListOfBytes([...])`` and hold that
        object, rather than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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

    # Class-level intern table — same role as ``CStr._retained_inputs``
    # (see that docstring for the program-lifetime contract). Keys =
    # canonical bytes (the encoded form of each string element), or
    # the ``CStr`` instance itself for `~.CStr` entries (so the
    # CStr's underlying buffer is also pinned for the program's
    # lifetime). Values = the same object — ``setdefault(k, k)``
    # returns the canonical entry. Bounded by the number of unique
    # element contents (and unique CStr instances) ever passed.
    _retained_inputs = {}

    def __repr__(self):
        return f"<ListOfBytes object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False
        self._len = -1

    @staticmethod
    cdef ListOfBytes fromPtr(void* ptr):
        cdef ListOfBytes wrapper = ListOfBytes.__new__(ListOfBytes)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            String entries (``bytes`` and ``str``) are interned in the
            ``ListOfBytes._retained_inputs`` class dict for the program's
            lifetime; the ``void**`` array stores pointers into the
            canonical bytes objects' internal buffers. ``str`` entries
            are UTF-8 encoded. ``CStr`` entries are pinned by reference
            (the CStr instance itself is added to the intern dict).

            If ``pyobj`` is an instance of `ListOfBytes`, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        cdef bytes b
        cdef bytes canonical

        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(void*))
            libc.string.memset(self._ptr, 0, len(pyobj)*sizeof(void*))
            for i, entry in enumerate(pyobj):
                if isinstance(entry, str):
                    b = (<str>entry).encode("utf-8")
                    canonical = ListOfBytes._retained_inputs.setdefault(b, b)
                    (<void**>self._ptr)[i] = <void*><const char*>canonical
                elif isinstance(entry, bytes):
                    canonical = ListOfBytes._retained_inputs.setdefault(entry, entry)
                    (<void**>self._ptr)[i] = <void*><const char*>canonical
                elif isinstance(entry, CStr):
                    # Pin the CStr instance itself in the intern dict.
                    # Its underlying buffer (whether a malloc'd one,
                    # a Py_buffer-bound one, or an interned-bytes one
                    # via the new CStr str/bytes paths) lives as long
                    # as the CStr instance does.
                    ListOfBytes._retained_inputs.setdefault(entry, entry)
                    (<void**>self._ptr)[i] = (<CStr>entry)._ptr
                else:
                    raise TypeError(
                        "input element must be of type 'bytes', 'str', or 'CStr'"
                    )
        elif isinstance(pyobj, ListOfBytes):
            self._ptr = (<ListOfBytes>pyobj)._ptr
        else:
            Pointer.init_from_pyobj(self, pyobj)

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
            pyobj (`object`): Must be either `None`, a simple, contiguous buffer
            according to the buffer protocol, or of type `ListOfBytes`, `int`, or
            `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of ListOfBytes.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfBytes wrapper

        if isinstance(pyobj, ListOfBytes):
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

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfBytes` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfBytes.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` ``char *`` slots.

        The returned `~.ListOfBytes` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfBytes wrapper = ListOfBytes.__new__(ListOfBytes)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(void*))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(void*))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        cdef const char* s
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        s = <const char*>(<void**>self._ptr)[i]
        return None if s == NULL else <bytes>s

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `bytes`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `bytes`."""
        return tuple(self)

cdef class ListOfPointer(Pointer):
    """Handler for Python `list`/`tuple` whose entries can be converted to `~.Pointer`

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to type `~.Pointer`. Such entries might be of type `None`, `int`,
    `ctypes.c_void_p`, Python buffer interface implementors, CUDA array interface
    implementors, `~.Pointer`, subclasses of Pointer.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to `~.Pointer`:

        A `list` or `tuple` of types that can be converted to `~.Pointer`. In this
        case, this type allocates an array of ``void *`` pointers wherein it stores the
        addresses obtained from the `list`/`tuple` entries. Furthermore, the instance's
        `self._is_ptr_owner` C attribute is set to `True` in this case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfPointer([...])`` and hold that object, rather
        than a bare `list`. Note also that this type stores addresses without
        holding references to the objects they belong to, so those objects must
        be kept alive independently for as long as the C side may dereference
        them.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer.__init__` for more
        information.

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
        self._len = -1

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
        if isinstance(pyobj, ListOfPointer):
            self._ptr = (<ListOfPointer>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(void *))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(void *))
            for i, entry in enumerate(pyobj):
                (<void**>self._ptr)[i] = cpython.long.PyLong_AsVoidPtr(
                    int(Pointer.fromPyobj(entry))
                )
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

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
                to `~.Pointer`, or any other `object` that is accepted as input by
                `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if `pyobj` is an instance of ListOfPointer.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfPointer wrapper

        if isinstance(pyobj, ListOfPointer):
            return pyobj
        else:
            wrapper = ListOfPointer.__new__(ListOfPointer)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfPointer` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfPointer.init_from_pyobj(self, pyobj)

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` ``void *`` slots.

        The returned `~.ListOfPointer` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`. Each element is
        returned as a `~.Pointer`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfPointer wrapper = ListOfPointer.__new__(ListOfPointer)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(void *))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(void *))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return Pointer.fromPtr((<void**>self._ptr)[i])

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `~.Pointer`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `~.Pointer`."""
        return tuple(self)

cdef class ListOfInt(Pointer):
    """Handler for `list` / `tuple` whose entries can be converted to C type ``int``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``int``. Such entries might be of Python type `None`, `int`,
    or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``int``:

        A `list` or `tuple` of types that can be converted to C type ``int``.
        In this case, this type allocates an array of C ``int`` values wherein it
        stores the values obtained from the `list`/`tuple` entries. Furthermore, the
        instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfInt([...])`` and hold that object, rather than
        a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        self._len = -1

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
        if isinstance(pyobj, ListOfInt):
            self._ptr = (<ListOfInt>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(int))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(int))
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<int*>self._ptr)[i] = <int>cpython.long.PyLong_AsLongLong(entry)
                elif isinstance(entry, (
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
                    (<int*>self._ptr)[i] = <int>cpython.long.PyLong_AsLongLong(
                        entry.value
                    )
                else:
                    raise ValueError(f"cannot cast input element '{i}' to C int")
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

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
                to C type ``int``, or any other `object` that is accepted as input by
                `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of `ListOfInt`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfInt wrapper

        if isinstance(pyobj, ListOfInt):
            return pyobj
        else:
            wrapper = ListOfInt.__new__(ListOfInt)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfInt` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfInt.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C ``int`` slots.

        The returned `~.ListOfInt` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfInt wrapper = ListOfInt.__new__(ListOfInt)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(int))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(int))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<int*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef class ListOfLong(Pointer):
    """Handler for `list` / `tuple` whose entries can be converted to C type ``long``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``long``. Such entries might be of Python type `None`, `int`,
    or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``long``:

        A `list` or `tuple` of types that can be converted to C type ``long``.
        In this case, this type allocates an array of C ``long`` values wherein it
        stores the values obtained from the `list`/`tuple` entries. Furthermore, the
        instance's `self._is_ptr_owner` C attribute is set to `True` in this case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfLong([...])`` and hold that object, rather
        than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        return f"<ListOfLong object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False
        self._len = -1

    @staticmethod
    cdef ListOfLong fromPtr(void* ptr):
        cdef ListOfLong wrapper = ListOfLong.__new__(ListOfLong)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of ListOfLong, only the pointer is copied.
            Releasing an acquired Py_buffer and temporary memory are still obligations
            of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj, ListOfLong):
            self._ptr = (<ListOfLong>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(long))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(long))
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<long*>self._ptr)[i] = <long>cpython.long.PyLong_AsLongLong(entry)
                elif isinstance(entry, (
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
                    (<long*>self._ptr)[i] = <long>cpython.long.PyLong_AsLongLong(
                        entry.value
                    )
                else:
                    raise ValueError(f"cannot cast input element '{i}' to C long")
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfLong from the given object.

        In case ``pyobj`` is itself a ``ListOfLong`` instance, this method
        returns it directly. No new ``ListOfLong`` is created.
        """
        return ListOfLong.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfLong fromPyobj(object pyobj):
        """Derives a ListOfLong from the given object.

        In case ``pyobj`` is itself an ``ListOfLong`` instance, this method
        returns it directly. No new ``ListOfLong`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``long``, or any other `object` that is accepted as input by
                `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of `ListOfLong`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfLong wrapper

        if isinstance(pyobj, ListOfLong):
            return pyobj
        else:
            wrapper = ListOfLong.__new__(ListOfLong)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfLong` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfLong.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C ``long`` slots.

        The returned `~.ListOfLong` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfLong wrapper = ListOfLong.__new__(ListOfLong)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(long))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(long))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<long*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef class ListOfUnsigned(Pointer):
    """Handler for `list` / `tuple` whose entries can be converted to C ``unsigned``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``unsigned``. Such entries might be of Python type `None`,
    `int`, or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``unsigned``:

        A `list` or `tuple` of types that can be converted to C type ``unsigned``.
        In this case, this type allocates an array of C ``unsigned`` values wherein it
        stores the values obtained from the `list`/`tuple` entries. Furthermore, the
        instance's ``self._is_ptr_owner`` C attribute is set to `True` in this case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfUnsigned([...])`` and hold that object, rather
        than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        self._len = -1

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
        if isinstance(pyobj, ListOfUnsigned):
            self._ptr = (<ListOfUnsigned>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned int))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(unsigned int))
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<unsigned int*>self._ptr)[i] =\
                        <unsigned int>cpython.long.PyLong_AsUnsignedLongLong(
                            entry)
                elif isinstance(entry, (
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
                    (<unsigned int*>self._ptr)[i] =\
                        <unsigned int>cpython.long.PyLong_AsUnsignedLongLong(
                            entry.value)
                else:
                    raise ValueError(
                        f"input element '{i}' cannot be converted to unsigned int"
                    )
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

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
                to C type ``unsigned``, or any other `object` that is accepted as input
                by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of `ListOfUnsigned`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsigned wrapper

        if isinstance(pyobj, ListOfUnsigned):
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

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfUnsigned` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfUnsigned.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C ``unsigned`` slots.

        The returned `~.ListOfUnsigned` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfUnsigned wrapper = ListOfUnsigned.__new__(ListOfUnsigned)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(unsigned int))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(unsigned int))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<unsigned int*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef class ListOfUnsignedLong(Pointer):
    """Handler for `list`/`tuple` whose entries can be converted to C ``unsigned long``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``unsigned long``. Such entries might be of Python type
    `None`, `int`, or of any `ctypes` integer type.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``unsigned long``:

        A `list` or `tuple` of types that can be converted to C type ``unsigned long``.
        In this case, this type allocates an array of C ``unsigned long`` values
        wherein it stores the values obtained from the `list`/`tuple` entries.
        Furthermore, the instance's `self._is_ptr_owner` C attribute is set to `True`
        in this case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfUnsignedLong([...])`` and hold that object,
        rather than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        self._len = -1

    @staticmethod
    cdef ListOfUnsignedLong fromPtr(void* ptr):
        cdef ListOfUnsignedLong wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfUnsignedLong`, only the pointer is
            copied. Releasing an acquired `Py_buffer` and temporary memory are still
            obligations of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj, ListOfUnsignedLong):
            self._ptr = (<ListOfUnsignedLong>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(len(pyobj)*sizeof(unsigned long))
            libc.string.memset(<void*>self._ptr, 0, len(pyobj)*sizeof(unsigned long))
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<unsigned long*>self._ptr)[i] = \
                        <unsigned long>cpython.long.PyLong_AsUnsignedLongLong(
                            entry
                        )
                elif isinstance(entry, (
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
                    (<unsigned long*>self._ptr)[i] =\
                        <unsigned long>cpython.long.PyLong_AsUnsignedLongLong(
                            entry.value
                        )
                else:
                    raise ValueError(
                        f"element '{i}' of input cannot be converted to"
                        + " C unsigned long type"
                    )
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

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
                to C type ``unsigned long``, or any other `object` that is accepted as
                input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of ListOfUnsignedLong.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUnsignedLong wrapper

        if isinstance(pyobj, ListOfUnsignedLong):
            return pyobj
        else:
            wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfUnsigned` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfUnsignedLong.init_from_pyobj(self, pyobj)

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C
        ``unsigned long`` slots.

        The returned `~.ListOfUnsignedLong` owns the buffer (freed on
        garbage collection) and has a known length, so it is indexable,
        iterable, and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfUnsignedLong wrapper = ListOfUnsignedLong.__new__(ListOfUnsignedLong)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(unsigned long))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(unsigned long))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<unsigned long*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef class ListOfInt64(Pointer):
    """Handler for `list` / `tuple` whose entries can be converted to C ``int64_t``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``int64_t``. Such entries might be of Python type `None`,
    `int`, or of any `ctypes` integer type.

    Unlike `~.ListOfLong`, the element width does not depend on the data model of
    the platform: ``int64_t`` is 64 bits on LP64 (Linux) and LLP64 (Windows)
    alike, while C ``long`` is 64 bits on the former and 32 bits on the latter.
    This is the handler for parameters whose declaration pins the width --
    ``int64_t``, ``ssize_t``, ``ptrdiff_t``, ``intptr_t`` and library typedefs
    aliased to them.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``int64_t``:

        A `list` or `tuple` of types that can be converted to C type ``int64_t``.
        In this case, this type allocates an array of C ``int64_t`` values wherein
        it stores the values obtained from the `list`/`tuple` entries. Furthermore,
        the instance's `self._is_ptr_owner` C attribute is set to `True` in this
        case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfInt64([...])`` and hold that object, rather
        than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        return f"<ListOfInt64 object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False
        self._len = -1

    @staticmethod
    cdef ListOfInt64 fromPtr(void* ptr):
        cdef ListOfInt64 wrapper = ListOfInt64.__new__(ListOfInt64)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfInt64`, only the pointer is copied.
            Releasing an acquired `Py_buffer` and temporary memory are still
            obligations of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj, ListOfInt64):
            self._ptr = (<ListOfInt64>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(
                len(pyobj)*sizeof(libc.stdint.int64_t)
            )
            libc.string.memset(
                <void*>self._ptr, 0, len(pyobj)*sizeof(libc.stdint.int64_t)
            )
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<libc.stdint.int64_t*>self._ptr)[i] = \
                        cpython.long.PyLong_AsLongLong(entry)
                elif isinstance(entry, (
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
                    (<libc.stdint.int64_t*>self._ptr)[i] = \
                        cpython.long.PyLong_AsLongLong(entry.value)
                else:
                    raise ValueError(
                        f"cannot cast input element '{i}' to C int64_t"
                    )
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfInt64 from the given object.

        In case ``pyobj`` is itself a ``ListOfInt64`` instance, this method
        returns it directly. No new ``ListOfInt64`` is created.
        """
        return ListOfInt64.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfInt64 fromPyobj(object pyobj):
        """Derives a ListOfInt64 from the given object.

        In case ``pyobj`` is itself an ``ListOfInt64`` instance, this method
        returns it directly. No new ``ListOfInt64`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``int64_t``, or any other `object` that is accepted as input
                by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of `ListOfInt64`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfInt64 wrapper

        if isinstance(pyobj, ListOfInt64):
            return pyobj
        else:
            wrapper = ListOfInt64.__new__(ListOfInt64)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfInt64` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfInt64.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C ``int64_t`` slots.

        The returned `~.ListOfInt64` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfInt64 wrapper = ListOfInt64.__new__(ListOfInt64)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(
                count*sizeof(libc.stdint.int64_t)
            )
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(
                wrapper._ptr, 0, count*sizeof(libc.stdint.int64_t)
            )
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<libc.stdint.int64_t*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef class ListOfUInt64(Pointer):
    """Handler for `list` / `tuple` whose entries can be converted to C ``uint64_t``

    Datatype for handling Python `list` and `tuple` objects with entries that can be
    converted to C type ``uint64_t``. Such entries might be of Python type `None`,
    `int`, or of any `ctypes` integer type.

    Unlike `~.ListOfUnsignedLong`, the element width does not depend on the data
    model of the platform: ``uint64_t`` is 64 bits on LP64 (Linux) and LLP64
    (Windows) alike, while C ``unsigned long`` is 64 bits on the former and 32
    bits on the latter. This is the handler for parameters whose declaration pins
    the width -- ``uint64_t``, ``size_t``, ``uintptr_t`` and library typedefs
    aliased to them.

    The type can be initialized from the following Python objects:

    * `list` / `tuple` of types that can be converted to C type ``uint64_t``:

        A `list` or `tuple` of types that can be converted to C type ``uint64_t``.
        In this case, this type allocates an array of C ``uint64_t`` values wherein
        it stores the values obtained from the `list`/`tuple` entries. Furthermore,
        the instance's `self._is_ptr_owner` C attribute is set to `True` in this
        case.

        The array lives exactly as long as this instance, which frees it in
        ``__dealloc__``. Generated wrappers bind the adapter to a local
        variable, so the array is alive for the whole C call. If a C function
        retains the pointer past its return, the caller must keep the adapter
        alive too — pass ``ListOfUInt64([...])`` and hold that object, rather
        than a bare `list`.

    * `object` that is accepted as input by `~.Pointer.__init__`:

        In this case, init code from `~.Pointer` is used and the C attribute
        ``self._is_ptr_owner`` remains unchanged. See `~.Pointer` for more
        information.

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
        return f"<ListOfUInt64 object, _ptr={int(self)}>"

    def __cinit__(self):
        self._is_ptr_owner = False
        self._len = -1

    @staticmethod
    cdef ListOfUInt64 fromPtr(void* ptr):
        cdef ListOfUInt64 wrapper = ListOfUInt64.__new__(ListOfUInt64)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        """
        Note:
            If ``pyobj`` is an instance of `ListOfUInt64`, only the pointer is
            copied. Releasing an acquired `Py_buffer` and temporary memory are still
            obligations of the original object.
        """
        self._py_buffer_acquired = False
        self._is_ptr_owner = False
        if isinstance(pyobj, ListOfUInt64):
            self._ptr = (<ListOfUInt64>pyobj)._ptr

        elif isinstance(pyobj, (tuple, list)):
            self._is_ptr_owner = True
            self._len = len(pyobj)
            self._ptr = libc.stdlib.malloc(
                len(pyobj)*sizeof(libc.stdint.uint64_t)
            )
            libc.string.memset(
                <void*>self._ptr, 0, len(pyobj)*sizeof(libc.stdint.uint64_t)
            )
            for i, entry in enumerate(pyobj):
                if isinstance(entry, int):
                    (<libc.stdint.uint64_t*>self._ptr)[i] = \
                        cpython.long.PyLong_AsUnsignedLongLong(entry)
                elif isinstance(entry, (
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
                    (<libc.stdint.uint64_t*>self._ptr)[i] = \
                        cpython.long.PyLong_AsUnsignedLongLong(entry.value)
                else:
                    raise ValueError(
                        f"cannot cast input element '{i}' to C uint64_t"
                    )
        else:
            self._is_ptr_owner = False
            Pointer.init_from_pyobj(self, pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a ListOfUInt64 from the given object.

        In case ``pyobj`` is itself a ``ListOfUInt64`` instance, this method
        returns it directly. No new ``ListOfUInt64`` is created.
        """
        return ListOfUInt64.fromPyobj(pyobj)

    @staticmethod
    cdef ListOfUInt64 fromPyobj(object pyobj):
        """Derives a ListOfUInt64 from the given object.

        In case ``pyobj`` is itself an ``ListOfUInt64`` instance, this method
        returns it directly. No new ``ListOfUInt64`` is created.

        Args:
            pyobj (`object`):
                Must be either a `list` or `tuple` of objects that can be converted
                to C type ``uint64_t``, or any other `object` that is accepted as
                input by `~.Pointer.__init__`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of `ListOfUInt64`.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef ListOfUInt64 wrapper

        if isinstance(pyobj, ListOfUInt64):
            return pyobj
        else:
            wrapper = ListOfUInt64.__new__(ListOfUInt64)
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        if self._py_buffer_acquired:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        if self._is_ptr_owner:
            libc.stdlib.free(self._ptr)

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.ListOfUInt64` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
        """
        ListOfUInt64.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count):
        """Allocate an owned, zero-initialized array of ``count`` C ``uint64_t`` slots.

        The returned `~.ListOfUInt64` owns the buffer (freed on garbage
        collection) and has a known length, so it is indexable, iterable,
        and convertible via `~.to_list`/`~.to_tuple`.
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef ListOfUInt64 wrapper = ListOfUInt64.__new__(ListOfUInt64)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(
                count*sizeof(libc.stdint.uint64_t)
            )
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(
                wrapper._ptr, 0, count*sizeof(libc.stdint.uint64_t)
            )
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    def __len__(self):
        return _listof_require_len(self._len)

    def __getitem__(self, subscript):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        if isinstance(subscript, slice):
            return [self[j] for j in range(*subscript.indices(n))]
        i = _listof_norm_index(subscript, n)
        return (<libc.stdint.uint64_t*>self._ptr)[i]

    def __iter__(self):
        cdef Py_ssize_t n = _listof_require_len(self._len)
        cdef Py_ssize_t i
        for i in range(n):
            yield self[i]

    def to_list(self):
        """Return the elements as a Python `list` of `int`."""
        return list(self)

    def to_tuple(self):
        """Return the elements as a Python `tuple` of `int`."""
        return tuple(self)

cdef int _pointerto_require_scalar(object pyobj) except -1:
    """Reject multi-element sequence initializers for a ``PointerTo*``.

    A ``PointerTo*`` wraps a pointer to a *single* scalar, so when the
    initializer is a ``list`` / ``tuple`` it must have exactly one element
    (mirroring the length-1 nature of the wrapper). Non-sequence inputs (a
    raw address ``int`` / buffer / another wrapper) are unaffected and keep
    the inherited `~.Pointer` semantics.
    """
    if isinstance(pyobj, (tuple, list)) and len(pyobj) != 1:
        raise ValueError(
            "a 'PointerTo*' wraps a single scalar; a 'list'/'tuple' "
            f"initializer must have exactly one element (got {len(pyobj)})"
        )
    return 0

cdef class PointerToInt(ListOfInt):
    """Handler for a rank-0 pointer to a single C ``int``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfInt`): it wraps a ``T *`` that points at a
    *single* value rather than a sized buffer. Use it for a caller-allocated
    scalar pointer argument (an ``IN`` / ``INOUT`` / caller-allocated ``OUT``
    ``int *`` parameter): allocate one slot, pass it to the C call, then read
    the result back through `~.value` (or ``self[0]``).

    Accepts the same inputs as `~.ListOfInt` (a `list` / `tuple`, another
    ``ListOfInt`` / ``PointerToInt``, or any object accepted by `~.Pointer`),
    except that a `list` / `tuple` initializer must have exactly one element
    (a ``PointerTo*`` points at a single scalar); `~.allocate` defaults to a
    single slot.
    """
    def __repr__(self):
        return f"<PointerToInt object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToInt fromPtr(void* ptr):
        cdef PointerToInt wrapper = PointerToInt.__new__(PointerToInt)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToInt from the given object.

        In case ``pyobj`` is itself a ``PointerToInt`` instance, this method
        returns it directly. No new ``PointerToInt`` is created.
        """
        return PointerToInt.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToInt fromPyobj(object pyobj):
        """Derives a PointerToInt from the given object.

        In case ``pyobj`` is itself a ``PointerToInt`` instance, this method
        returns it directly. No new ``PointerToInt`` is created.
        """
        cdef PointerToInt wrapper
        if isinstance(pyobj, PointerToInt):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToInt.__new__(PointerToInt)
            ListOfInt.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToInt` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfInt.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``int`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToInt wrapper = PointerToInt.__new__(PointerToInt)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(int))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(int))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``int`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToInt")
        return (<int*>self._ptr)[0]

    @value.setter
    def value(self, int v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToInt")
        (<int*>self._ptr)[0] = v

cdef class PointerToLong(ListOfLong):
    """Handler for a rank-0 pointer to a single C ``long``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfLong`): it wraps a ``T *`` that points at a
    *single* value rather than a sized buffer. Use it for a caller-allocated
    scalar pointer argument (an ``IN`` / ``INOUT`` / caller-allocated ``OUT``
    ``long *`` parameter, e.g. hipFILE's async ``bytes_read_p`` /
    ``bytes_written_p``): allocate one slot, pass it to the C call, then read
    the result back through `~.value` (or ``self[0]``).

    Accepts the same inputs as `~.ListOfLong` (a `list` / `tuple`, another
    ``ListOfLong`` / ``PointerToLong``, or any object accepted by `~.Pointer`),
    except that a `list` / `tuple` initializer must have exactly one element
    (a ``PointerTo*`` points at a single scalar); `~.allocate` defaults to a
    single slot.
    """
    def __repr__(self):
        return f"<PointerToLong object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToLong fromPtr(void* ptr):
        cdef PointerToLong wrapper = PointerToLong.__new__(PointerToLong)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToLong from the given object.

        In case ``pyobj`` is itself a ``PointerToLong`` instance, this method
        returns it directly. No new ``PointerToLong`` is created.
        """
        return PointerToLong.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToLong fromPyobj(object pyobj):
        """Derives a PointerToLong from the given object.

        In case ``pyobj`` is itself a ``PointerToLong`` instance, this method
        returns it directly. No new ``PointerToLong`` is created.
        """
        cdef PointerToLong wrapper
        if isinstance(pyobj, PointerToLong):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToLong.__new__(PointerToLong)
            ListOfLong.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToLong` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfLong.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``long`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToLong wrapper = PointerToLong.__new__(PointerToLong)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(long))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(long))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``long`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToLong")
        return (<long*>self._ptr)[0]

    @value.setter
    def value(self, long v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToLong")
        (<long*>self._ptr)[0] = v

cdef class PointerToUnsigned(ListOfUnsigned):
    """Handler for a rank-0 pointer to a single C ``unsigned int``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfUnsigned`): it wraps a ``T *`` that points at
    a *single* value rather than a sized buffer. Use it for a caller-allocated
    scalar pointer argument (an ``IN`` / ``INOUT`` / caller-allocated ``OUT``
    ``unsigned int *`` parameter): allocate one slot, pass it to the C call,
    then read the result back through `~.value` (or ``self[0]``).

    Accepts the same inputs as `~.ListOfUnsigned` (a `list` / `tuple`, another
    ``ListOfUnsigned`` / ``PointerToUnsigned``, or any object accepted by
    `~.Pointer`), except that a `list` / `tuple` initializer must have exactly
    one element (a ``PointerTo*`` points at a single scalar); `~.allocate`
    defaults to a single slot.
    """
    def __repr__(self):
        return f"<PointerToUnsigned object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToUnsigned fromPtr(void* ptr):
        cdef PointerToUnsigned wrapper = PointerToUnsigned.__new__(PointerToUnsigned)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToUnsigned from the given object.

        In case ``pyobj`` is itself a ``PointerToUnsigned`` instance, this method
        returns it directly. No new ``PointerToUnsigned`` is created.
        """
        return PointerToUnsigned.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToUnsigned fromPyobj(object pyobj):
        """Derives a PointerToUnsigned from the given object.

        In case ``pyobj`` is itself a ``PointerToUnsigned`` instance, this method
        returns it directly. No new ``PointerToUnsigned`` is created.
        """
        cdef PointerToUnsigned wrapper
        if isinstance(pyobj, PointerToUnsigned):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToUnsigned.__new__(PointerToUnsigned)
            ListOfUnsigned.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToUnsigned` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfUnsigned.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``unsigned int`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToUnsigned wrapper = PointerToUnsigned.__new__(PointerToUnsigned)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(unsigned int))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(unsigned int))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``unsigned int`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToUnsigned")
        return (<unsigned int*>self._ptr)[0]

    @value.setter
    def value(self, unsigned int v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToUnsigned")
        (<unsigned int*>self._ptr)[0] = v

cdef class PointerToUnsignedLong(ListOfUnsignedLong):
    """Handler for a rank-0 pointer to a single C ``unsigned long``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfUnsignedLong`): it wraps a ``T *`` that points
    at a *single* value rather than a sized buffer. Use it for a caller-
    allocated scalar pointer argument (an ``IN`` / ``INOUT`` / caller-allocated
    ``OUT`` ``unsigned long *`` / ``size_t *`` parameter): allocate one slot,
    pass it to the C call, then read the result back through `~.value` (or
    ``self[0]``).

    Accepts the same inputs as `~.ListOfUnsignedLong` (a `list` / `tuple`,
    another ``ListOfUnsignedLong`` / ``PointerToUnsignedLong``, or any object
    accepted by `~.Pointer`), except that a `list` / `tuple` initializer must
    have exactly one element (a ``PointerTo*`` points at a single scalar);
    `~.allocate` defaults to a single slot.
    """
    def __repr__(self):
        return f"<PointerToUnsignedLong object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToUnsignedLong fromPtr(void* ptr):
        cdef PointerToUnsignedLong wrapper = PointerToUnsignedLong.__new__(PointerToUnsignedLong)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToUnsignedLong from the given object.

        In case ``pyobj`` is itself a ``PointerToUnsignedLong`` instance, this
        method returns it directly. No new ``PointerToUnsignedLong`` is created.
        """
        return PointerToUnsignedLong.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToUnsignedLong fromPyobj(object pyobj):
        """Derives a PointerToUnsignedLong from the given object.

        In case ``pyobj`` is itself a ``PointerToUnsignedLong`` instance, this
        method returns it directly. No new ``PointerToUnsignedLong`` is created.
        """
        cdef PointerToUnsignedLong wrapper
        if isinstance(pyobj, PointerToUnsignedLong):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToUnsignedLong.__new__(PointerToUnsignedLong)
            ListOfUnsignedLong.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToUnsignedLong` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfUnsignedLong.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``unsigned long`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToUnsignedLong wrapper = PointerToUnsignedLong.__new__(PointerToUnsignedLong)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(count*sizeof(unsigned long))
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(wrapper._ptr, 0, count*sizeof(unsigned long))
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``unsigned long`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToUnsignedLong")
        return (<unsigned long*>self._ptr)[0]

    @value.setter
    def value(self, unsigned long v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToUnsignedLong")
        (<unsigned long*>self._ptr)[0] = v

cdef class PointerToInt64(ListOfInt64):
    """Handler for a rank-0 pointer to a single C ``int64_t``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfInt64`): it wraps a ``T *`` that points at a
    *single* value rather than a sized buffer. Use it for a caller-allocated
    scalar pointer argument whose declaration pins a signed 64-bit width (an
    ``IN`` / ``INOUT`` / caller-allocated ``OUT`` ``int64_t *``, ``ssize_t *``
    or aliased ``hoff_t *`` parameter, e.g. hipFILE's async ``bytes_read_p`` /
    ``bytes_written_p``): allocate one slot, pass it to the C call, then read
    the result back through `~.value` (or ``self[0]``).

    Accepts the same inputs as `~.ListOfInt64` (a `list` / `tuple`, another
    ``ListOfInt64`` / ``PointerToInt64``, or any object accepted by
    `~.Pointer`), except that a `list` / `tuple` initializer must have exactly
    one element (a ``PointerTo*`` points at a single scalar); `~.allocate`
    defaults to a single slot.
    """
    def __repr__(self):
        return f"<PointerToInt64 object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToInt64 fromPtr(void* ptr):
        cdef PointerToInt64 wrapper = PointerToInt64.__new__(PointerToInt64)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToInt64 from the given object.

        In case ``pyobj`` is itself a ``PointerToInt64`` instance, this method
        returns it directly. No new ``PointerToInt64`` is created.
        """
        return PointerToInt64.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToInt64 fromPyobj(object pyobj):
        """Derives a PointerToInt64 from the given object.

        In case ``pyobj`` is itself a ``PointerToInt64`` instance, this method
        returns it directly. No new ``PointerToInt64`` is created.
        """
        cdef PointerToInt64 wrapper
        if isinstance(pyobj, PointerToInt64):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToInt64.__new__(PointerToInt64)
            ListOfInt64.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToInt64` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfInt64.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``int64_t`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToInt64 wrapper = PointerToInt64.__new__(PointerToInt64)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(
                count*sizeof(libc.stdint.int64_t)
            )
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(
                wrapper._ptr, 0, count*sizeof(libc.stdint.int64_t)
            )
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``int64_t`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToInt64")
        return (<libc.stdint.int64_t*>self._ptr)[0]

    @value.setter
    def value(self, libc.stdint.int64_t v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToInt64")
        (<libc.stdint.int64_t*>self._ptr)[0] = v

cdef class PointerToUInt64(ListOfUInt64):
    """Handler for a rank-0 pointer to a single C ``uint64_t``.

    A ``PointerTo*`` is a length-1 specialization of the matching
    ``ListOf*`` (here `~.ListOfUInt64`): it wraps a ``T *`` that points at a
    *single* value rather than a sized buffer. Use it for a caller-allocated
    scalar pointer argument whose declaration pins an unsigned 64-bit width (an
    ``IN`` / ``INOUT`` / caller-allocated ``OUT`` ``uint64_t *`` or ``size_t *``
    parameter): allocate one slot, pass it to the C call, then read the result
    back through `~.value` (or ``self[0]``).

    Accepts the same inputs as `~.ListOfUInt64` (a `list` / `tuple`, another
    ``ListOfUInt64`` / ``PointerToUInt64``, or any object accepted by
    `~.Pointer`), except that a `list` / `tuple` initializer must have exactly
    one element (a ``PointerTo*`` points at a single scalar); `~.allocate`
    defaults to a single slot.
    """
    def __repr__(self):
        return f"<PointerToUInt64 object, _ptr={int(self)}>"

    @staticmethod
    cdef PointerToUInt64 fromPtr(void* ptr):
        cdef PointerToUInt64 wrapper = PointerToUInt64.__new__(PointerToUInt64)
        wrapper._ptr = ptr
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        """Creates a PointerToUInt64 from the given object.

        In case ``pyobj`` is itself a ``PointerToUInt64`` instance, this method
        returns it directly. No new ``PointerToUInt64`` is created.
        """
        return PointerToUInt64.fromPyobj(pyobj)

    @staticmethod
    cdef PointerToUInt64 fromPyobj(object pyobj):
        """Derives a PointerToUInt64 from the given object.

        In case ``pyobj`` is itself a ``PointerToUInt64`` instance, this method
        returns it directly. No new ``PointerToUInt64`` is created.
        """
        cdef PointerToUInt64 wrapper
        if isinstance(pyobj, PointerToUInt64):
            return pyobj
        else:
            _pointerto_require_scalar(pyobj)
            wrapper = PointerToUInt64.__new__(PointerToUInt64)
            ListOfUInt64.init_from_pyobj(wrapper, pyobj)
            return wrapper

    def __init__(self, object pyobj):
        """Constructor.

        Args:
            pyobj (`object`):
                See the class description `~.PointerToUInt64` for information
                about accepted types for ``pyobj``.

        Raises:
            `TypeError`: If the input object ``pyobj`` is not of the right type.
            `ValueError`: If ``pyobj`` is a `list` / `tuple` with more than one element.
        """
        _pointerto_require_scalar(pyobj)
        ListOfUInt64.init_from_pyobj(self, pyobj)

    @staticmethod
    def allocate(Py_ssize_t count = 1):
        """Allocate an owned, zero-initialized array of ``count`` C ``uint64_t`` slots.

        Defaults to a single slot (the rank-0 ``PointerTo*`` use case).
        """
        if count < 0:
            raise ValueError("'count' must be non-negative")
        cdef PointerToUInt64 wrapper = PointerToUInt64.__new__(PointerToUInt64)
        if count > 0:
            wrapper._ptr = libc.stdlib.malloc(
                count*sizeof(libc.stdint.uint64_t)
            )
            if wrapper._ptr == NULL:
                raise MemoryError()
            libc.string.memset(
                wrapper._ptr, 0, count*sizeof(libc.stdint.uint64_t)
            )
        wrapper._is_ptr_owner = True
        wrapper._len = count
        return wrapper

    @property
    def value(self):
        """The pointed-to ``uint64_t`` value (dereferences the first slot)."""
        if self._ptr == NULL:
            raise ValueError("cannot dereference a NULL PointerToUInt64")
        return (<libc.stdint.uint64_t*>self._ptr)[0]

    @value.setter
    def value(self, libc.stdint.uint64_t v):
        if self._ptr == NULL:
            raise ValueError("cannot write through a NULL PointerToUInt64")
        (<libc.stdint.uint64_t*>self._ptr)[0] = v


def _clear_retained_inputs():
    """Clear the program-lifetime intern dicts on `~.CStr` and
    `~.ListOfBytes`.

    NOT public API. Intended only for tests that need a clean
    baseline for the dedup-bound assertions, and for embedding
    scenarios that intentionally release ahead of process exit.

    Calling this while any backend may still hold a pointer into
    the canonical bytes is undefined behaviour — the backend will
    dereference freed memory the next time it consults the
    pointer.
    """
    CStr._retained_inputs.clear()
    ListOfBytes._retained_inputs.clear()
