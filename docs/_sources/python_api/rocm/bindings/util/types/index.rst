rocm.bindings.util.types
========================

.. py:module:: rocm.bindings.util.types


Classes
-------

.. autoapisummary::

   rocm.bindings.util.types.CStr
   rocm.bindings.util.types.DeviceArray
   rocm.bindings.util.types.ListOfBytes
   rocm.bindings.util.types.ListOfInt
   rocm.bindings.util.types.ListOfInt64
   rocm.bindings.util.types.ListOfLong
   rocm.bindings.util.types.ListOfPointer
   rocm.bindings.util.types.ListOfUInt64
   rocm.bindings.util.types.ListOfUnsigned
   rocm.bindings.util.types.ListOfUnsignedLong
   rocm.bindings.util.types.NDBuffer
   rocm.bindings.util.types.Pointer
   rocm.bindings.util.types.PointerToInt
   rocm.bindings.util.types.PointerToInt64
   rocm.bindings.util.types.PointerToLong
   rocm.bindings.util.types.PointerToUInt64
   rocm.bindings.util.types.PointerToUnsigned
   rocm.bindings.util.types.PointerToUnsignedLong


Module Contents
---------------

.. py:class:: CStr(pyobj)

   Bases: :py:obj:`Pointer`


   CStr(pyobj)

   Datatype for handling C strings (`char *` and related).

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


   .. py:method:: decode(encoding=..., errors=...) -> Any

      CStr.decode(self, /, encoding='utf-8', errors='strict')

      Return a `str` object with respect to the enconding.

      See:
          `bytes.decode`



   .. py:method:: encode(encoding=..., errors=...) -> Any

      CStr.encode(self, /, encoding='utf-8', errors='strict')

      Return a `bytes` object with respect to the encoding.

      See:
          `str.encode`



   .. py:method:: free() -> void

      CStr.free(self) -> void

      Free dynamically allocated data.

      Note:
          Simply returns if the data pointer is NULL.
      Note:
          Throws `~.RuntimeError` if this instance does not own the data that ought
          to be freed.
      Note:
          Unsets the _is_ptr_owner flag.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      CStr.fromObj(pyobj)

      Creates a CStr from the given object.

      In case ``pyobj`` is itself a ``CStr`` instance, this method
      returns it directly. No new ``CStr`` is created.



   .. py:method:: malloc(Py_ssize_tcontent_len) -> void

      CStr.malloc(self, Py_ssize_t content_len) -> void

      Allocate a zeroed buffer with room for ``content_len`` chars plus a
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



.. py:class:: DeviceArray(*args, **kwargs)

   Bases: :py:obj:`NDBuffer`


   Datatype for handling device buffers.

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
       _cuda_array_interface (`dict`, protected):
           The CUDA array interface metadata, handed out as a copy by
           `~.NDBuffer.__cuda_array_interface__`.
       _pybuffer_obj (`object`, protected):
           Keeps the exporter of a wrapped `Py_buffer` alive.
       _typestr_bytes (`bytes`, protected):
           NUL-terminated format string handed to consumers of this
           Python buffer. Must stay alive as long as any view exists.


   .. py:method:: DeviceArray(pyobj) -> Any
      :staticmethod:


      DeviceArray.DeviceArray(pyobj)

      Creates a NDBuffer from the given object.

      In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
      returns it directly. No new ``NDBuffer`` is created.



.. py:class:: ListOfBytes(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfBytes(pyobj)

   Handler for `list` / `tuple` whose entries are `bytes`, `str`, or `~.CStr`.

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfBytes.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` ``char *`` slots.

      The returned `~.ListOfBytes` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfBytes.fromObj(pyobj)

      Creates a ListOfBytes from the given object.

      In case ``pyobj`` is itself an ``ListOfBytes`` instance, this method
      returns it directly. No new ``ListOfBytes`` is created.



   .. py:method:: to_list() -> Any

      ListOfBytes.to_list(self)

      Return the elements as a Python `list` of `bytes`.



   .. py:method:: to_tuple() -> Any

      ListOfBytes.to_tuple(self)

      Return the elements as a Python `tuple` of `bytes`.



.. py:class:: ListOfInt(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfInt(pyobj)

   Handler for `list` / `tuple` whose entries can be converted to C type ``int``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfInt.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C ``int`` slots.

      The returned `~.ListOfInt` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfInt.fromObj(pyobj)

      Creates a ListOfInt from the given object.

      In case ``pyobj`` is itself a ``ListOfInt`` instance, this method
      returns it directly. No new ``ListOfInt`` is created.



   .. py:method:: to_list() -> Any

      ListOfInt.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfInt.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: ListOfInt64(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfInt64(pyobj)

   Handler for `list` / `tuple` whose entries can be converted to C ``int64_t``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfInt64.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C ``int64_t`` slots.

      The returned `~.ListOfInt64` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfInt64.fromObj(pyobj)

      Creates a ListOfInt64 from the given object.

      In case ``pyobj`` is itself a ``ListOfInt64`` instance, this method
      returns it directly. No new ``ListOfInt64`` is created.



   .. py:method:: to_list() -> Any

      ListOfInt64.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfInt64.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: ListOfLong(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfLong(pyobj)

   Handler for `list` / `tuple` whose entries can be converted to C type ``long``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfLong.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C ``long`` slots.

      The returned `~.ListOfLong` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfLong.fromObj(pyobj)

      Creates a ListOfLong from the given object.

      In case ``pyobj`` is itself a ``ListOfLong`` instance, this method
      returns it directly. No new ``ListOfLong`` is created.



   .. py:method:: to_list() -> Any

      ListOfLong.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfLong.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: ListOfPointer(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfPointer(pyobj)

   Handler for Python `list`/`tuple` whose entries can be converted to `~.Pointer`

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfPointer.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` ``void *`` slots.

      The returned `~.ListOfPointer` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`. Each element is
      returned as a `~.Pointer`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfPointer.fromObj(pyobj)

      Creates a ListOfPointer from the given object.

      In case ``pyobj`` is itself a ``ListOfPointer`` instance, this method
      returns it directly. No new ``ListOfPointer`` is created.



   .. py:method:: to_list() -> Any

      ListOfPointer.to_list(self)

      Return the elements as a Python `list` of `~.Pointer`.



   .. py:method:: to_tuple() -> Any

      ListOfPointer.to_tuple(self)

      Return the elements as a Python `tuple` of `~.Pointer`.



.. py:class:: ListOfUInt64(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfUInt64(pyobj)

   Handler for `list` / `tuple` whose entries can be converted to C ``uint64_t``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfUInt64.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C ``uint64_t`` slots.

      The returned `~.ListOfUInt64` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfUInt64.fromObj(pyobj)

      Creates a ListOfUInt64 from the given object.

      In case ``pyobj`` is itself a ``ListOfUInt64`` instance, this method
      returns it directly. No new ``ListOfUInt64`` is created.



   .. py:method:: to_list() -> Any

      ListOfUInt64.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfUInt64.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: ListOfUnsigned(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfUnsigned(pyobj)

   Handler for `list` / `tuple` whose entries can be converted to C ``unsigned``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfUnsigned.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C ``unsigned`` slots.

      The returned `~.ListOfUnsigned` owns the buffer (freed on garbage
      collection) and has a known length, so it is indexable, iterable,
      and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfUnsigned.fromObj(pyobj)

      Creates a ListOfUnsigned from the given object.

      In case ``pyobj`` is itself an ``ListOfUnsigned`` instance, this method
      returns it directly. No new ``ListOfUnsigned`` is created.



   .. py:method:: to_list() -> Any

      ListOfUnsigned.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfUnsigned.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: ListOfUnsignedLong(pyobj)

   Bases: :py:obj:`Pointer`


   ListOfUnsignedLong(pyobj)

   Handler for `list`/`tuple` whose entries can be converted to C ``unsigned long``

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


   .. py:method:: allocate(Py_ssize_tcount) -> Any
      :staticmethod:


      ListOfUnsignedLong.allocate(Py_ssize_t count)

      Allocate an owned, zero-initialized array of ``count`` C
      ``unsigned long`` slots.

      The returned `~.ListOfUnsignedLong` owns the buffer (freed on
      garbage collection) and has a known length, so it is indexable,
      iterable, and convertible via `~.to_list`/`~.to_tuple`.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      ListOfUnsignedLong.fromObj(pyobj)

      Creates a ListOfUnsignedLong from the given object.

      In case ``pyobj`` is itself an ``ListOfUnsignedLong`` instance, this method
      returns it directly. No new ``ListOfUnsignedLong`` is created.



   .. py:method:: to_list() -> Any

      ListOfUnsignedLong.to_list(self)

      Return the elements as a Python `list` of `int`.



   .. py:method:: to_tuple() -> Any

      ListOfUnsignedLong.to_tuple(self)

      Return the elements as a Python `tuple` of `int`.



.. py:class:: NDBuffer(pyobj)

   Bases: :py:obj:`Pointer`


   NDBuffer(pyobj)

   Handler for contiguous n-dimensional buffers of various element types

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

   Note:
       This type represents a dense, C-contiguous array; all of its
       addressing is derived from the shape and the itemsize. An input that
       implements the CUDA array interface protocol is therefore rejected
       if it carries a mask, a non-zero offset, or strides that describe a
       non-contiguous layout. Strides that spell out the contiguous layout
       the shape already implies are accepted.
   See:
       `~.configure`

   C Attributes:
       _ptr (``void *``, protected):
           Stores a pointer to the data of the original Python object.
       _py_buffer (`~.Py_buffer`, protected):
           Stores a pointer to the data of the original Python object.
       _py_buffer_acquired (`bool`, protected):
           Stores a pointer to the data of the original Python object.
       _cuda_array_interface (`dict`, protected):
           The CUDA array interface metadata, handed out as a copy by
           `~.NDBuffer.__cuda_array_interface__`.
       _pybuffer_obj (`object`, protected):
           Keeps the exporter of a wrapped `Py_buffer` alive.
       _typestr_bytes (`bytes`, protected):
           NUL-terminated format string handed to consumers of this
           Python buffer. Must stay alive as long as any view exists.
       _itemsize (``size_t``, protected):
           Stores the itemsize. The item size is not member of
           ``__cuda_array_interface__``.
       _py_buffer_shape (``Py_Ssize_t*``, private):
           A buffer to pass shape information to consumers
           of this Python buffer.


   .. py:attribute:: NUMPY_CHAR_CODES
      :type:  ClassVar[tuple]
      :value: Ellipsis



   .. py:attribute:: is_read_only
      :type:  _typeshed.Incomplete


   .. py:attribute:: itemsize
      :type:  _typeshed.Incomplete


   .. py:attribute:: rank
      :type:  _typeshed.Incomplete


   .. py:attribute:: shape
      :type:  _typeshed.Incomplete


   .. py:attribute:: size
      :type:  _typeshed.Incomplete


   .. py:attribute:: stream_as_int
      :type:  _typeshed.Incomplete


   .. py:attribute:: typestr
      :type:  _typeshed.Incomplete


   .. py:method:: configure(**kwargs) -> Any

      NDBuffer.configure(self, **kwargs)

      (Re-)configure this contiguous n-dimensional buffer.

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



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      NDBuffer.fromObj(pyobj)

      Creates a NDBuffer from the given object.

      In case ``pyobj`` is itself a ``NDBuffer`` instance, this method
      returns it directly. No new ``NDBuffer`` is created.



.. py:class:: Pointer(pyobj=...)

   Pointer(pyobj=None)

   Handler for Python arguments that need to be converted to a pointer type

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


   .. py:attribute:: is_ptr_null
      :type:  _typeshed.Incomplete


   .. py:method:: as_c_void_p() -> Any

      Pointer.as_c_void_p(self)

      Data pointer as ``ctypes.c_void_p``.



   .. py:method:: createRef() -> Pointer

      Pointer.createRef(self) -> Pointer

      Creates are reference to this pointer.

      Returns a `~.Pointer` that stores the address of this `~.Pointer's data pointer.

      Note:
          No ownership information is transferred.



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      Pointer.fromObj(pyobj)

      Creates a Pointer from the given object.

      In case ``pyobj`` is itself a ``Pointer`` instance, this method
      returns it directly. No new ``Pointer`` is created.



.. py:class:: PointerToInt(pyobj)

   Bases: :py:obj:`ListOfInt`


   PointerToInt(pyobj)

   Handler for a rank-0 pointer to a single C ``int``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToInt.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``int`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToInt.fromObj(pyobj)

      Creates a PointerToInt from the given object.

      In case ``pyobj`` is itself a ``PointerToInt`` instance, this method
      returns it directly. No new ``PointerToInt`` is created.



.. py:class:: PointerToInt64(pyobj)

   Bases: :py:obj:`ListOfInt64`


   PointerToInt64(pyobj)

   Handler for a rank-0 pointer to a single C ``int64_t``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToInt64.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``int64_t`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToInt64.fromObj(pyobj)

      Creates a PointerToInt64 from the given object.

      In case ``pyobj`` is itself a ``PointerToInt64`` instance, this method
      returns it directly. No new ``PointerToInt64`` is created.



.. py:class:: PointerToLong(pyobj)

   Bases: :py:obj:`ListOfLong`


   PointerToLong(pyobj)

   Handler for a rank-0 pointer to a single C ``long``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToLong.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``long`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToLong.fromObj(pyobj)

      Creates a PointerToLong from the given object.

      In case ``pyobj`` is itself a ``PointerToLong`` instance, this method
      returns it directly. No new ``PointerToLong`` is created.



.. py:class:: PointerToUInt64(pyobj)

   Bases: :py:obj:`ListOfUInt64`


   PointerToUInt64(pyobj)

   Handler for a rank-0 pointer to a single C ``uint64_t``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToUInt64.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``uint64_t`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToUInt64.fromObj(pyobj)

      Creates a PointerToUInt64 from the given object.

      In case ``pyobj`` is itself a ``PointerToUInt64`` instance, this method
      returns it directly. No new ``PointerToUInt64`` is created.



.. py:class:: PointerToUnsigned(pyobj)

   Bases: :py:obj:`ListOfUnsigned`


   PointerToUnsigned(pyobj)

   Handler for a rank-0 pointer to a single C ``unsigned int``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToUnsigned.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``unsigned int`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToUnsigned.fromObj(pyobj)

      Creates a PointerToUnsigned from the given object.

      In case ``pyobj`` is itself a ``PointerToUnsigned`` instance, this method
      returns it directly. No new ``PointerToUnsigned`` is created.



.. py:class:: PointerToUnsignedLong(pyobj)

   Bases: :py:obj:`ListOfUnsignedLong`


   PointerToUnsignedLong(pyobj)

   Handler for a rank-0 pointer to a single C ``unsigned long``.

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


   .. py:attribute:: value
      :type:  Any


   .. py:method:: allocate(Py_ssize_tcount=...) -> Any
      :staticmethod:


      PointerToUnsignedLong.allocate(Py_ssize_t count=1)

      Allocate an owned, zero-initialized array of ``count`` C ``unsigned long`` slots.

      Defaults to a single slot (the rank-0 ``PointerTo*`` use case).



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:


      PointerToUnsignedLong.fromObj(pyobj)

      Creates a PointerToUnsignedLong from the given object.

      In case ``pyobj`` is itself a ``PointerToUnsignedLong`` instance, this
      method returns it directly. No new ``PointerToUnsignedLong`` is created.



