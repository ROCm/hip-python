rocm.hipfile
============

.. py:module:: rocm.hipfile

.. autoapi-nested-parse::

   High-level Pythonic interface to the hipFile (Accelerated I/O Storage)
   library.

   This sub-package wraps the lower-level `~.rocm.bindings.hipfile`
   auto-generated bindings with idiomatic Python classes:

   * `~.Driver` — context manager for the hipFile driver lifecycle.
   * `~.FileHandle` — context manager for an open + registered file.
   * `~.Buffer` — context manager for a registered GPU memory region.

   plus the mirrored enums (`~.enums.OpError`,
   `~.enums.FileHandleType`), the `~.HipFileException` type, and
   the standalone `~.properties.driver_get_properties` /
   `~.properties.get_version` helpers.

   The complete copy-via-GPU-memory example lives in
   ``examples/0_Basic_Usage/hipfile_copy.py``.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /python_api/rocm/hipfile/buffer/index
   /python_api/rocm/hipfile/driver/index
   /python_api/rocm/hipfile/enums/index
   /python_api/rocm/hipfile/error/index
   /python_api/rocm/hipfile/file/index
   /python_api/rocm/hipfile/properties/index


Exceptions
----------

.. autoapisummary::

   rocm.hipfile.HipFileException


Classes
-------

.. autoapisummary::

   rocm.hipfile.Buffer
   rocm.hipfile.Driver
   rocm.hipfile.FileHandleType
   rocm.hipfile.OpError
   rocm.hipfile.FileHandle


Functions
---------

.. autoapisummary::

   rocm.hipfile.driver_get_properties
   rocm.hipfile.get_version


Package Contents
----------------

.. py:class:: Buffer(buffer_ptr, length, flags)

   Lifecycle manager for a hipFile-registered GPU memory region.

   The caller pre-allocates the underlying device buffer (typically
   via `~.rocm.bindings.hip.hipMalloc` or any equivalent
   GPU-allocator) and constructs a `~.Buffer` to register it
   with the hipFile driver. Deregistration happens on context exit
   (or via explicit `~.deregister`).

   Use as a context manager:

       with Buffer(ptr, length, flags=0) as registered_buf:
           ...


   .. py:method:: from_ctypes_void_p(ctypes_void_p: ctypes.c_void_p, length, flags)
      :classmethod:


      Construct a `~.Buffer` from a :py:class:`ctypes.c_void_p`.



   .. py:property:: ptr


   .. py:method:: deregister()


   .. py:method:: register()


.. py:class:: Driver

   Lifecycle manager for the hipFile driver.

   Each instance brackets one ``hipFileDriverOpen`` /
   ``hipFileDriverClose`` pair. The driver is reference-counted by the
   library; multiple ``Driver`` instances coexist safely.

   Use as a context manager:

       with Driver():
           ...

   or call `~.Driver.open` / `~.Driver.close` explicitly.


   .. py:method:: use_count()
      :staticmethod:


      Return the current driver reference count.



   .. py:method:: close()


   .. py:method:: open()


.. py:class:: FileHandleType

   Bases: :py:obj:`enum.IntEnum`


   Python enum mirroring ``hipFileFileHandleType_t`` with friendly names.


   .. py:attribute:: OPAQUE_FD


   .. py:attribute:: OPAQUE_WIN32


   .. py:attribute:: USERSPACE_FS


.. py:class:: OpError

   Bases: :py:obj:`enum.IntEnum`


   Python enum mirroring ``hipFileOpError_t`` with upstream-friendly names.


   .. py:attribute:: SUCCESS


   .. py:attribute:: DRIVER_NOT_INITIALIZED


   .. py:attribute:: DRIVER_INVALID_PROPS


   .. py:attribute:: DRIVER_UNSUPPORTED_LIMIT


   .. py:attribute:: DRIVER_VERSION_MISMATCH


   .. py:attribute:: DRIVER_VERSION_READ_ERROR


   .. py:attribute:: DRIVER_CLOSING


   .. py:attribute:: PLATFORM_NOT_SUPPORTED


   .. py:attribute:: IO_NOT_SUPPORTED


   .. py:attribute:: DEVICE_NOT_SUPPORTED


   .. py:attribute:: DRIVER_ERROR


   .. py:attribute:: HIP_DRIVER_ERROR


   .. py:attribute:: HIP_POINTER_INVALID


   .. py:attribute:: HIP_MEMORY_TYPE_INVALID


   .. py:attribute:: HIP_POINTER_RANGE_ERROR


   .. py:attribute:: HIP_CONTEXT_MISMATCH


   .. py:attribute:: INVALID_MAPPING_SIZE


   .. py:attribute:: INVALID_MAPPING_RANGE


   .. py:attribute:: INVALID_FILE_TYPE


   .. py:attribute:: INVALID_FILE_OPEN_FLAG


   .. py:attribute:: DIO_NOT_SET


   .. py:attribute:: INVALID_VALUE


   .. py:attribute:: MEMORY_ALREADY_REGISTERED


   .. py:attribute:: MEMORY_NOT_REGISTERED


   .. py:attribute:: PERMISSION_DENIED


   .. py:attribute:: DRIVER_ALREADY_OPEN


   .. py:attribute:: HANDLE_NOT_REGISTERED


   .. py:attribute:: HANDLE_ALREADY_REGISTERED


   .. py:attribute:: DEVICE_NOT_FOUND


   .. py:attribute:: INTERNAL_ERROR


   .. py:attribute:: GET_NEW_FD_FAILED


   .. py:attribute:: DRIVER_SETUP_ERROR


   .. py:attribute:: IO_DISABLED


   .. py:attribute:: BATCH_SUBMIT_FAILED


   .. py:attribute:: GPU_MEMORY_PINNING_FAILED


   .. py:attribute:: BATCH_FULL


   .. py:attribute:: ASYNC_NOT_SUPPORTED


   .. py:attribute:: IO_MAX_ERROR


.. py:exception:: HipFileException(hipfile_err, hip_err)

   Bases: :py:obj:`Exception`


   Exception raised on a non-success hipFile error.

   Carries both the hipFile-level error code (``hipfile_err``) and the
   underlying HIP driver error (``hip_err``) when the former is
   ``OpError.HIP_DRIVER_ERROR``.


   .. py:property:: hipfile_err


   .. py:property:: hip_err


.. py:class:: FileHandle(path, flags, mode=DEFAULT_MODE, handle_type=FileHandleType.OPAQUE_FD)

   Lifecycle manager for a hipFile-registered open file.

   Wraps `~.hipFileHandleRegister` / `~.hipFileHandleDeregister` plus
   synchronous `~.hipFileRead` / `~.hipFileWrite`.

   Use as a context manager:

       with FileHandle(path, os.O_RDWR | os.O_DIRECT) as fh:
           fh.read(buf, size, file_offset, buffer_offset)
           fh.write(buf, size, file_offset, buffer_offset)


   .. py:attribute:: DEFAULT_MODE
      :value: 420



   .. py:property:: handle_type


   .. py:property:: flags


   .. py:property:: handle


   .. py:property:: mode


   .. py:property:: path


   .. py:method:: open()


   .. py:method:: close()


   .. py:method:: read(buffer, size, file_offset, buffer_offset)

      Synchronous read into a registered `~.buffer.Buffer`.

      Returns the number of bytes read on success. Raises
      `~.error.HipFileException` (with the parsed
      `~.enums.OpError` and HIP driver error) on a hipFile-level
      error, or ``OSError`` (with the real ``errno``) on a
      POSIX-level error.



   .. py:method:: write(buffer, size, file_offset, buffer_offset)

      Synchronous write from a registered `~.buffer.Buffer`.

      Returns the number of bytes written on success. Same error
      semantics as `~.FileHandle.read`.



.. py:function:: driver_get_properties()

   Return the driver properties struct as set by `~.hipFileDriverGetProperties`.


.. py:function:: get_version()

   Return ``(major, minor, patch)`` for the loaded libhipfile.so.


