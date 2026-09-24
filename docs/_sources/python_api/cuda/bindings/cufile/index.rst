cuda.bindings.cufile
====================

.. py:module:: cuda.bindings.cufile

.. autoapi-nested-parse::

   ``cuda.bindings.cufile`` interop layer implemented on top of hipFILE.

   This is a HAND-WRITTEN Cython module (it is NOT emitted by the hip-python code
   generator). It mirrors the public ``cuda.bindings.cufile`` cpdef surface
   (snake_case functions, ``intptr_t`` pointer arguments, ``cuFileError`` on
   failure, and the ``Descr``/``IOParams``/``IOEvents`` array helpers) so that
   cuFile code can run unmodified against AMD's hipFILE. Every call is forwarded
   to the corresponding hipFILE C symbol via the low-level
   ``cuda.bindings.cycufile`` alias module.



Exceptions
----------

.. autoapisummary::

   cuda.bindings.cufile.cuFileError


Classes
-------

.. autoapisummary::

   cuda.bindings.cufile.OpError
   cuda.bindings.cufile.DriverStatusFlags
   cuda.bindings.cufile.DriverControlFlags
   cuda.bindings.cufile.FeatureFlags
   cuda.bindings.cufile.FileHandleType
   cuda.bindings.cufile.Opcode
   cuda.bindings.cufile.Status
   cuda.bindings.cufile.BatchMode
   cuda.bindings.cufile.SizeTConfigParameter
   cuda.bindings.cufile.BoolConfigParameter
   cuda.bindings.cufile.StringConfigParameter
   cuda.bindings.cufile.Descr
   cuda.bindings.cufile.IOParams
   cuda.bindings.cufile.IOEvents


Functions
---------

.. autoapisummary::

   cuda.bindings.cufile.driver_open
   cuda.bindings.cufile.driver_close
   cuda.bindings.cufile.use_count
   cuda.bindings.cufile.driver_get_properties
   cuda.bindings.cufile.driver_set_poll_mode
   cuda.bindings.cufile.driver_set_max_direct_io_size
   cuda.bindings.cufile.driver_set_max_cache_size
   cuda.bindings.cufile.driver_set_max_pinned_mem_size
   cuda.bindings.cufile.handle_register
   cuda.bindings.cufile.handle_deregister
   cuda.bindings.cufile.buf_register
   cuda.bindings.cufile.buf_deregister
   cuda.bindings.cufile.read
   cuda.bindings.cufile.write
   cuda.bindings.cufile.batch_io_set_up
   cuda.bindings.cufile.batch_io_submit
   cuda.bindings.cufile.batch_io_get_status
   cuda.bindings.cufile.batch_io_cancel
   cuda.bindings.cufile.batch_io_destroy
   cuda.bindings.cufile.read_async
   cuda.bindings.cufile.write_async
   cuda.bindings.cufile.stream_register
   cuda.bindings.cufile.stream_deregister
   cuda.bindings.cufile.get_version
   cuda.bindings.cufile.get_parameter_size_t
   cuda.bindings.cufile.get_parameter_bool
   cuda.bindings.cufile.get_parameter_string
   cuda.bindings.cufile.set_parameter_size_t
   cuda.bindings.cufile.set_parameter_bool
   cuda.bindings.cufile.set_parameter_string
   cuda.bindings.cufile.op_status_error


Module Contents
---------------

.. py:class:: OpError

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileOpError``.


   .. py:attribute:: SUCCESS
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_NOT_INITIALIZED
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_INVALID_PROPS
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_UNSUPPORTED_LIMIT
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_VERSION_MISMATCH
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_VERSION_READ_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_CLOSING
      :type:  ClassVar[int]


   .. py:attribute:: PLATFORM_NOT_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: IO_NOT_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: DEVICE_NOT_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NVFS_DRIVER_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: CUDA_DRIVER_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: CUDA_POINTER_INVALID
      :type:  ClassVar[int]


   .. py:attribute:: CUDA_MEMORY_TYPE_INVALID
      :type:  ClassVar[int]


   .. py:attribute:: CUDA_POINTER_RANGE_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: CUDA_CONTEXT_MISMATCH
      :type:  ClassVar[int]


   .. py:attribute:: INVALID_MAPPING_SIZE
      :type:  ClassVar[int]


   .. py:attribute:: INVALID_MAPPING_RANGE
      :type:  ClassVar[int]


   .. py:attribute:: INVALID_FILE_TYPE
      :type:  ClassVar[int]


   .. py:attribute:: INVALID_FILE_OPEN_FLAG
      :type:  ClassVar[int]


   .. py:attribute:: DIO_NOT_SET
      :type:  ClassVar[int]


   .. py:attribute:: INVALID_VALUE
      :type:  ClassVar[int]


   .. py:attribute:: MEMORY_ALREADY_REGISTERED
      :type:  ClassVar[int]


   .. py:attribute:: MEMORY_NOT_REGISTERED
      :type:  ClassVar[int]


   .. py:attribute:: PERMISSION_DENIED
      :type:  ClassVar[int]


   .. py:attribute:: DRIVER_ALREADY_OPEN
      :type:  ClassVar[int]


   .. py:attribute:: HANDLE_NOT_REGISTERED
      :type:  ClassVar[int]


   .. py:attribute:: HANDLE_ALREADY_REGISTERED
      :type:  ClassVar[int]


   .. py:attribute:: DEVICE_NOT_FOUND
      :type:  ClassVar[int]


   .. py:attribute:: INTERNAL_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: GETNEWFD_FAILED
      :type:  ClassVar[int]


   .. py:attribute:: NVFS_SETUP_ERROR
      :type:  ClassVar[int]


   .. py:attribute:: IO_DISABLED
      :type:  ClassVar[int]


   .. py:attribute:: BATCH_SUBMIT_FAILED
      :type:  ClassVar[int]


   .. py:attribute:: GPU_MEMORY_PINNING_FAILED
      :type:  ClassVar[int]


   .. py:attribute:: BATCH_FULL
      :type:  ClassVar[int]


   .. py:attribute:: ASYNC_NOT_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: IO_MAX_ERROR
      :type:  ClassVar[int]


.. py:class:: DriverStatusFlags

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileDriverStatusFlags_t``.


   .. py:attribute:: LUSTRE_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: WEKAFS_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NFS_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: GPFS_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NVME_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NVMEOF_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: SCSI_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: SCALEFLUX_CSD_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NVMESH_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: BEEGFS_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: NVME_P2P_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: SCATEFS_SUPPORTED
      :type:  ClassVar[int]


.. py:class:: DriverControlFlags

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileDriverControlFlags_t``.


   .. py:attribute:: USE_POLL_MODE
      :type:  ClassVar[int]


   .. py:attribute:: ALLOW_COMPAT_MODE
      :type:  ClassVar[int]


.. py:class:: FeatureFlags

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileFeatureFlags_t``.


   .. py:attribute:: DYN_ROUTING_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: BATCH_IO_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: STREAMS_SUPPORTED
      :type:  ClassVar[int]


   .. py:attribute:: PARALLEL_IO_SUPPORTED
      :type:  ClassVar[int]


.. py:class:: FileHandleType

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileFileHandleType``.


   .. py:attribute:: OPAQUE_FD
      :type:  ClassVar[int]


   .. py:attribute:: OPAQUE_WIN32
      :type:  ClassVar[int]


   .. py:attribute:: USERSPACE_FS
      :type:  ClassVar[int]


.. py:class:: Opcode

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileOpcode_t``.


   .. py:attribute:: READ
      :type:  ClassVar[int]


   .. py:attribute:: WRITE
      :type:  ClassVar[int]


.. py:class:: Status

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileStatus_t``.


   .. py:attribute:: WAITING
      :type:  ClassVar[int]


   .. py:attribute:: PENDING
      :type:  ClassVar[int]


   .. py:attribute:: INVALID
      :type:  ClassVar[int]


   .. py:attribute:: CANCELED
      :type:  ClassVar[int]


   .. py:attribute:: COMPLETE
      :type:  ClassVar[int]


   .. py:attribute:: TIMEOUT
      :type:  ClassVar[int]


   .. py:attribute:: FAILED
      :type:  ClassVar[int]


.. py:class:: BatchMode

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileBatchMode_t``.


   .. py:attribute:: BATCH
      :type:  ClassVar[int]


.. py:class:: SizeTConfigParameter

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileSizeTConfigParameter_t``.


   .. py:attribute:: PROFILE_STATS
      :type:  ClassVar[int]


   .. py:attribute:: EXECUTION_MAX_IO_QUEUE_DEPTH
      :type:  ClassVar[int]


   .. py:attribute:: EXECUTION_MAX_IO_THREADS
      :type:  ClassVar[int]


   .. py:attribute:: EXECUTION_MIN_IO_THRESHOLD_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: EXECUTION_MAX_REQUEST_PARALLELISM
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_MAX_DIRECT_IO_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_PER_BUFFER_CACHE_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_IO_BATCHSIZE
      :type:  ClassVar[int]


   .. py:attribute:: POLLTHRESHOLD_SIZE_KB
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_BATCH_IO_TIMEOUT_MS
      :type:  ClassVar[int]


.. py:class:: BoolConfigParameter

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileBoolConfigParameter_t``.


   .. py:attribute:: PROPERTIES_USE_POLL_MODE
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_ALLOW_COMPAT_MODE
      :type:  ClassVar[int]


   .. py:attribute:: FORCE_COMPAT_MODE
      :type:  ClassVar[int]


   .. py:attribute:: FS_MISC_API_CHECK_AGGRESSIVE
      :type:  ClassVar[int]


   .. py:attribute:: EXECUTION_PARALLEL_IO
      :type:  ClassVar[int]


   .. py:attribute:: PROFILE_NVTX
      :type:  ClassVar[int]


   .. py:attribute:: PROPERTIES_ALLOW_SYSTEM_MEMORY
      :type:  ClassVar[int]


   .. py:attribute:: USE_PCIP2PDMA
      :type:  ClassVar[int]


   .. py:attribute:: PREFER_IO_URING
      :type:  ClassVar[int]


   .. py:attribute:: FORCE_ODIRECT_MODE
      :type:  ClassVar[int]


   .. py:attribute:: SKIP_TOPOLOGY_DETECTION
      :type:  ClassVar[int]


   .. py:attribute:: STREAM_MEMOPS_BYPASS
      :type:  ClassVar[int]


.. py:class:: StringConfigParameter

   Bases: :py:obj:`enum.IntEnum`


   See ``hipFileStringConfigParameter_t``.


   .. py:attribute:: LOGGING_LEVEL
      :type:  ClassVar[int]


   .. py:attribute:: ENV_LOGFILE_PATH
      :type:  ClassVar[int]


   .. py:attribute:: LOG_DIR
      :type:  ClassVar[int]


.. py:exception:: cuFileError(status: int, cu_err: int | None = ...)

   Bases: :py:obj:`Exception`


   Raised when a cuFile operation returns a non-``SUCCESS`` status.

   Args:
       status (``int``):
           the ``hipFileOpError`` / `~.OpError` status code.

       cu_err (``int`` | ``None``):
           for ``OpError.CUDA_DRIVER_ERROR`` this carries the
           underlying HIP driver error code (``hipError_t``); ``None``
           otherwise.


   .. py:attribute:: status
      :type:  int


   .. py:attribute:: cu_err
      :type:  int | None


.. py:class:: Descr(size: int = ...)

   Empty-initialize an array of ``hipFileDescr_t``.

   A ``hipFileDescr_t`` carries the OS-neutral file identity handed to
   `~.handle_register`: a ``type`` (a `~.FileHandleType`), a ``handle``
   union (the Linux ``fd`` or a Windows handle), and an optional ``fs_ops``
   table. Element ``0`` is exposed directly through the ``type`` / ``handle`` /
   ``fs_ops`` properties; use ``descr[i]`` to view any other element. ``ptr``
   yields the base C address to hand to the cuFile calls.

   Args:
       size (``int``):
           the number of contiguous elements to allocate (default 1).


   .. py:property:: ptr
      :type: int



   .. py:property:: type
      :type: int



   .. py:property:: handle
      :type: Any



   .. py:property:: fs_ops
      :type: int



.. py:class:: IOParams(size: int = ...)

   Empty-initialize an array of ``hipFileIOParams_t``.

   Each ``hipFileIOParams_t`` describes one request in a batch submitted with
   `~.batch_io_submit`: the ``mode`` (a `~.BatchMode`), the file
   handle ``fh``, the ``opcode`` (a `~.Opcode`), an opaque ``cookie``, and
   the per-request ``u.batch`` fields (device pointer base/offset, file offset
   and size). Element ``0`` is exposed directly through the properties; use
   ``params[i]`` to view any other element, and ``ptr`` for the base C address.

   Args:
       size (``int``):
           the number of contiguous elements to allocate (default 1).


   .. py:property:: ptr
      :type: int



   .. py:property:: mode
      :type: int



   .. py:property:: u
      :type: Any



   .. py:property:: fh
      :type: int



   .. py:property:: opcode
      :type: int



   .. py:property:: cookie
      :type: int



.. py:class:: IOEvents(size: int = ...)

   Empty-initialize an array of ``hipFileIOEvents_t``.

   Each ``hipFileIOEvents_t`` receives the outcome of one batch request from
   `~.batch_io_get_status`: the request ``cookie``, the ``status`` (a
   `~.Status`), and ``ret`` (the bytes transacted, valid only once the
   request has completed successfully). Element ``0`` is exposed directly
   through the properties; use ``events[i]`` to view any other element, and
   ``ptr`` for the base C address to hand to `~.batch_io_get_status`.

   Args:
       size (``int``):
           the number of contiguous elements to allocate (default 1).


   .. py:property:: ptr
      :type: int



   .. py:property:: cookie
      :type: int



   .. py:property:: status
      :type: int



   .. py:property:: ret
      :type: int



.. py:function:: driver_open() -> None

   Initialize the cuFile library and open the driver.

   Explicitly opens the cuFile driver session used for the file IO
   operations. Calling this is optional: driver initialization otherwise
   happens implicitly on the first use of `~.handle_register`,
   `~.read`, `~.write`, or `~.buf_register`.

   Raises:
       `~.cuFileError`:
           if the driver fails to initialize, e.g.
           ``OpError.DRIVER_NOT_INITIALIZED``, ``OpError.PERMISSION_DENIED``,
           ``OpError.DRIVER_VERSION_MISMATCH``, or
           ``OpError.PLATFORM_NOT_SUPPORTED``.


.. py:function:: driver_close() -> None

   Reset the cuFile library and release the driver.

   Closes the driver session and frees the associated resources. Any
   buffers still registered via `~.buf_register` are implicitly
   deregistered, and any in-flight IO receives an error. The driver may be
   reopened afterwards; this cleanup also happens implicitly on process exit.

   Raises:
       `~.cuFileError`:
           if the driver was not initialized
           (``OpError.DRIVER_NOT_INITIALIZED``).


.. py:function:: use_count() -> int

   Return the process-wide cuFile driver use count.

   Returns:
       ``int``:
           the number of times the cuFile driver is currently in use by
           this process at the moment of the call.


.. py:function:: driver_get_properties(props: int) -> None

   Get the driver session properties.

   If the driver is not open, the staged/default properties are returned;
   otherwise the current properties are returned. The structure reports the
   driver capabilities (supported filesystems, poll/compat control
   flags, feature flags, and the IO/cache/pinned-memory size limits).

   Args:
       props (``int``):
           address (as a Python integer) of a caller-allocated
           ``hipFileDriverProps_t`` structure to fill in.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_NOT_INITIALIZED``,
           ``OpError.DRIVER_VERSION_MISMATCH``, or ``OpError.INVALID_VALUE``
           if ``props`` is invalid.


.. py:function:: driver_set_poll_mode(poll: bool, poll_threshold_size: int) -> None

   Set whether the Read/Write APIs use polling to do IO operations.

   Must be called before the driver is opened. When poll mode is enabled, IO
   whose size is less than or equal to ``poll_threshold_size`` is completed by
   polling.

   Args:
       poll (``bool``):
           whether to enable poll mode.

       poll_threshold_size (``int``):
           the IO size threshold for polling, in KB
           (must be 4K aligned; the default is 4KB).

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_NOT_INITIALIZED`` or
           ``OpError.DRIVER_UNSUPPORTED_LIMIT`` for an invalid threshold.


.. py:function:: driver_set_max_direct_io_size(max_direct_io_size: int) -> None

   Set the max direct IO size used to talk to the driver.

   Must be called before the driver is opened. This is the maximum IO chunk
   size the driver issues to the underlying filesystem (and, in compatibility
   mode, the maximum chunk size the library uses for POSIX read/write).

   Args:
       max_direct_io_size (``int``):
           the maximum direct IO size, in KB (must be
           4K aligned; the default is 16384KB).

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_NOT_INITIALIZED`` or
           ``OpError.DRIVER_UNSUPPORTED_LIMIT`` for an invalid size.


.. py:function:: driver_set_max_cache_size(max_cache_size: int) -> None

   Set the max GPU memory reserved per device for internal buffering.

   Must be called before the driver is opened. This is the per-device GPU
   buffer space the library uses internally, e.g. to handle unaligned IO and
   optimal IO path routing; it may be rounded down to the nearest GPU page
   size.

   Args:
       max_cache_size (``int``):
           the maximum per-device GPU cache size, in KB
           (must be 4K aligned; the default is 131072KB).

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_NOT_INITIALIZED`` or
           ``OpError.DRIVER_UNSUPPORTED_LIMIT`` for an invalid size.


.. py:function:: driver_set_max_pinned_mem_size(max_pinned_size: int) -> None

   Set the max buffer space that is pinned for ``buf_register``.

   Must be called before the driver is opened. This is the upper limit on GPU
   memory that can be pinned and mapped for device IO (as used by
   `~.buf_register`); it may be rounded down to the nearest GPU page size.

   Args:
       max_pinned_size (``int``):
           the maximum pinned buffer space, in KB (must be
           4K aligned). ``UINT64_MAX`` is equivalent to no enforced limit.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_NOT_INITIALIZED`` or
           ``OpError.DRIVER_UNSUPPORTED_LIMIT`` for an invalid size.


.. py:function:: handle_register(descr: int) -> int

   Register an open file for GPU IO.

   Wraps an OS-specific file descriptor in an OS-agnostic ``hipFileHandle_t``
   and performs (memoized) checks on IO supportability based on the mount
   point and how the file was opened. Registration is required before issuing
   cuFile IO on a file.

   Args:
       descr (``int``):
           address (as a Python integer) of a caller-populated
           ``hipFileDescr_t``; see `~.Descr`. For
           Linux this carries the file's ``fd`` and a ``type`` of
           ``FileHandleType.OPAQUE_FD``.

   Returns:
       ``int``:
           an opaque ``hipFileHandle_t`` (as a Python integer) to pass to the
           read/write/async/batch APIs.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.IO_NOT_SUPPORTED``,
           ``OpError.INVALID_VALUE``, ``OpError.INVALID_FILE_OPEN_FLAG``,
           ``OpError.INVALID_FILE_TYPE``, or
           ``OpError.HANDLE_ALREADY_REGISTERED``.


.. py:function:: handle_deregister(fh: int) -> None

   Release a registered file handle from cuFile.

   Frees the cuFile resources claimed by `~.handle_register`. Call this
   only after ensuring no IO is outstanding on the handle (otherwise the
   behavior is undefined). The underlying file descriptor is *not* closed; the
   caller must still ``os.close`` it.

   Args:
       fh (``int``):
           the file handle (as a Python integer) returned by
           `~.handle_register`.


.. py:function:: buf_register(buf_ptr_base: int, length: int, flags: int) -> None

   Register a device/host memory region with cuFile for GPU IO.

   Pins existing device memory (or host memory) for direct file IO.
   Registration is optional but recommended: it incurs a
   significant one-time cost that should be amortized off the critical path.
   `~.read` / `~.write` must use the same ``buf_ptr_base`` as their
   base address to benefit from the registration.

   Args:
       buf_ptr_base (``int``):
           base address (as a Python integer) of the device or
           host buffer to register.

       length (``int``):
           the size, in bytes from the start of the buffer, to map.

       flags (``int``):
           reserved for future use; must be 0.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.MEMORY_ALREADY_REGISTERED``,
           ``OpError.CUDA_MEMORY_TYPE_INVALID``,
           ``OpError.CUDA_POINTER_RANGE_ERROR``,
           ``OpError.INVALID_MAPPING_SIZE``, or
           ``OpError.GPU_MEMORY_PINNING_FAILED``.


.. py:function:: buf_deregister(buf_ptr_base: int) -> None

   Deregister a device/host memory region from cuFile.

   Releases the pinned-memory mappings created by `~.buf_register`.

   Args:
       buf_ptr_base (``int``):
           the base address (as a Python integer) that was
           passed to `~.buf_register`.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.MEMORY_NOT_REGISTERED`` if ``buf_ptr_base``
           was not registered.


.. py:function:: read(fh: int, buf_ptr_base: int, size: int, file_offset: int, buf_ptr_offset: int) -> int

   Read from a registered file handle into device/host memory.

   Synchronously reads ``size`` bytes from the file at ``file_offset`` into the
   buffer. Works for unaligned offsets and sizes
   (with a possible performance cost), and blocks until the IO completes.

   Args:
       fh (``int``):
           the file handle from `~.handle_register`.

       buf_ptr_base (``int``):
           base address of the destination device/host buffer.
           For registered buffers this must equal the base address passed to
           `~.buf_register`.

       size (``int``):
           the number of bytes to read.

       file_offset (``int``):
           the offset in the file to read from.

       buf_ptr_offset (``int``):
           the offset relative to ``buf_ptr_base`` to read
           into (use 0 to read into the very start; only meaningful for
           registered buffers).

   Returns:
       ``int``:
           the number of bytes read.

   Raises:
       ``OSError``:
           on a POSIX/filesystem error (raw return ``-1``); ``errno`` is
           set accordingly.

       `~.cuFileError`:
           on any other (cuFile-specific) error.


.. py:function:: write(fh: int, buf_ptr_base: int, size: int, file_offset: int, buf_ptr_offset: int) -> int

   Write device/host memory to a registered file handle.

   Synchronously writes ``size`` bytes from the buffer to the file at
   ``file_offset``. Works for unaligned offsets and
   sizes (with a possible performance cost), and blocks until the IO completes.
   Note that the write does not guarantee metadata is flushed; use ``os.fsync``
   (or open the file with ``O_SYNC``) for durability.

   Args:
       fh (``int``):
           the file handle from `~.handle_register`.

       buf_ptr_base (``int``):
           base address of the source device/host buffer. For
           registered buffers this must equal the base address passed to
           `~.buf_register`.

       size (``int``):
           the number of bytes to write.

       file_offset (``int``):
           the offset in the file to write to.

       buf_ptr_offset (``int``):
           the offset relative to ``buf_ptr_base`` to write
           from (use 0 to write from the very start; only meaningful for
           registered buffers).

   Returns:
       ``int``:
           the number of bytes written.

   Raises:
       ``OSError``:
           on a POSIX/filesystem error (raw return ``-1``); ``errno`` is
           set accordingly.

       `~.cuFileError`:
           on any other (cuFile-specific) error.


.. py:function:: batch_io_set_up(nr: int) -> int

   Prepare a batch IO operation.

   Must be the first call in a batch IO sequence. Reserves capacity for up to
   ``nr`` batch entries and returns a handle for the subsequent batch calls.

   Args:
       nr (``int``):
           the maximum number of entries (events) the batch will hold;
           should be at least 1 and within the driver's supported batch size.

   Returns:
       ``int``:
           an opaque ``hipFileBatchHandle_t`` (as a Python integer) for use
           with the other ``batch_io_*`` functions.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INTERNAL_ERROR`` on failure.


.. py:function:: batch_io_submit(batch_idp: int, nr: int, iocbp: int, flags: int) -> None

   Enqueue a batch of IO requests.

   Submits ``nr`` read/write requests described by an array of
   ``hipFileIOParams_t``. This is asynchronous with respect to the host thread:
   monitor progress with `~.batch_io_get_status` and cancel/destroy with
   `~.batch_io_cancel` / `~.batch_io_destroy`.

   Args:
       batch_idp (``int``):
           the batch handle from `~.batch_io_set_up`.

       nr (``int``):
           the number of requests to submit; must be > 0 and <= the
           ``nr`` passed to `~.batch_io_set_up`.

       iocbp (``int``):
           address of a ``hipFileIOParams_t`` array of length ``nr``;
           see `~.IOParams`.

       flags (``int``):
           reserved for future use; must be 0.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INTERNAL_ERROR`` on failure.


.. py:function:: batch_io_get_status(batch_idp: int, min_nr: int, nr: int, iocbp: int, timeout: int) -> None

   Poll for the status of completed batch IO operations.

   Waits for at least ``min_nr`` completions (or until ``timeout`` elapses),
   filling an array of ``hipFileIOEvents_t`` with the per-IO status, error and
   bytes transacted. The bytes-transacted field is valid only for
   successfully completed IOs.

   Args:
       batch_idp (``int``):
           the batch handle from `~.batch_io_set_up`.

       min_nr (``int``):
           the minimum number of completed entries to wait for; must
           be >= 0 and <= ``*nr``.

       nr (``int``):
           address of an ``unsigned int`` used as input/output: on input
           the maximum number of entries to poll for, on output the number of
           completed IOs.

       iocbp (``int``):
           address of a ``hipFileIOEvents_t`` array to receive the
           completed IO statuses; see `~.IOEvents`.

       timeout (``int``):
           address of a ``struct timespec`` giving the maximum time
           to wait; if it elapses, fewer than ``min_nr`` entries may be
           returned.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` for an invalid batch ID.
           Note that success here refers to the API call itself; inspect the
           per-IO ``iocbp`` entries for the individual IO status.


.. py:function:: batch_io_cancel(batch_idp: int) -> None

   Cancel all pending batch IO operations.

   Attempts to cancel the in-flight IOs for the batch; there is no guarantee
   an already-executing IO can be canceled. Canceled IOs report
   ``Status.CANCELED`` via `~.batch_io_get_status`.

   Args:
       batch_idp (``int``):
           the batch handle from `~.batch_io_set_up`.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` on failure.


.. py:function:: batch_io_destroy(batch_idp: int) -> None

   Destroy the batch IO handle and free the associated resources.

   Destroys the batch context and the resources allocated by
   `~.batch_io_set_up`.

   Args:
       batch_idp (``int``):
           the batch handle from `~.batch_io_set_up`.


.. py:function:: read_async(fh: int, buf_ptr_base: int, size_p: int, file_offset_p: int, buf_ptr_offset_p: int, bytes_read_p: int, stream: int) -> None

   Enqueue an asynchronous read on ``stream``.

   Enqueues a read into device/host memory, FIFO-ordered within the CUDA/HIP
   stream. The size/offset arguments are passed by pointer because, unless
   fixed via `~.stream_register`, they are not evaluated until the
   operation executes; ``size_p`` should be set to the maximum possible IO
   size at submission time. All of these pointers are caller-allocated
   (``intptr_t``) and must outlive the operation.

   Args:
       fh (``int``):
           the file handle from `~.handle_register`.

       buf_ptr_base (``int``):
           base address of the destination device/host buffer.
           For registered buffers this must equal the base address passed to
           `~.buf_register`.

       size_p (``int``):
           address of a ``size_t`` holding the number of bytes to
           read.

       file_offset_p (``int``):
           address of an ``off_t`` holding the file offset to
           read from.

       buf_ptr_offset_p (``int``):
           address of an ``off_t`` holding the offset
           relative to ``buf_ptr_base``.

       bytes_read_p (``int``):
           address of an ``ssize_t`` (initialized to 0) that,
           after the stream completes, holds the number of bytes read (``-1``
           on an IO error, or a negative ``hipFileOpError`` value otherwise).

       stream (``int``):
           the CUDA/HIP stream to enqueue on; 0 (NULL) makes the
           operation synchronous.

   Raises:
       `~.cuFileError`:
           on a submission error.


.. py:function:: write_async(fh: int, buf_ptr_base: int, size_p: int, file_offset_p: int, buf_ptr_offset_p: int, bytes_written_p: int, stream: int) -> None

   Enqueue an asynchronous write on ``stream``.

   Enqueues a write from device/host memory, FIFO-ordered within the CUDA/HIP
   stream. The size/offset arguments are passed by pointer because, unless
   fixed via `~.stream_register`, they are not evaluated until the
   operation executes; ``size_p`` should be set to the maximum possible IO
   size at submission time. All of these pointers are caller-allocated
   (``intptr_t``) and must outlive the operation.

   Args:
       fh (``int``):
           the file handle from `~.handle_register`.

       buf_ptr_base (``int``):
           base address of the source device/host buffer. For
           registered buffers this must equal the base address passed to
           `~.buf_register`.

       size_p (``int``):
           address of a ``size_t`` holding the number of bytes to
           write.

       file_offset_p (``int``):
           address of an ``off_t`` holding the file offset to
           write to.

       buf_ptr_offset_p (``int``):
           address of an ``off_t`` holding the offset
           relative to ``buf_ptr_base``.

       bytes_written_p (``int``):
           address of an ``ssize_t`` (initialized to 0)
           that, after the stream completes, holds the number of bytes written
           (``-1`` on an IO error, or a negative ``hipFileOpError`` value
           otherwise).

       stream (``int``):
           the CUDA/HIP stream to enqueue on; 0 (NULL) makes the
           operation synchronous.

   Raises:
       `~.cuFileError`:
           on a submission error.


.. py:function:: stream_register(stream: int, flags: int) -> None

   Register a stream for asynchronous GPU IO.

   Optional API that allocates resources for stream IO and declares which IO
   parameters are already known at submission time. The call synchronizes on
   the stream before allocating resources.

   Args:
       stream (``int``):
           the CUDA/HIP stream to register; 0 (NULL) selects the
           default stream.

       flags (``int``):
           bitmask declaring which inputs are fixed at submission
           time: ``0x1`` buffer offset, ``0x2`` file offset, ``0x4`` size,
           ``0x8`` all inputs 4K-aligned; ``0xf`` means all are aligned and
           known (best performance). ``0x0`` means all parameters are valid
           only at execution time.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` or
           ``OpError.PLATFORM_NOT_SUPPORTED``.


.. py:function:: stream_deregister(stream: int) -> None

   Deregister a stream and free the associated resources.

   Optional API that frees the resources allocated by `~.stream_register`.
   The call synchronizes on the stream first. Streams are also deregistered
   automatically by `~.driver_close`.

   Args:
       stream (``int``):
           the stream (as a Python integer) previously passed to
           `~.stream_register`.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` or
           ``OpError.PLATFORM_NOT_SUPPORTED``.


.. py:function:: get_version() -> int

   Return the cuFile library version as a packed integer.

   The version is packed as ``1000 * major + 10 * minor + patch`` (e.g. cuFile
   1.7.0 is ``1070``). hipFILE reports the ``(major, minor, patch)`` components
   separately; this shim packs them to match cuFile's single-integer format.
   It can be used to gate on the presence of a specific library feature.

   Returns:
       ``int``:
           the packed version number.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.DRIVER_VERSION_READ_ERROR`` if the version
           is unavailable.


.. py:function:: get_parameter_size_t(param: int) -> int

   Get the value of a ``size_t`` configuration parameter.

   If the driver is open the current runtime value is returned; otherwise the
   currently staged value is returned (staged values are cleared when the
   driver opens).

   Args:
       param (``int``):
           the parameter to read; a `~.SizeTConfigParameter`
           value.

   Returns:
       ``int``:
           the parameter's value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` for an invalid parameter.


.. py:function:: get_parameter_bool(param: int) -> bool

   Get the value of a Boolean configuration parameter.

   If the driver is open the current runtime value is returned; otherwise the
   currently staged value is returned (staged values are cleared when the
   driver opens).

   Args:
       param (``int``):
           the parameter to read; a `~.BoolConfigParameter`
           value.

   Returns:
       ``bool``:
           the parameter's value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` for an invalid parameter.


.. py:function:: get_parameter_string(param: int, len: int) -> str

   Get the value of a string configuration parameter.

   If the driver is open the current runtime value is returned; otherwise the
   currently staged value is returned (staged values are cleared when the
   driver opens).

   Args:
       param (``int``):
           the parameter to read; a `~.StringConfigParameter`
           value.

       len (``int``):
           the size, in bytes, of the internal buffer to allocate for
           the returned string.

   Returns:
       ``str``:
           the parameter's value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` for an invalid parameter.


.. py:function:: set_parameter_size_t(param: int, value: int) -> None

   Set the value of a ``size_t`` configuration parameter.

   Must be called before the driver is opened; the value takes effect once the
   driver opens. If the same parameter is set multiple times, the last value
   wins. Precedence (highest to lowest) is ``set_parameter_*`` > environment
   variable > the built-in defaults.

   Args:
       param (``int``):
           the parameter to set; a `~.SizeTConfigParameter`
           value.

       value (``int``):
           the new value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` or
           ``OpError.DRIVER_ALREADY_OPEN``.


.. py:function:: set_parameter_bool(param: int, value: bool) -> None

   Set the value of a Boolean configuration parameter.

   Must be called before the driver is opened; the value takes effect once the
   driver opens. If the same parameter is set multiple times, the last value
   wins. Precedence (highest to lowest) is ``set_parameter_*`` > environment
   variable > the built-in defaults.

   Args:
       param (``int``):
           the parameter to set; a `~.BoolConfigParameter`
           value.

       value (``bool``):
           the new value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` or
           ``OpError.DRIVER_ALREADY_OPEN``.


.. py:function:: set_parameter_string(param: int, desc_str: int) -> None

   Set the value of a string configuration parameter.

   Must be called before the driver is opened; the value takes effect once the
   driver opens. See `~.set_parameter_size_t` for the precedence rules.

   Args:
       param (``int``):
           the parameter to set; a `~.StringConfigParameter`
           value.

       desc_str (``int``):
           address of a NUL-terminated C string (``char*``)
           holding the new value.

   Raises:
       `~.cuFileError`:
           e.g. ``OpError.INVALID_VALUE`` or
           ``OpError.DRIVER_ALREADY_OPEN``.


.. py:function:: op_status_error(status: int) -> str

   Return the cuFile status string for ``status``.

   Args:
       status (``int``):
           a ``hipFileOpError`` / `~.OpError` status code.

   Returns:
       ``str``:
           a human-readable description of the status.


