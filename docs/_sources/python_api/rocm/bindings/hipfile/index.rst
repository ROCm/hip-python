rocm.bindings.hipfile
=====================

.. py:module:: rocm.bindings.hipfile


Attributes
----------

.. autoapisummary::

   rocm.bindings.hipfile.HIPFILE_VERSION_MAJOR
   rocm.bindings.hipfile.HIPFILE_VERSION_MINOR
   rocm.bindings.hipfile.HIPFILE_VERSION_PATCH
   rocm.bindings.hipfile.HIPFILE_BASE_ERR
   rocm.bindings.hipfile.hipFileOpError_t
   rocm.bindings.hipfile.hipFileError_t
   rocm.bindings.hipfile.hipFileDriverStatusFlags_t
   rocm.bindings.hipfile.hipFileDriverControlFlags_t
   rocm.bindings.hipfile.hipFileFeatureFlags_t
   rocm.bindings.hipfile.hipFileDriverProps_t
   rocm.bindings.hipfile.hipFileRDMAInfo_t
   rocm.bindings.hipfile.hipFileFSOps_t
   rocm.bindings.hipfile.hipFileFileHandleType_t
   rocm.bindings.hipfile.hipFileDescr_t
   rocm.bindings.hipfile.hipFileOpcode_t
   rocm.bindings.hipfile.hipFileStatus_t
   rocm.bindings.hipfile.hipFileBatchMode_t
   rocm.bindings.hipfile.hipFileIOParams_t
   rocm.bindings.hipfile.hipFileIOEvents_t


Classes
-------

.. autoapisummary::

   rocm.bindings.hipfile.hipFileOpError
   rocm.bindings.hipfile.hipFileError
   rocm.bindings.hipfile.hipFileDriverStatusFlags
   rocm.bindings.hipfile.hipFileDriverControlFlags
   rocm.bindings.hipfile.hipFileFeatureFlags
   rocm.bindings.hipfile.hipFileDriverProps_struct_0
   rocm.bindings.hipfile.hipFileDriverProps
   rocm.bindings.hipfile.hipFileRDMAInfo
   rocm.bindings.hipfile.hipFileFSOps_anon_funptr_0
   rocm.bindings.hipfile.hipFileFSOps_anon_funptr_1
   rocm.bindings.hipfile.hipFileFSOps_anon_funptr_2
   rocm.bindings.hipfile.hipFileFSOps_anon_funptr_3
   rocm.bindings.hipfile.hipFileFSOps_anon_funptr_4
   rocm.bindings.hipfile.hipFileFSOps
   rocm.bindings.hipfile.hipFileFileHandleType
   rocm.bindings.hipfile.hipFileDescr_union_0
   rocm.bindings.hipfile.hipFileDescr
   rocm.bindings.hipfile.hipFileOpcode
   rocm.bindings.hipfile.hipFileStatus
   rocm.bindings.hipfile.hipFileBatchMode
   rocm.bindings.hipfile.hipFileIOParams_union_0_struct_0
   rocm.bindings.hipfile.hipFileIOParams_union_0
   rocm.bindings.hipfile.hipFileIOParams
   rocm.bindings.hipfile.hipFileIOEvents
   rocm.bindings.hipfile.hipFileSizeTConfigParameter_t
   rocm.bindings.hipfile.hipFileBoolConfigParameter_t
   rocm.bindings.hipfile.hipFileStringConfigParameter_t


Functions
---------

.. autoapisummary::

   rocm.bindings.hipfile.has_symbol
   rocm.bindings.hipfile.hipFileGetOpErrorString
   rocm.bindings.hipfile.hipFileHandleRegister
   rocm.bindings.hipfile.hipFileHandleDeregister
   rocm.bindings.hipfile.hipFileBufRegister
   rocm.bindings.hipfile.hipFileBufDeregister
   rocm.bindings.hipfile.hipFileRead
   rocm.bindings.hipfile.hipFileWrite
   rocm.bindings.hipfile.hipFileDriverOpen
   rocm.bindings.hipfile.hipFileDriverClose
   rocm.bindings.hipfile.hipFileUseCount
   rocm.bindings.hipfile.hipFileDriverGetProperties
   rocm.bindings.hipfile.hipFileDriverSetPollMode
   rocm.bindings.hipfile.hipFileDriverSetMaxDirectIOSize
   rocm.bindings.hipfile.hipFileDriverSetMaxCacheSize
   rocm.bindings.hipfile.hipFileDriverSetMaxPinnedMemSize
   rocm.bindings.hipfile.hipFileBatchIOSetUp
   rocm.bindings.hipfile.hipFileBatchIOSubmit
   rocm.bindings.hipfile.hipFileBatchIOGetStatus
   rocm.bindings.hipfile.hipFileBatchIOCancel
   rocm.bindings.hipfile.hipFileBatchIODestroy
   rocm.bindings.hipfile.hipFileReadAsync
   rocm.bindings.hipfile.hipFileWriteAsync
   rocm.bindings.hipfile.hipFileStreamRegister
   rocm.bindings.hipfile.hipFileStreamDeregister
   rocm.bindings.hipfile.hipFileGetVersion
   rocm.bindings.hipfile.hipFileGetParameterSizeT
   rocm.bindings.hipfile.hipFileGetParameterBool
   rocm.bindings.hipfile.hipFileGetParameterString
   rocm.bindings.hipfile.hipFileSetParameterSizeT
   rocm.bindings.hipfile.hipFileSetParameterBool
   rocm.bindings.hipfile.hipFileSetParameterString


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:data:: HIPFILE_VERSION_MAJOR
   :type:  Any

.. py:data:: HIPFILE_VERSION_MINOR
   :type:  Any

.. py:data:: HIPFILE_VERSION_PATCH
   :type:  Any

.. py:data:: HIPFILE_BASE_ERR
   :type:  Any

.. py:class:: hipFileOpError

   Bases: :py:obj:`enum.IntEnum`


   hipFile function return codes

   An error code of -1 indicates a that a C or POSIX error has occurred and
   errno is likely to have been set.

   Note:
       HIPFILE_BASE_ERR + 21 and 32 are intentionally omitted.


   .. py:attribute:: hipFileSuccess
      :type:  int


   .. py:attribute:: hipFileDriverNotInitialized
      :type:  int


   .. py:attribute:: hipFileDriverInvalidProps
      :type:  int


   .. py:attribute:: hipFileDriverUnsupportedLimit
      :type:  int


   .. py:attribute:: hipFileDriverVersionMismatch
      :type:  int


   .. py:attribute:: hipFileDriverVersionReadError
      :type:  int


   .. py:attribute:: hipFileDriverClosing
      :type:  int


   .. py:attribute:: hipFilePlatformNotSupported
      :type:  int


   .. py:attribute:: hipFileIONotSupported
      :type:  int


   .. py:attribute:: hipFileDeviceNotSupported
      :type:  int


   .. py:attribute:: hipFileDriverError
      :type:  int


   .. py:attribute:: hipFileHipDriverError
      :type:  int


   .. py:attribute:: hipFileHipPointerInvalid
      :type:  int


   .. py:attribute:: hipFileHipMemoryTypeInvalid
      :type:  int


   .. py:attribute:: hipFileHipPointerRangeError
      :type:  int


   .. py:attribute:: hipFileHipContextMismatch
      :type:  int


   .. py:attribute:: hipFileInvalidMappingSize
      :type:  int


   .. py:attribute:: hipFileInvalidMappingRange
      :type:  int


   .. py:attribute:: hipFileInvalidFileType
      :type:  int


   .. py:attribute:: hipFileInvalidFileOpenFlag
      :type:  int


   .. py:attribute:: hipFileDIONotSet
      :type:  int


   .. py:attribute:: hipFileInvalidValue
      :type:  int


   .. py:attribute:: hipFileMemoryAlreadyRegistered
      :type:  int


   .. py:attribute:: hipFileMemoryNotRegistered
      :type:  int


   .. py:attribute:: hipFilePermissionDenied
      :type:  int


   .. py:attribute:: hipFileDriverAlreadyOpen
      :type:  int


   .. py:attribute:: hipFileHandleNotRegistered
      :type:  int


   .. py:attribute:: hipFileHandleAlreadyRegistered
      :type:  int


   .. py:attribute:: hipFileDeviceNotFound
      :type:  int


   .. py:attribute:: hipFileInternalError
      :type:  int


   .. py:attribute:: hipFileGetNewFDFailed
      :type:  int


   .. py:attribute:: hipFileDriverSetupError
      :type:  int


   .. py:attribute:: hipFileIODisabled
      :type:  int


   .. py:attribute:: hipFileBatchSubmitFailed
      :type:  int


   .. py:attribute:: hipFileGPUMemoryPinningFailed
      :type:  int


   .. py:attribute:: hipFileBatchFull
      :type:  int


   .. py:attribute:: hipFileAsyncNotSupported
      :type:  int


   .. py:attribute:: hipFileIOMaxError
      :type:  int


.. py:data:: hipFileOpError_t

.. py:function:: hipFileGetOpErrorString(status)

   Return a descriptive string for a hipFile error

   Args:
       status (:py:obj:`~.hipFileOpError`) -- *IN*:
           Return code provided by hipFile

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.bytes`: Description of the error encountered

   .. rubric:: C signature

   .. code-block:: c

       const char * hipFileGetOpErrorString(hipFileOpError_t status)


.. py:class:: hipFileError(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Error status returned from hipFile API calls

   Note:
       This struct has the `[[nodiscard]]` attribute in C++ >= 17 and
       C >= 23 so unhandled return values will generate warnings


   .. py:attribute:: err
      :type:  Any


   .. py:attribute:: hip_drv_err
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileError_t

.. py:class:: hipFileDriverStatusFlags

   Bases: :py:obj:`enum.IntEnum`


   Filesystems/storage protocols supported by GPU IO on this system

   Note:
       Value 10 is reserved for YRCloudFile


   .. py:attribute:: hipFileLustreSupported
      :type:  int


   .. py:attribute:: hipFileWekaFSSupported
      :type:  int


   .. py:attribute:: hipFileNFSSupported
      :type:  int


   .. py:attribute:: hipFileGPFSSupported
      :type:  int


   .. py:attribute:: hipFileNVMeSupported
      :type:  int


   .. py:attribute:: hipFileNVMeoFSupported
      :type:  int


   .. py:attribute:: hipFileSCSISupported
      :type:  int


   .. py:attribute:: hipFileScaleFluxCSDSupported
      :type:  int


   .. py:attribute:: hipFileNVMeshSupported
      :type:  int


   .. py:attribute:: hipFileBeeGFSSupported
      :type:  int


   .. py:attribute:: hipFileNVMeP2PSupported
      :type:  int


   .. py:attribute:: hipFileScatefsSupported
      :type:  int


.. py:data:: hipFileDriverStatusFlags_t

.. py:class:: hipFileDriverControlFlags

   Bases: :py:obj:`enum.IntEnum`


   Control flags for passing to the GPU IO driver
       


   .. py:attribute:: hipFileUsePollMode
      :type:  int


   .. py:attribute:: hipFileAllowCompatMode
      :type:  int


.. py:data:: hipFileDriverControlFlags_t

.. py:class:: hipFileFeatureFlags

   Bases: :py:obj:`enum.IntEnum`


   GPU IO Transport & Features supported by the system
       


   .. py:attribute:: hipFileDynRoutingSupported
      :type:  int


   .. py:attribute:: hipFileBatchIOSupported
      :type:  int


   .. py:attribute:: hipFileStreamsSupported
      :type:  int


   .. py:attribute:: hipFileParallelIOSupported
      :type:  int


.. py:data:: hipFileFeatureFlags_t

.. py:class:: hipFileDriverProps_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   GPU IO Driver Configuration
       


   .. py:attribute:: major_version
      :type:  Any


   .. py:attribute:: minor_version
      :type:  Any


   .. py:attribute:: poll_thresh_size
      :type:  Any


   .. py:attribute:: max_direct_io_size
      :type:  Any


   .. py:attribute:: driver_status_flags
      :type:  Any


   .. py:attribute:: driver_control_flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFileDriverProps(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   GPU IO configuration
       


   .. py:attribute:: nvfs
      :type:  Any


   .. py:attribute:: feature_flags
      :type:  Any


   .. py:attribute:: max_device_cache_size
      :type:  Any


   .. py:attribute:: per_buffer_cache_size
      :type:  Any


   .. py:attribute:: max_device_pinned_mem_size
      :type:  Any


   .. py:attribute:: max_batch_io_count
      :type:  Any


   .. py:attribute:: max_batch_io_timeout_msecs
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileDriverProps_t

.. py:class:: hipFileRDMAInfo(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Userspace RDMA configuration
       


   .. py:attribute:: version
      :type:  Any


   .. py:attribute:: desc_len
      :type:  Any


   .. py:attribute:: desc_str
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileRDMAInfo_t

.. py:class:: hipFileFSOps_anon_funptr_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Type of remote FS used. If NULL, use fstat to discover.
       


.. py:class:: hipFileFSOps_anon_funptr_1(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Get a list of host RDMA addresses. If NULL, use any address.
       


.. py:class:: hipFileFSOps_anon_funptr_2(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Get the assigned priority of a RDMA device. If -1, there is no preference.
       


.. py:class:: hipFileFSOps_anon_funptr_3(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Read from the remote filesystem. If NULL, use the Linux VFS.
       


.. py:class:: hipFileFSOps_anon_funptr_4(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Write to the remote filesystem. If NULL, use the Linux VFS.
       


.. py:class:: hipFileFSOps(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   IO operations for RDMA filesystems
       


   .. py:attribute:: fs_type
      :type:  Any


   .. py:attribute:: getRDMADeviceList
      :type:  Any


   .. py:attribute:: getRDMADevicePriority
      :type:  Any


   .. py:attribute:: read
      :type:  Any


   .. py:attribute:: write
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileFSOps_t

.. py:class:: hipFileFileHandleType

   Bases: :py:obj:`enum.IntEnum`


   Type of file handle being used
       


   .. py:attribute:: hipFileHandleTypeOpaqueFD
      :type:  int


   .. py:attribute:: hipFileHandleTypeOpaqueWin32
      :type:  int


   .. py:attribute:: hipFileHandleTypeUserspaceFS
      :type:  int


.. py:data:: hipFileFileHandleType_t

.. py:class:: hipFileDescr_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: fd
      :type:  Any


   .. py:attribute:: hFile
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFileDescr(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Top-level structure for performing GPU IO

   hipFileHandleTypeOpaqueFD    -> handle.fd non-negative, fs_ops ignored
   hipFileHandleTypeOpaqueWin32 -> handle.hFile non-NULL, fs_ops ignored
   hipFileHandleTypeUserspaceFS -> handle.fd non-negative, fs_ops non-NULL


   .. py:attribute:: type
      :type:  Any


   .. py:attribute:: handle
      :type:  Any


   .. py:attribute:: fs_ops
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileDescr_t

.. py:function:: hipFileHandleRegister(descr)

   Registers an open file for GPU IO

   Note:
       If the library has not already been initialized, the first call to
       `hipFileHandleRegister()` will initialize the library and increment
       the reference count.

   Args:
       descr (:py:obj:`~.hipFileDescr`/:py:obj:`~.object`) -- *IN*:
           Parameters for opening the file

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               \hipfile_handle_param

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileHandleRegister(hipFileHandle_t * fh, hipFileDescr_t * descr)


.. py:function:: hipFileHandleDeregister(fh)

   Deregisters a file from GPU IO

   Args:
       fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipfile_handle_param

   .. rubric:: C signature

   .. code-block:: c

       void hipFileHandleDeregister(hipFileHandle_t fh)


.. py:function:: hipFileBufRegister(buffer_base, length, flags)

   Registers a GPU memory region to be used with GPU IO

   The memory region should be allocated by the user before being passed to the API call

   Note:
       If the library has not already been initialized, the first call to
       `hipFileBufRegister()` will initialize the library and increment
       the reference count.

   Args:
       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \buffer_base_param

       length (:py:obj:`~.int`) -- *IN*:
           Size of the allocated buffer in bytes

       flags (:py:obj:`~.int`) -- *IN*:
           Control flags for this buffer

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBufRegister(const void * buffer_base, size_t length, int flags)


.. py:function:: hipFileBufDeregister(buffer_base)

   Deregisters a GPU memory region from being used with GPU IO

   Args:
       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \buffer_base_param

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBufDeregister(const void * buffer_base)


.. py:function:: hipFileRead(fh, buffer_base, size, file_offset, buffer_offset)

   Synchronously read data from a file into a GPU buffer.

   Args:
       fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           hipFile handle for the target file.

       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Base pointer of the registered GPU buffer.

       size (:py:obj:`~.int`) -- *IN*:
           Number of bytes to read.

       file_offset (:py:obj:`~.int`) -- *IN*:
           Offset into the file.

       buffer_offset (:py:obj:`~.int`) -- *IN*:
           Offset into the GPU buffer.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: The raw ``ssize_t`` result. One of:
               - if >= 0: Number of bytes transferred
               - if -1:   POSIX system error (see the ``errno`` element)
               - else:    Negated :py:obj:`~.hipFileOpError_t`; when it equals
                          ``-hipFileHipDriverError`` the HIP driver error is
                          carried in the ``hip_drv_err`` element

       * :py:obj:`~.int`: ``errno``, snapshotted inside the ``with nogil`` block
               right after the call (meaningful only when the result is -1).

       * :py:obj:`~.int`: the :py:obj:`~.hipError_t` value from
               ``hipPeekAtLastError()``, snapshotted inside the same
               ``with nogil`` block (meaningful only when the result is
               ``-hipFileHipDriverError``).


.. py:function:: hipFileWrite(fh, buffer_base, size, file_offset, buffer_offset)

   Synchronously write data from a GPU buffer to a file.

   Args:
       fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           hipFile handle for the target file.

       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Base pointer of the registered GPU buffer.

       size (:py:obj:`~.int`) -- *IN*:
           Number of bytes to write.

       file_offset (:py:obj:`~.int`) -- *IN*:
           Offset into the file.

       buffer_offset (:py:obj:`~.int`) -- *IN*:
           Offset into the GPU buffer.

   Returns:
       A :py:obj:`~.tuple` of size 3 that contains (in that order):

       * :py:obj:`~.int`: The raw ``ssize_t`` result. One of:
               - if >= 0: Number of bytes transferred
               - if -1:   POSIX system error (see the ``errno`` element)
               - else:    Negated :py:obj:`~.hipFileOpError_t`; when it equals
                          ``-hipFileHipDriverError`` the HIP driver error is
                          carried in the ``hip_drv_err`` element

       * :py:obj:`~.int`: ``errno``, snapshotted inside the ``with nogil`` block
               right after the call (meaningful only when the result is -1).

       * :py:obj:`~.int`: the :py:obj:`~.hipError_t` value from
               ``hipPeekAtLastError()``, snapshotted inside the same
               ``with nogil`` block (meaningful only when the result is
               ``-hipFileHipDriverError``).


.. py:function:: hipFileDriverOpen()

   Initialize the GPU IO driver for this process

   Each call to `hipFileDriverOpen()` increments the library's reference
   count. If a call to `hipFileDriverOpen()` results in the reference count
   transitioning from zero to one, the library's state will be initialized.

   Calling `hipFileDriverOpen()` is optional. The first call to
   `hipFileBufRegister()` or `hipFileHandleRegister()` will trigger
   library initialization and increment the library's reference count.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverOpen()


.. py:function:: hipFileDriverClose()

   Close the GPU IO driver for this process

   Each call to `hipFileDriverClose()` decrements the library's reference
   count. If a call to `hipFileDriverClose()` results in the reference count
   transitioning from one to zero, the library's state will be destroyed.

   Explicitly closing the library is not required; the library's state will be
   destroyed automatically at program exit.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverClose()


.. py:function:: hipFileUseCount()

   Obtain the current reference count for the library

   See:
       :py:obj:`~.hipFileDriverOpen`

   See:
       :py:obj:`~.hipFileDriverClose`

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.int`: The library's reference count

   .. rubric:: C signature

   .. code-block:: c

       int64_t hipFileUseCount()


.. py:function:: hipFileDriverGetProperties(props)

   Get a list of GPU IO driver properties

   \warn_not_implemented

   Args:
       props (:py:obj:`~.hipFileDriverProps`/:py:obj:`~.object`) -- *OUT*:
           See `hipFileDriverProps_t` for what properties are reported

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t * props)


.. py:function:: hipFileDriverSetPollMode(poll, poll_threshold_size)

   Enable/disable polling mode for GPU IO

   \warn_not_implemented

   Args:
       poll (:py:obj:`~.bint`) -- *IN*:
           `true` to enable polling, `false` to disable

       poll_threshold_size (:py:obj:`~.int`) -- *IN*:
           Maximum IO size (in KiB) for which polling is
           used when enabled

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverSetPollMode(_Bool poll, size_t poll_threshold_size)


.. py:function:: hipFileDriverSetMaxDirectIOSize(max_direct_io_size)

   Set the maximum IO chunk size

   \warn_not_implemented

   Args:
       max_direct_io_size (:py:obj:`~.int`) -- *IN*:
           Maximum IO chunk size (in KiB) for each IO request

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)


.. py:function:: hipFileDriverSetMaxCacheSize(max_cache_size)

   Set the maximum amount of GPU memory that can be used for bounce buffers

   \warn_not_implemented

   Args:
       max_cache_size (:py:obj:`~.int`) -- *IN*:
           Maximum GPU memory (in KiB) that can be reserved for bounce buffers

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size)


.. py:function:: hipFileDriverSetMaxPinnedMemSize(max_pinned_size)

   Set the maximum amount of GPU memory that can be pinned

   \warn_not_implemented

   Args:
       max_pinned_size (:py:obj:`~.int`) -- *IN*:
           Maximum GPU memory (in KiB) that can be pinned

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)


.. py:class:: hipFileOpcode

   Bases: :py:obj:`enum.IntEnum`


   The direction of data movement in a batch IO request
       


   .. py:attribute:: hipFileBatchRead
      :type:  int


   .. py:attribute:: hipFileBatchWrite
      :type:  int


.. py:data:: hipFileOpcode_t

.. py:class:: hipFileStatus

   Bases: :py:obj:`enum.IntEnum`


   The status of a batch IO operation
       


   .. py:attribute:: hipFileWaiting
      :type:  int


   .. py:attribute:: hipFilePending
      :type:  int


   .. py:attribute:: hipFileInvalid
      :type:  int


   .. py:attribute:: hipFileCanceled
      :type:  int


   .. py:attribute:: hipFileComplete
      :type:  int


   .. py:attribute:: hipFileTimeout
      :type:  int


   .. py:attribute:: hipFileFailed
      :type:  int


.. py:data:: hipFileStatus_t

.. py:class:: hipFileBatchMode

   Bases: :py:obj:`enum.IntEnum`


   Mode of a batch IO operation
       


   .. py:attribute:: hipFileBatch
      :type:  int


.. py:data:: hipFileBatchMode_t

.. py:class:: hipFileIOParams_union_0_struct_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: devPtr_base
      :type:  Any


   .. py:attribute:: file_offset
      :type:  Any


   .. py:attribute:: devPtr_offset
      :type:  Any


   .. py:attribute:: size
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFileIOParams_union_0(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


   .. py:attribute:: batch
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: hipFileIOParams(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Input parameters for a batch IO request
       


   .. py:attribute:: mode
      :type:  Any


   .. py:attribute:: u
      :type:  Any


   .. py:attribute:: fh
      :type:  Any


   .. py:attribute:: opcode
      :type:  Any


   .. py:attribute:: cookie
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileIOParams_t

.. py:class:: hipFileIOEvents(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Status of a batch IO operation
       


   .. py:attribute:: cookie
      :type:  Any


   .. py:attribute:: status
      :type:  Any


   .. py:attribute:: ret
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: hipFileIOEvents_t

.. py:function:: hipFileBatchIOSetUp(max_nr)

   Prepare the system to perform a batch IO operation

   Args:
       max_nr (:py:obj:`~.int`) -- *IN*:
           Maximum number of requests that can be submitted to this batch handle

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`:
               \batch_handle_param

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBatchIOSetUp(hipFileBatchHandle_t * batch_idp, unsigned int max_nr)


.. py:function:: hipFileBatchIOSubmit(batch_idp, nr, iocbp, flags)

   Enqueue a batch of IO requests for the GPU to complete asynchronously

   Args:
       batch_idp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \batch_handle_param

       nr (:py:obj:`~.int`) -- *IN*:
           Number of batch IO requests to submit

       iocbp (:py:obj:`~.hipFileIOParams`/:py:obj:`~.object`) -- *IN*:
           An array of `nr` batch IO requests to submit to the GPU.
           Data will be read into or written from the buffer specified in
           each request.

       flags (:py:obj:`~.int`) -- *IN*:
           Control Flags for the batch IO. Currently unused.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned int nr, hipFileIOParams_t * iocbp, unsigned int flags)


.. py:function:: hipFileBatchIOGetStatus(batch_idp, min_nr, nr, iocbp, timeout)

   Poll for the status of completed batch IO operations

   \warn_not_implemented

   Args:
       batch_idp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \batch_handle_param

       min_nr (:py:obj:`~.int`) -- *IN*:
           Minimum number of batch operation statuses that should be returned.
           If `timeout` is exceeded, fewer statuses may be returned.

       nr (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`) -- *IN,OUT*:
           Maximum number of batch operation statuses that can be returned.
           This is parameter is modified to return the number of statuses returned in `iocbp`.

       iocbp (:py:obj:`~.hipFileIOEvents`/:py:obj:`~.object`) -- *OUT*:
           An array containing up to `nr` statuses from the overall batch operation

       timeout (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           Maximum amount of time this function should poll for status updates

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned int min_nr, unsigned int * nr, hipFileIOEvents_t * iocbp, struct timespec * timeout)


.. py:function:: hipFileBatchIOCancel(batch_idp)

   Cancels all pending batch IO operations

   \warn_not_implemented

   Args:
       batch_idp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \batch_handle_param

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp)


.. py:function:: hipFileBatchIODestroy(batch_idp)

   Destroys the batch IO handle and frees the associated resources

   \warn_not_implemented_void

   Args:
       batch_idp (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \batch_handle_param

   .. rubric:: C signature

   .. code-block:: c

       void hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp)


.. py:function:: hipFileReadAsync(fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_read_p, stream)

   Perform an asynchronous read from a stream

   \max_io_size_note

   Args:
       fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipfile_handle_param

       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \buffer_base_param

       size_p (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN*:
           Number of bytes that should be read

       file_offset_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN*:
           Offset into the file that should be read from

       buffer_offset_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN*:
           Offset of the GPU buffer that that the data should be written to

       bytes_read_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *OUT*:
           Number of bytes read

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipstream_param. \hipstream_if_null.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileReadAsync(hipFileHandle_t fh, void * buffer_base, size_t * size_p, hoff_t * file_offset_p, hoff_t * buffer_offset_p, ssize_t * bytes_read_p, hipStream_t stream)


.. py:function:: hipFileWriteAsync(fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_written_p, stream)

   Perform an asynchronous write to a stream

   \max_io_size_note

   Args:
       fh (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipfile_handle_param

       buffer_base (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \buffer_base_param

       size_p (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`) -- *IN*:
           Number of bytes that should be written

       file_offset_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN*:
           Offset into the file that should be written to

       buffer_offset_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *IN*:
           Offset of the GPU buffer that that the data should be read from

       bytes_written_p (:py:obj:`~.rocm.bindings.util.types.PointerToInt64`/:py:obj:`~.object`) -- *OUT*:
           Number of bytes written

       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipstream_param. \hipstream_if_null.

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileWriteAsync(hipFileHandle_t fh, void * buffer_base, size_t * size_p, hoff_t * file_offset_p, hoff_t * buffer_offset_p, ssize_t * bytes_written_p, hipStream_t stream)


.. py:function:: hipFileStreamRegister(stream, flags)

   Register a stream to be used by for asynchronous GPU IO

   Args:
       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipstream_param

       flags (:py:obj:`~.int`) -- *IN*:
           Flags that can optimize stream processing if parameters
           are known/are aligned at time of request submission

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileStreamRegister(hipStream_t stream, unsigned int flags)


.. py:function:: hipFileStreamDeregister(stream)

   Deregister a stream and free the associated resources

   Args:
       stream (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`) -- *IN*:
           \hipstream_param

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileStreamDeregister(hipStream_t stream)


.. py:function:: hipFileGetVersion()

   Get the version of the hipFile library

   Note:
       Parameters can be set to NULL to ignore that part of the version

   Returns:
       A :py:obj:`~.tuple` of size 4 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.int`:
               The major version
       * :py:obj:`~.int`:
               The minor version
       * :py:obj:`~.int`:
               The patch version

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileGetVersion(unsigned int * major, unsigned int * minor, unsigned int * patch)


.. py:class:: hipFileSizeTConfigParameter_t

   Bases: :py:obj:`enum.IntEnum`


   size_t configuration parameters
       


   .. py:attribute:: hipFileParamProfileStats
      :type:  int


   .. py:attribute:: hipFileParamExecutionMaxIOQueueDepth
      :type:  int


   .. py:attribute:: hipFileParamExecutionMaxIOThreads
      :type:  int


   .. py:attribute:: hipFileParamExecutionMinIOThresholdSizeKB
      :type:  int


   .. py:attribute:: hipFileParamExecutionMaxRequestParallelism
      :type:  int


   .. py:attribute:: hipFileParamPropertiesMaxDirectIOSizeKB
      :type:  int


   .. py:attribute:: hipFileParamPropertiesMaxDeviceCacheSizeKB
      :type:  int


   .. py:attribute:: hipFileParamPropertiesPerBufferCacheSizeKB
      :type:  int


   .. py:attribute:: hipFileParamPropertiesMaxDevicePinnedMemSizeKB
      :type:  int


   .. py:attribute:: hipFileParamPropertiesIOBatchsize
      :type:  int


   .. py:attribute:: hipFileParamPollthresholdSizeKB
      :type:  int


   .. py:attribute:: hipFileParamPropertiesBatchIOTimeoutMs
      :type:  int


.. py:class:: hipFileBoolConfigParameter_t

   Bases: :py:obj:`enum.IntEnum`


   Boolean configuration parameters
       


   .. py:attribute:: hipFileParamPropertiesUsePollMode
      :type:  int


   .. py:attribute:: hipFileParamPropertiesAllowCompatMode
      :type:  int


   .. py:attribute:: hipFileParamForceCompatMode
      :type:  int


   .. py:attribute:: hipFileParamFsMiscApiCheckAggressive
      :type:  int


   .. py:attribute:: hipFileParamExecutionParallelIO
      :type:  int


   .. py:attribute:: hipFileParamProfileNvtx
      :type:  int


   .. py:attribute:: hipFileParamPropertiesAllowSystemMemory
      :type:  int


   .. py:attribute:: hipFileParamUsePcip2pdma
      :type:  int


   .. py:attribute:: hipFileParamPreferIOUring
      :type:  int


   .. py:attribute:: hipFileParamForceOdirectMode
      :type:  int


   .. py:attribute:: hipFileParamSkipTopologyDetection
      :type:  int


   .. py:attribute:: hipFileParamStreamMemopsBypass
      :type:  int


.. py:class:: hipFileStringConfigParameter_t

   Bases: :py:obj:`enum.IntEnum`


   String configuration parameters
       


   .. py:attribute:: hipFileParamLoggingLevel
      :type:  int


   .. py:attribute:: hipFileParamEnvLogfilePath
      :type:  int


   .. py:attribute:: hipFileParamLogDir
      :type:  int


.. py:function:: hipFileGetParameterSizeT(param)

   Get the value of a size_t configuration parameter

   \warn_not_implemented

   Note:
       If the driver is open, the value returned is the value currently in use by the driver.

   Note:
       If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.

   Args:
       param (:py:obj:`~.hipFileSizeTConfigParameter_t`):
           The configuration parameter

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.int`:
               The location to store the value of the configuration parameter

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t * value)


.. py:function:: hipFileGetParameterBool(param)

   Get the value of a Boolean configuration parameter

   \warn_not_implemented

   Note:
       If the driver is open, the value returned is the value currently in use by the driver.

   Note:
       If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.

   Args:
       param (:py:obj:`~.hipFileBoolConfigParameter_t`):
           The configuration parameter

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.bool`:
               The location to store the value of the configuration parameter

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param, _Bool * value)


.. py:function:: hipFileGetParameterString(param, len)

   Get the value of a string configuration parameter

   \warn_not_implemented

   Note:
       If the driver is open, the value returned is the value currently in use by the driver.

   Note:
       If the driver is closed, the value returned is the value that was last set by hipFileSetParameter*.

   Args:
       param (:py:obj:`~.hipFileStringConfigParameter_t`):
           The configuration parameter

       len (:py:obj:`~.int`):
           The length of the desc_str parameter

   Returns:
       A :py:obj:`~.tuple` of size 2 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return
       * :py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`:
               The location to store the value of the configuration parameter

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param, char * desc_str, int len)


.. py:function:: hipFileSetParameterSizeT(param, value)

   Set the value of a size_t configuration parameter

   \warn_not_implemented

   Note:
       Configuration parameter values may only be set when the driver is closed. Values are applied when the
       driver is opened.

   Args:
       param (:py:obj:`~.hipFileSizeTConfigParameter_t`):
           The configuration parameter

       value (:py:obj:`~.int`):
           The value of the configuration parameter

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)


.. py:function:: hipFileSetParameterBool(param, value)

   Set the value of a Boolean configuration parameter

   \warn_not_implemented

   Note:
       Configuration parameter values may only be set when the driver is closed. Values are applied when the
       driver is opened.

   Args:
       param (:py:obj:`~.hipFileBoolConfigParameter_t`):
           The configuration parameter

       value (:py:obj:`~.bint`):
           The value of the configuration parameter

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, _Bool value)


.. py:function:: hipFileSetParameterString(param, desc_str)

   Set the value of a string configuration parameter

   \warn_not_implemented

   Note:
       Configuration parameter values may only be set when the driver is closed. Values are applied when the
       driver is opened.

   Args:
       param (:py:obj:`~.hipFileStringConfigParameter_t`):
           The configuration parameter

       desc_str (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The value of the configuration parameter

   Returns:
       A :py:obj:`~.tuple` of size 1 that contains (in that order):

       * :py:obj:`~.hipFileError`: \hipfile_error_return

   .. rubric:: C signature

   .. code-block:: c

       hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char * desc_str)


