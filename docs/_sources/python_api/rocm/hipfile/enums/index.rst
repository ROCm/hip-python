rocm.hipfile.enums
==================

.. py:module:: rocm.hipfile.enums


Classes
-------

.. autoapisummary::

   rocm.hipfile.enums.OpError
   rocm.hipfile.enums.FileHandleType


Module Contents
---------------

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


.. py:class:: FileHandleType

   Bases: :py:obj:`enum.IntEnum`


   Python enum mirroring ``hipFileFileHandleType_t`` with friendly names.


   .. py:attribute:: OPAQUE_FD


   .. py:attribute:: OPAQUE_WIN32


   .. py:attribute:: USERSPACE_FS


