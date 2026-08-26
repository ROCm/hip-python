# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# The companion `cufile.pyi` is HAND-MAINTAINED: edit it in the same commit
# as this file. Unlike the other handcoded Cython modules, cufile is not
# registered in `HIP_PYTHON_STUBGEN_MODULES` and `all_stubs` will not
# refresh it. Its public surface is almost entirely module-level `cpdef`
# functions, which `mypy stubgen` can only render as opaque
# `cython_function_or_method` attributes — that would throw away the module
# docstring, the enum members and every typed signature that sphinx-autoapi
# renders into the API reference. See `share/design/BUILDING.md`,
# "Regenerating stubs for handcoded Cython modules".

"""``cuda.bindings.cufile`` interop layer implemented on top of hipFILE.

This is a HAND-WRITTEN Cython module (it is NOT emitted by the hip-python code
generator). It mirrors the public ``cuda.bindings.cufile`` cpdef surface
(snake_case functions, ``intptr_t`` pointer arguments, ``cuFileError`` on
failure, and the ``Descr``/``IOParams``/``IOEvents`` array helpers) so that
cuFile code can run unmodified against AMD's hipFILE. Every call is forwarded
to the corresponding hipFILE C symbol via the low-level
``cuda.bindings.cycufile`` alias module.
"""

import enum
import os

cimport cuda.bindings.cycufile as cycufile
from libc.errno cimport errno
from libc.stdint cimport int64_t, intptr_t
from libc.stdlib cimport calloc, free, malloc
from rocm.bindings.cyhip cimport hipPeekAtLastError, hipStream_t
from rocm.bindings.cyhipfile cimport timespec

# ---------------------------------------------------------------------------
# Enums. Member *names* mirror cuda-python's ``cuda.bindings.cufile`` enums
# (the ``CU_FILE_``/``CUFILE_`` prefix stripped); member *values* come from the
# hipFILE C constants, which share cuFILE's numeric layout.
# ---------------------------------------------------------------------------

class OpError(enum.IntEnum):
    """See ``hipFileOpError``."""
    SUCCESS = cycufile.hipFileSuccess
    DRIVER_NOT_INITIALIZED = cycufile.hipFileDriverNotInitialized
    DRIVER_INVALID_PROPS = cycufile.hipFileDriverInvalidProps
    DRIVER_UNSUPPORTED_LIMIT = cycufile.hipFileDriverUnsupportedLimit
    DRIVER_VERSION_MISMATCH = cycufile.hipFileDriverVersionMismatch
    DRIVER_VERSION_READ_ERROR = cycufile.hipFileDriverVersionReadError
    DRIVER_CLOSING = cycufile.hipFileDriverClosing
    PLATFORM_NOT_SUPPORTED = cycufile.hipFilePlatformNotSupported
    IO_NOT_SUPPORTED = cycufile.hipFileIONotSupported
    DEVICE_NOT_SUPPORTED = cycufile.hipFileDeviceNotSupported
    NVFS_DRIVER_ERROR = cycufile.hipFileDriverError
    CUDA_DRIVER_ERROR = cycufile.hipFileHipDriverError
    CUDA_POINTER_INVALID = cycufile.hipFileHipPointerInvalid
    CUDA_MEMORY_TYPE_INVALID = cycufile.hipFileHipMemoryTypeInvalid
    CUDA_POINTER_RANGE_ERROR = cycufile.hipFileHipPointerRangeError
    CUDA_CONTEXT_MISMATCH = cycufile.hipFileHipContextMismatch
    INVALID_MAPPING_SIZE = cycufile.hipFileInvalidMappingSize
    INVALID_MAPPING_RANGE = cycufile.hipFileInvalidMappingRange
    INVALID_FILE_TYPE = cycufile.hipFileInvalidFileType
    INVALID_FILE_OPEN_FLAG = cycufile.hipFileInvalidFileOpenFlag
    DIO_NOT_SET = cycufile.hipFileDIONotSet
    INVALID_VALUE = cycufile.hipFileInvalidValue
    MEMORY_ALREADY_REGISTERED = cycufile.hipFileMemoryAlreadyRegistered
    MEMORY_NOT_REGISTERED = cycufile.hipFileMemoryNotRegistered
    PERMISSION_DENIED = cycufile.hipFilePermissionDenied
    DRIVER_ALREADY_OPEN = cycufile.hipFileDriverAlreadyOpen
    HANDLE_NOT_REGISTERED = cycufile.hipFileHandleNotRegistered
    HANDLE_ALREADY_REGISTERED = cycufile.hipFileHandleAlreadyRegistered
    DEVICE_NOT_FOUND = cycufile.hipFileDeviceNotFound
    INTERNAL_ERROR = cycufile.hipFileInternalError
    GETNEWFD_FAILED = cycufile.hipFileGetNewFDFailed
    NVFS_SETUP_ERROR = cycufile.hipFileDriverSetupError
    IO_DISABLED = cycufile.hipFileIODisabled
    BATCH_SUBMIT_FAILED = cycufile.hipFileBatchSubmitFailed
    GPU_MEMORY_PINNING_FAILED = cycufile.hipFileGPUMemoryPinningFailed
    BATCH_FULL = cycufile.hipFileBatchFull
    ASYNC_NOT_SUPPORTED = cycufile.hipFileAsyncNotSupported
    IO_MAX_ERROR = cycufile.hipFileIOMaxError


class DriverStatusFlags(enum.IntEnum):
    """See ``hipFileDriverStatusFlags_t``."""
    LUSTRE_SUPPORTED = cycufile.hipFileLustreSupported
    WEKAFS_SUPPORTED = cycufile.hipFileWekaFSSupported
    NFS_SUPPORTED = cycufile.hipFileNFSSupported
    GPFS_SUPPORTED = cycufile.hipFileGPFSSupported
    NVME_SUPPORTED = cycufile.hipFileNVMeSupported
    NVMEOF_SUPPORTED = cycufile.hipFileNVMeoFSupported
    SCSI_SUPPORTED = cycufile.hipFileSCSISupported
    SCALEFLUX_CSD_SUPPORTED = cycufile.hipFileScaleFluxCSDSupported
    NVMESH_SUPPORTED = cycufile.hipFileNVMeshSupported
    BEEGFS_SUPPORTED = cycufile.hipFileBeeGFSSupported
    NVME_P2P_SUPPORTED = cycufile.hipFileNVMeP2PSupported
    SCATEFS_SUPPORTED = cycufile.hipFileScatefsSupported


class DriverControlFlags(enum.IntEnum):
    """See ``hipFileDriverControlFlags_t``."""
    USE_POLL_MODE = cycufile.hipFileUsePollMode
    ALLOW_COMPAT_MODE = cycufile.hipFileAllowCompatMode


class FeatureFlags(enum.IntEnum):
    """See ``hipFileFeatureFlags_t``."""
    DYN_ROUTING_SUPPORTED = cycufile.hipFileDynRoutingSupported
    BATCH_IO_SUPPORTED = cycufile.hipFileBatchIOSupported
    STREAMS_SUPPORTED = cycufile.hipFileStreamsSupported
    PARALLEL_IO_SUPPORTED = cycufile.hipFileParallelIOSupported


class FileHandleType(enum.IntEnum):
    """See ``hipFileFileHandleType``."""
    OPAQUE_FD = cycufile.hipFileHandleTypeOpaqueFD
    OPAQUE_WIN32 = cycufile.hipFileHandleTypeOpaqueWin32
    USERSPACE_FS = cycufile.hipFileHandleTypeUserspaceFS


class Opcode(enum.IntEnum):
    """See ``hipFileOpcode_t``."""
    READ = cycufile.hipFileBatchRead
    WRITE = cycufile.hipFileBatchWrite


class Status(enum.IntEnum):
    """See ``hipFileStatus_t``."""
    WAITING = cycufile.hipFileWaiting
    PENDING = cycufile.hipFilePending
    INVALID = cycufile.hipFileInvalid
    CANCELED = cycufile.hipFileCanceled
    COMPLETE = cycufile.hipFileComplete
    TIMEOUT = cycufile.hipFileTimeout
    FAILED = cycufile.hipFileFailed


class BatchMode(enum.IntEnum):
    """See ``hipFileBatchMode_t``."""
    BATCH = cycufile.hipFileBatch


class SizeTConfigParameter(enum.IntEnum):
    """See ``hipFileSizeTConfigParameter_t``."""
    PROFILE_STATS = cycufile.hipFileParamProfileStats
    EXECUTION_MAX_IO_QUEUE_DEPTH = cycufile.hipFileParamExecutionMaxIOQueueDepth
    EXECUTION_MAX_IO_THREADS = cycufile.hipFileParamExecutionMaxIOThreads
    EXECUTION_MIN_IO_THRESHOLD_SIZE_KB = cycufile.hipFileParamExecutionMinIOThresholdSizeKB
    EXECUTION_MAX_REQUEST_PARALLELISM = cycufile.hipFileParamExecutionMaxRequestParallelism
    PROPERTIES_MAX_DIRECT_IO_SIZE_KB = cycufile.hipFileParamPropertiesMaxDirectIOSizeKB
    PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB = cycufile.hipFileParamPropertiesMaxDeviceCacheSizeKB
    PROPERTIES_PER_BUFFER_CACHE_SIZE_KB = cycufile.hipFileParamPropertiesPerBufferCacheSizeKB
    PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB = cycufile.hipFileParamPropertiesMaxDevicePinnedMemSizeKB
    PROPERTIES_IO_BATCHSIZE = cycufile.hipFileParamPropertiesIOBatchsize
    POLLTHRESHOLD_SIZE_KB = cycufile.hipFileParamPollthresholdSizeKB
    PROPERTIES_BATCH_IO_TIMEOUT_MS = cycufile.hipFileParamPropertiesBatchIOTimeoutMs


class BoolConfigParameter(enum.IntEnum):
    """See ``hipFileBoolConfigParameter_t``."""
    PROPERTIES_USE_POLL_MODE = cycufile.hipFileParamPropertiesUsePollMode
    PROPERTIES_ALLOW_COMPAT_MODE = cycufile.hipFileParamPropertiesAllowCompatMode
    FORCE_COMPAT_MODE = cycufile.hipFileParamForceCompatMode
    FS_MISC_API_CHECK_AGGRESSIVE = cycufile.hipFileParamFsMiscApiCheckAggressive
    EXECUTION_PARALLEL_IO = cycufile.hipFileParamExecutionParallelIO
    PROFILE_NVTX = cycufile.hipFileParamProfileNvtx
    PROPERTIES_ALLOW_SYSTEM_MEMORY = cycufile.hipFileParamPropertiesAllowSystemMemory
    USE_PCIP2PDMA = cycufile.hipFileParamUsePcip2pdma
    PREFER_IO_URING = cycufile.hipFileParamPreferIOUring
    FORCE_ODIRECT_MODE = cycufile.hipFileParamForceOdirectMode
    SKIP_TOPOLOGY_DETECTION = cycufile.hipFileParamSkipTopologyDetection
    STREAM_MEMOPS_BYPASS = cycufile.hipFileParamStreamMemopsBypass


class StringConfigParameter(enum.IntEnum):
    """See ``hipFileStringConfigParameter_t``."""
    LOGGING_LEVEL = cycufile.hipFileParamLoggingLevel
    ENV_LOGFILE_PATH = cycufile.hipFileParamEnvLogfilePath
    LOG_DIR = cycufile.hipFileParamLogDir


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class cuFileError(Exception):
    """Raised when a cuFile operation returns a non-``SUCCESS`` status.

    Args:
        status:
            the ``hipFileOpError`` / `~.OpError` status code.

        cu_err:
            for ``OpError.CUDA_DRIVER_ERROR`` this carries the underlying HIP
            driver error code (``hipError_t``); ``None`` otherwise.
    """

    def __init__(self, status, cu_err=None):
        self.status = status
        self.cu_err = cu_err
        try:
            name = OpError(status).name
        except ValueError:
            name = str(status)
        message = op_status_error(int(status))
        text = f"{name} ({int(status)}): {message}"
        if cu_err is not None:
            text += f" [cu_err={cu_err}]"
        super().__init__(text)


cdef inline int _check(cycufile.hipFileError err) except -1:
    if <int>err.err != <int>cycufile.hipFileSuccess:
        raise cuFileError(<int>err.err, <int>err.hip_drv_err)
    return 0


# ---------------------------------------------------------------------------
# Array helper types (Descr / IOParams / IOEvents)
# ---------------------------------------------------------------------------

cdef class _DescrHandle:
    """Proxy for the ``handle`` union of a ``Descr`` element."""
    cdef Descr _parent
    cdef size_t _idx

    def __cinit__(self, Descr parent, size_t idx):
        self._parent = parent
        self._idx = idx

    @property
    def fd(self):
        return self._parent._ptr[self._idx].handle.fd

    @fd.setter
    def fd(self, int value):
        self._parent._ptr[self._idx].handle.fd = value

    @property
    def handle(self):
        return <intptr_t>self._parent._ptr[self._idx].handle.hFile

    @handle.setter
    def handle(self, intptr_t value):
        self._parent._ptr[self._idx].handle.hFile = <void*>value


cdef class Descr:
    """Empty-initialize an array of ``hipFileDescr_t``.

    A ``hipFileDescr_t`` carries the OS-neutral file identity handed to
    `~.handle_register`: a ``type`` (a `~.FileHandleType`), a ``handle``
    union (the Linux ``fd`` or a Windows handle), and an optional ``fs_ops``
    table. Element ``0`` is exposed directly through the ``type`` / ``handle`` /
    ``fs_ops`` properties; use ``descr[i]`` to view any other element. ``ptr``
    yields the base C address to hand to the cuFile calls.

    Args:
        size (``int``):
            the number of contiguous elements to allocate (default 1).
            ``Descr(None)`` creates an unbacked view populated internally by
            ``__getitem__``.
    """

    def __cinit__(self, size=1):
        self._ptr = NULL
        self._n = 0
        self._owner = False
        if size is None:
            return
        cdef size_t n = <size_t>size
        if n == 0:
            return
        self._ptr = <cycufile.hipFileDescr*>calloc(n, sizeof(cycufile.hipFileDescr))
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        self._owner = True

    def __dealloc__(self):
        if self._owner and self._ptr != NULL:
            free(self._ptr)

    def __len__(self):
        return self._n

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or <size_t>idx >= self._n:
            raise IndexError("Descr index out of range")
        cdef Descr view = Descr(None)
        view._ptr = &self._ptr[idx]
        view._n = 1
        view._owner = False
        return view

    @property
    def ptr(self):
        return <intptr_t>self._ptr

    @property
    def type(self):
        return <int>self._ptr[0].type

    @type.setter
    def type(self, int value):
        self._ptr[0].type = <cycufile.hipFileFileHandleType>value

    @property
    def handle(self):
        return _DescrHandle(self, 0)

    @property
    def fs_ops(self):
        return <intptr_t>self._ptr[0].fs_ops

    @fs_ops.setter
    def fs_ops(self, intptr_t value):
        self._ptr[0].fs_ops = <const cycufile.hipFileFSOps*>value


cdef class _IOParamsBatch:
    """Proxy for the ``u.batch`` struct of an ``IOParams`` element."""
    cdef IOParams _parent
    cdef size_t _idx

    def __cinit__(self, IOParams parent, size_t idx):
        self._parent = parent
        self._idx = idx

    @property
    def devPtr_base(self):
        return <intptr_t>self._parent._ptr[self._idx].u.batch.devPtr_base

    @devPtr_base.setter
    def devPtr_base(self, intptr_t value):
        self._parent._ptr[self._idx].u.batch.devPtr_base = <void*>value

    @property
    def file_offset(self):
        return self._parent._ptr[self._idx].u.batch.file_offset

    @file_offset.setter
    def file_offset(self, int64_t value):
        self._parent._ptr[self._idx].u.batch.file_offset = value

    @property
    def devPtr_offset(self):
        return self._parent._ptr[self._idx].u.batch.devPtr_offset

    @devPtr_offset.setter
    def devPtr_offset(self, int64_t value):
        self._parent._ptr[self._idx].u.batch.devPtr_offset = value

    @property
    def size(self):
        return self._parent._ptr[self._idx].u.batch.size

    @size.setter
    def size(self, size_t value):
        self._parent._ptr[self._idx].u.batch.size = value


cdef class _IOParamsU:
    """Proxy for the ``u`` union of an ``IOParams`` element."""
    cdef IOParams _parent
    cdef size_t _idx

    def __cinit__(self, IOParams parent, size_t idx):
        self._parent = parent
        self._idx = idx

    @property
    def batch(self):
        return _IOParamsBatch(self._parent, self._idx)


cdef class IOParams:
    """Empty-initialize an array of ``hipFileIOParams_t``.

    Each ``hipFileIOParams_t`` describes one request in a batch submitted with
    `~.batch_io_submit`: the ``mode`` (a `~.BatchMode`), the file
    handle ``fh``, the ``opcode`` (a `~.Opcode`), an opaque ``cookie``, and
    the per-request ``u.batch`` fields (device pointer base/offset, file offset
    and size). Element ``0`` is exposed directly through the properties; use
    ``params[i]`` to view any other element, and ``ptr`` for the base C address.

    Args:
        size (``int``):
            the number of contiguous elements to allocate (default 1).
            ``IOParams(None)`` creates an unbacked view populated internally by
            ``__getitem__``.
    """

    def __cinit__(self, size=1):
        self._ptr = NULL
        self._n = 0
        self._owner = False
        if size is None:
            return
        cdef size_t n = <size_t>size
        if n == 0:
            return
        self._ptr = <cycufile.hipFileIOParams*>calloc(n, sizeof(cycufile.hipFileIOParams))
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        self._owner = True

    def __dealloc__(self):
        if self._owner and self._ptr != NULL:
            free(self._ptr)

    def __len__(self):
        return self._n

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or <size_t>idx >= self._n:
            raise IndexError("IOParams index out of range")
        cdef IOParams view = IOParams(None)
        view._ptr = &self._ptr[idx]
        view._n = 1
        view._owner = False
        return view

    @property
    def ptr(self):
        return <intptr_t>self._ptr

    @property
    def mode(self):
        return <int>self._ptr[0].mode

    @mode.setter
    def mode(self, int value):
        self._ptr[0].mode = <cycufile.hipFileBatchMode>value

    @property
    def u(self):
        return _IOParamsU(self, 0)

    @property
    def fh(self):
        return <intptr_t>self._ptr[0].fh

    @fh.setter
    def fh(self, intptr_t value):
        self._ptr[0].fh = <void*>value

    @property
    def opcode(self):
        return <int>self._ptr[0].opcode

    @opcode.setter
    def opcode(self, int value):
        self._ptr[0].opcode = <cycufile.hipFileOpcode>value

    @property
    def cookie(self):
        return <intptr_t>self._ptr[0].cookie

    @cookie.setter
    def cookie(self, intptr_t value):
        self._ptr[0].cookie = <void*>value


cdef class IOEvents:
    """Empty-initialize an array of ``hipFileIOEvents_t``.

    Each ``hipFileIOEvents_t`` receives the outcome of one batch request from
    `~.batch_io_get_status`: the request ``cookie``, the ``status`` (a
    `~.Status`), and ``ret`` (the bytes transacted, valid only once the
    request has completed successfully). Element ``0`` is exposed directly
    through the properties; use ``events[i]`` to view any other element, and
    ``ptr`` for the base C address to hand to `~.batch_io_get_status`.

    Args:
        size (``int``):
            the number of contiguous elements to allocate (default 1).
            ``IOEvents(None)`` creates an unbacked view populated internally by
            ``__getitem__``.
    """

    def __cinit__(self, size=1):
        self._ptr = NULL
        self._n = 0
        self._owner = False
        if size is None:
            return
        cdef size_t n = <size_t>size
        if n == 0:
            return
        self._ptr = <cycufile.hipFileIOEvents*>calloc(n, sizeof(cycufile.hipFileIOEvents))
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        self._owner = True

    def __dealloc__(self):
        if self._owner and self._ptr != NULL:
            free(self._ptr)

    def __len__(self):
        return self._n

    def __getitem__(self, Py_ssize_t idx):
        if idx < 0 or <size_t>idx >= self._n:
            raise IndexError("IOEvents index out of range")
        cdef IOEvents view = IOEvents(None)
        view._ptr = &self._ptr[idx]
        view._n = 1
        view._owner = False
        return view

    @property
    def ptr(self):
        return <intptr_t>self._ptr

    @property
    def cookie(self):
        return <intptr_t>self._ptr[0].cookie

    @cookie.setter
    def cookie(self, intptr_t value):
        self._ptr[0].cookie = <void*>value

    @property
    def status(self):
        return <int>self._ptr[0].status

    @status.setter
    def status(self, int value):
        self._ptr[0].status = <cycufile.hipFileStatus>value

    @property
    def ret(self):
        return self._ptr[0].ret

    @ret.setter
    def ret(self, size_t value):
        self._ptr[0].ret = value


# ---------------------------------------------------------------------------
# Driver management
# ---------------------------------------------------------------------------

cpdef driver_open():
    """Initialize the cuFile library and open the driver.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverOpen()
    _check(err)


cpdef driver_close():
    """Reset the cuFile library and release the driver.

    Closes the driver session and frees the associated resources. Any
    buffers still registered via `~.buf_register` are implicitly
    deregistered, and any in-flight IO receives an error. The driver may be
    reopened afterwards; this cleanup also happens implicitly on process exit.

    Raises:
        `~.cuFileError`:
            if the driver was not initialized
            (``OpError.DRIVER_NOT_INITIALIZED``).
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverClose()
    _check(err)


cpdef use_count():
    """Return the process-wide cuFile driver use count.

    Returns:
        ``int``:
            the number of times the cuFile driver is currently in use by
            this process at the moment of the call.
    """
    cdef int64_t count
    with nogil:
        count = cycufile.hipFileUseCount()
    return count


cpdef driver_get_properties(intptr_t props):
    """Get the driver session properties.

    If the driver is not open, the staged/default properties are returned;
    otherwise the current properties are returned. The structure reports the
    driver capabilities (supported filesystems, poll/compat control flags,
    feature flags, and the IO/cache/pinned-memory size limits).

    Args:
        props (``int``):
            address (as a Python integer) of a caller-allocated
            ``hipFileDriverProps_t`` structure to fill in.

    Raises:
        `~.cuFileError`:
            e.g. ``OpError.DRIVER_NOT_INITIALIZED``,
            ``OpError.DRIVER_VERSION_MISMATCH``, or ``OpError.INVALID_VALUE``
            if ``props`` is invalid.
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverGetProperties(<cycufile.hipFileDriverProps*>props)
    _check(err)


cpdef driver_set_poll_mode(bint poll, size_t poll_threshold_size):
    """Set whether the Read/Write APIs use polling to do IO operations.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverSetPollMode(poll, poll_threshold_size)
    _check(err)


cpdef driver_set_max_direct_io_size(size_t max_direct_io_size):
    """Set the max direct IO size used to talk to the driver.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverSetMaxDirectIOSize(max_direct_io_size)
    _check(err)


cpdef driver_set_max_cache_size(size_t max_cache_size):
    """Set the max GPU memory reserved per device for internal buffering.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverSetMaxCacheSize(max_cache_size)
    _check(err)


cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size):
    """Set the max buffer space that is pinned for ``buf_register``.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileDriverSetMaxPinnedMemSize(max_pinned_size)
    _check(err)


# ---------------------------------------------------------------------------
# Handle / buffer registration
# ---------------------------------------------------------------------------

cpdef intptr_t handle_register(intptr_t descr) except? 0:
    """Register an open file for GPU IO.

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
    """
    cdef cycufile.hipFileHandle_t fh = NULL
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileHandleRegister(<void**>&fh, <cycufile.hipFileDescr*>descr)
    _check(err)
    return <intptr_t>fh


cpdef handle_deregister(intptr_t fh):
    """Release a registered file handle from cuFile.

    Frees the cuFile resources claimed by `~.handle_register`. Call this
    only after ensuring no IO is outstanding on the handle (otherwise the
    behavior is undefined). The underlying file descriptor is *not* closed; the
    caller must still ``os.close`` it.

    Args:
        fh (``int``):
            the file handle (as a Python integer) returned by
            `~.handle_register`.
    """
    with nogil:
        cycufile.hipFileHandleDeregister(<void*>fh)


cpdef buf_register(intptr_t buf_ptr_base, size_t length, int flags):
    """Register a device/host memory region with cuFile for GPU IO.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBufRegister(<const void*>buf_ptr_base, length, flags)
    _check(err)


cpdef buf_deregister(intptr_t buf_ptr_base):
    """Deregister a device/host memory region from cuFile.

    Releases the pinned-memory mappings created by `~.buf_register`.

    Args:
        buf_ptr_base (``int``):
            the base address (as a Python integer) that was
            passed to `~.buf_register`.

    Raises:
        `~.cuFileError`:
            e.g. ``OpError.MEMORY_NOT_REGISTERED`` if ``buf_ptr_base``
            was not registered.
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBufDeregister(<const void*>buf_ptr_base)
    _check(err)


# ---------------------------------------------------------------------------
# Synchronous IO
# ---------------------------------------------------------------------------

cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, int64_t file_offset, int64_t buf_ptr_offset):
    """Read from a registered file handle into device/host memory.

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
    """
    cdef ssize_t retval
    cdef int err_no
    cdef int hip_drv_err
    cdef int status
    with nogil:
        retval = cycufile.hipFileRead(<void*>fh, <void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
        err_no = errno
        hip_drv_err = <int>hipPeekAtLastError()
    if retval >= 0:
        return retval
    if retval == -1:
        raise OSError(err_no, os.strerror(err_no))
    status = <int>(-retval)
    raise cuFileError(status, hip_drv_err if status == <int>OpError.CUDA_DRIVER_ERROR else None)


cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, int64_t file_offset, int64_t buf_ptr_offset):
    """Write device/host memory to a registered file handle.

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
    """
    cdef ssize_t retval
    cdef int err_no
    cdef int hip_drv_err
    cdef int status
    with nogil:
        retval = cycufile.hipFileWrite(<void*>fh, <const void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
        err_no = errno
        hip_drv_err = <int>hipPeekAtLastError()
    if retval >= 0:
        return retval
    if retval == -1:
        raise OSError(err_no, os.strerror(err_no))
    status = <int>(-retval)
    raise cuFileError(status, hip_drv_err if status == <int>OpError.CUDA_DRIVER_ERROR else None)


# ---------------------------------------------------------------------------
# Batch IO
# ---------------------------------------------------------------------------

cpdef intptr_t batch_io_set_up(unsigned int nr) except? 0:
    """Prepare a batch IO operation.

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
    """
    cdef cycufile.hipFileBatchHandle_t handle = NULL
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBatchIOSetUp(<void**>&handle, nr)
    _check(err)
    return <intptr_t>handle


cpdef batch_io_submit(intptr_t batch_idp, unsigned int nr, intptr_t iocbp, unsigned int flags):
    """Enqueue a batch of IO requests.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBatchIOSubmit(<void*>batch_idp, nr, <cycufile.hipFileIOParams*>iocbp, flags)
    _check(err)


cpdef batch_io_get_status(intptr_t batch_idp, unsigned int min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout):
    """Poll for the status of completed batch IO operations.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBatchIOGetStatus(
            <void*>batch_idp,
            min_nr,
            <unsigned int*>nr,
            <cycufile.hipFileIOEvents*>iocbp,
            <timespec*>timeout,
        )
    _check(err)


cpdef batch_io_cancel(intptr_t batch_idp):
    """Cancel all pending batch IO operations.

    Attempts to cancel the in-flight IOs for the batch; there is no guarantee
    an already-executing IO can be canceled. Canceled IOs report
    ``Status.CANCELED`` via `~.batch_io_get_status`.

    Args:
        batch_idp (``int``):
            the batch handle from `~.batch_io_set_up`.

    Raises:
        `~.cuFileError`:
            e.g. ``OpError.INVALID_VALUE`` on failure.
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileBatchIOCancel(<void*>batch_idp)
    _check(err)


cpdef batch_io_destroy(intptr_t batch_idp):
    """Destroy the batch IO handle and free the associated resources.

    Destroys the batch context and the resources allocated by
    `~.batch_io_set_up`.

    Args:
        batch_idp (``int``):
            the batch handle from `~.batch_io_set_up`.
    """
    with nogil:
        cycufile.hipFileBatchIODestroy(<void*>batch_idp)


# ---------------------------------------------------------------------------
# Asynchronous IO / streams
# ---------------------------------------------------------------------------

cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream):
    """Enqueue an asynchronous read on ``stream``.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileReadAsync(
            <void*>fh,
            <void*>buf_ptr_base,
            <size_t*>size_p,
            <int64_t*>file_offset_p,
            <int64_t*>buf_ptr_offset_p,
            <ssize_t*>bytes_read_p,
            <hipStream_t><void*>stream,
        )
    _check(err)


cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream):
    """Enqueue an asynchronous write on ``stream``.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileWriteAsync(
            <void*>fh,
            <void*>buf_ptr_base,
            <size_t*>size_p,
            <int64_t*>file_offset_p,
            <int64_t*>buf_ptr_offset_p,
            <ssize_t*>bytes_written_p,
            <hipStream_t><void*>stream,
        )
    _check(err)


cpdef stream_register(intptr_t stream, unsigned int flags):
    """Register a stream for asynchronous GPU IO.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileStreamRegister(<hipStream_t><void*>stream, flags)
    _check(err)


cpdef stream_deregister(intptr_t stream):
    """Deregister a stream and free the associated resources.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileStreamDeregister(<hipStream_t><void*>stream)
    _check(err)


# ---------------------------------------------------------------------------
# Version / parameters / error string
# ---------------------------------------------------------------------------

cpdef int get_version() except? 0:
    """Return the cuFile library version as a packed integer.

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
    """
    cdef unsigned int major = 0
    cdef unsigned int minor = 0
    cdef unsigned int patch = 0
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileGetVersion(&major, &minor, &patch)
    _check(err)
    return <int>(major * 1000 + minor * 10 + patch)


cpdef get_parameter_size_t(int param):
    """Get the value of a ``size_t`` configuration parameter.

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
    """
    cdef size_t value = 0
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileGetParameterSizeT(<cycufile.hipFileSizeTConfigParameter_t>param, &value)
    _check(err)
    return value


cpdef get_parameter_bool(int param):
    """Get the value of a Boolean configuration parameter.

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
    """
    cdef bint value = 0
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileGetParameterBool(<cycufile.hipFileBoolConfigParameter_t>param, &value)
    _check(err)
    return bool(value)


cpdef str get_parameter_string(int param, int len):
    """Get the value of a string configuration parameter.

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
    """
    cdef char* buffer = <char*>malloc(len)
    if buffer == NULL:
        raise MemoryError()
    cdef cycufile.hipFileError err
    try:
        with nogil:
            err = cycufile.hipFileGetParameterString(<cycufile.hipFileStringConfigParameter_t>param, buffer, len)
        _check(err)
        return buffer.decode("utf-8")
    finally:
        free(buffer)


cpdef set_parameter_size_t(int param, size_t value):
    """Set the value of a ``size_t`` configuration parameter.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileSetParameterSizeT(<cycufile.hipFileSizeTConfigParameter_t>param, value)
    _check(err)


cpdef set_parameter_bool(int param, bint value):
    """Set the value of a Boolean configuration parameter.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileSetParameterBool(<cycufile.hipFileBoolConfigParameter_t>param, value)
    _check(err)


cpdef set_parameter_string(int param, intptr_t desc_str):
    """Set the value of a string configuration parameter.

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
    """
    cdef cycufile.hipFileError err
    with nogil:
        err = cycufile.hipFileSetParameterString(<cycufile.hipFileStringConfigParameter_t>param, <const char*>desc_str)
    _check(err)


cpdef str op_status_error(int status):
    """Return the cuFile status string for ``status``.

    Args:
        status (``int``):
            a ``hipFileOpError`` / `~.OpError` status code.

    Returns:
        ``str``:
            a human-readable description of the status.
    """
    cdef const char* s
    with nogil:
        s = cycufile.hipFileGetOpErrorString(<cycufile.hipFileOpError>status)
    if s == NULL:
        return ""
    return s.decode("utf-8")
