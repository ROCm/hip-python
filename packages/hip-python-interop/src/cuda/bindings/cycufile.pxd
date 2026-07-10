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

# C-level (``cimport``-only) declarations for the ``cuda.bindings.cufile``
# interop layer.
#
# This is a HAND-WRITTEN file (it is NOT emitted by the hip-python code
# generator). It aliases the public cuFile C API onto the auto-generated
# ``rocm.bindings.cyhipfile`` module the same way ``cydriver.pxd`` /
# ``cyruntime.pxd`` alias the HIP runtime. hipFILE is a near-verbatim clone
# of cuFILE (identical struct layouts and identical enumerator *values*), so
# every cuFile symbol below is a straight re-export of its hipFILE twin.
#
# For maximum flexibility every symbol is re-exported under BOTH names:
#   * the original ``hipFile*`` / ``HIPFILE_*`` name, and
#   * the ``cuFile*`` / ``CU_FILE_*`` / ``CUFILE_*`` alias
# so that Cython consumers can ``cimport`` either spelling. The actual symbol
# that is linked in every case is the hipFILE one (resolved through the
# ``rocm.bindings.cyhipfile`` runtime lazy-loader).

# ---------------------------------------------------------------------------
# Version / base-error constants
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    HIPFILE_VERSION_MAJOR,
    HIPFILE_VERSION_MINOR,
    HIPFILE_VERSION_PATCH,
    HIPFILE_BASE_ERR,
)

# ---------------------------------------------------------------------------
# Enums: types (both ``hipFile*`` and ``hipFile*_t`` are kept, plus the cuFile
# alias). Enumerator constants follow further below.
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileOpError,
    hipFileOpError_t,
    hipFileOpError as CUfileOpError,
    hipFileOpError_t as CUfileOpError_t,
    hipFileDriverStatusFlags,
    hipFileDriverStatusFlags_t,
    hipFileDriverStatusFlags as CUfileDriverStatusFlags,
    hipFileDriverStatusFlags_t as CUfileDriverStatusFlags_t,
    hipFileDriverControlFlags,
    hipFileDriverControlFlags_t,
    hipFileDriverControlFlags as CUfileDriverControlFlags,
    hipFileDriverControlFlags_t as CUfileDriverControlFlags_t,
    hipFileFeatureFlags,
    hipFileFeatureFlags_t,
    hipFileFeatureFlags as CUfileFeatureFlags,
    hipFileFeatureFlags_t as CUfileFeatureFlags_t,
    hipFileFileHandleType,
    hipFileFileHandleType_t,
    hipFileFileHandleType as CUfileFileHandleType,
    hipFileFileHandleType_t as CUfileFileHandleType_t,
    hipFileOpcode,
    hipFileOpcode_t,
    hipFileOpcode as CUfileOpcode,
    hipFileOpcode_t as CUfileOpcode_t,
    hipFileStatus,
    hipFileStatus_t,
    hipFileStatus as CUfileStatus,
    hipFileStatus_t as CUfileStatus_t,
    hipFileBatchMode,
    hipFileBatchMode_t,
    hipFileBatchMode as CUfileBatchMode,
    hipFileBatchMode_t as CUfileBatchMode_t,
    hipFileSizeTConfigParameter_t,
    hipFileSizeTConfigParameter_t as CUFileSizeTConfigParameter_t,
    hipFileBoolConfigParameter_t,
    hipFileBoolConfigParameter_t as CUFileBoolConfigParameter_t,
    hipFileStringConfigParameter_t,
    hipFileStringConfigParameter_t as CUFileStringConfigParameter_t,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileOpError  <->  CU_FILE_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileSuccess,
    hipFileSuccess as CU_FILE_SUCCESS,
    hipFileDriverNotInitialized,
    hipFileDriverNotInitialized as CU_FILE_DRIVER_NOT_INITIALIZED,
    hipFileDriverInvalidProps,
    hipFileDriverInvalidProps as CU_FILE_DRIVER_INVALID_PROPS,
    hipFileDriverUnsupportedLimit,
    hipFileDriverUnsupportedLimit as CU_FILE_DRIVER_UNSUPPORTED_LIMIT,
    hipFileDriverVersionMismatch,
    hipFileDriverVersionMismatch as CU_FILE_DRIVER_VERSION_MISMATCH,
    hipFileDriverVersionReadError,
    hipFileDriverVersionReadError as CU_FILE_DRIVER_VERSION_READ_ERROR,
    hipFileDriverClosing,
    hipFileDriverClosing as CU_FILE_DRIVER_CLOSING,
    hipFilePlatformNotSupported,
    hipFilePlatformNotSupported as CU_FILE_PLATFORM_NOT_SUPPORTED,
    hipFileIONotSupported,
    hipFileIONotSupported as CU_FILE_IO_NOT_SUPPORTED,
    hipFileDeviceNotSupported,
    hipFileDeviceNotSupported as CU_FILE_DEVICE_NOT_SUPPORTED,
    hipFileDriverError,
    hipFileDriverError as CU_FILE_NVFS_DRIVER_ERROR,
    hipFileHipDriverError,
    hipFileHipDriverError as CU_FILE_CUDA_DRIVER_ERROR,
    hipFileHipPointerInvalid,
    hipFileHipPointerInvalid as CU_FILE_CUDA_POINTER_INVALID,
    hipFileHipMemoryTypeInvalid,
    hipFileHipMemoryTypeInvalid as CU_FILE_CUDA_MEMORY_TYPE_INVALID,
    hipFileHipPointerRangeError,
    hipFileHipPointerRangeError as CU_FILE_CUDA_POINTER_RANGE_ERROR,
    hipFileHipContextMismatch,
    hipFileHipContextMismatch as CU_FILE_CUDA_CONTEXT_MISMATCH,
    hipFileInvalidMappingSize,
    hipFileInvalidMappingSize as CU_FILE_INVALID_MAPPING_SIZE,
    hipFileInvalidMappingRange,
    hipFileInvalidMappingRange as CU_FILE_INVALID_MAPPING_RANGE,
    hipFileInvalidFileType,
    hipFileInvalidFileType as CU_FILE_INVALID_FILE_TYPE,
    hipFileInvalidFileOpenFlag,
    hipFileInvalidFileOpenFlag as CU_FILE_INVALID_FILE_OPEN_FLAG,
    hipFileDIONotSet,
    hipFileDIONotSet as CU_FILE_DIO_NOT_SET,
    hipFileInvalidValue,
    hipFileInvalidValue as CU_FILE_INVALID_VALUE,
    hipFileMemoryAlreadyRegistered,
    hipFileMemoryAlreadyRegistered as CU_FILE_MEMORY_ALREADY_REGISTERED,
    hipFileMemoryNotRegistered,
    hipFileMemoryNotRegistered as CU_FILE_MEMORY_NOT_REGISTERED,
    hipFilePermissionDenied,
    hipFilePermissionDenied as CU_FILE_PERMISSION_DENIED,
    hipFileDriverAlreadyOpen,
    hipFileDriverAlreadyOpen as CU_FILE_DRIVER_ALREADY_OPEN,
    hipFileHandleNotRegistered,
    hipFileHandleNotRegistered as CU_FILE_HANDLE_NOT_REGISTERED,
    hipFileHandleAlreadyRegistered,
    hipFileHandleAlreadyRegistered as CU_FILE_HANDLE_ALREADY_REGISTERED,
    hipFileDeviceNotFound,
    hipFileDeviceNotFound as CU_FILE_DEVICE_NOT_FOUND,
    hipFileInternalError,
    hipFileInternalError as CU_FILE_INTERNAL_ERROR,
    hipFileGetNewFDFailed,
    hipFileGetNewFDFailed as CU_FILE_GETNEWFD_FAILED,
    hipFileDriverSetupError,
    hipFileDriverSetupError as CU_FILE_NVFS_SETUP_ERROR,
    hipFileIODisabled,
    hipFileIODisabled as CU_FILE_IO_DISABLED,
    hipFileBatchSubmitFailed,
    hipFileBatchSubmitFailed as CU_FILE_BATCH_SUBMIT_FAILED,
    hipFileGPUMemoryPinningFailed,
    hipFileGPUMemoryPinningFailed as CU_FILE_GPU_MEMORY_PINNING_FAILED,
    hipFileBatchFull,
    hipFileBatchFull as CU_FILE_BATCH_FULL,
    hipFileAsyncNotSupported,
    hipFileAsyncNotSupported as CU_FILE_ASYNC_NOT_SUPPORTED,
    hipFileIOMaxError,
    hipFileIOMaxError as CU_FILE_IO_MAX_ERROR,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileDriverStatusFlags  <->  CU_FILE_*_SUPPORTED
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileLustreSupported,
    hipFileLustreSupported as CU_FILE_LUSTRE_SUPPORTED,
    hipFileWekaFSSupported,
    hipFileWekaFSSupported as CU_FILE_WEKAFS_SUPPORTED,
    hipFileNFSSupported,
    hipFileNFSSupported as CU_FILE_NFS_SUPPORTED,
    hipFileGPFSSupported,
    hipFileGPFSSupported as CU_FILE_GPFS_SUPPORTED,
    hipFileNVMeSupported,
    hipFileNVMeSupported as CU_FILE_NVME_SUPPORTED,
    hipFileNVMeoFSupported,
    hipFileNVMeoFSupported as CU_FILE_NVMEOF_SUPPORTED,
    hipFileSCSISupported,
    hipFileSCSISupported as CU_FILE_SCSI_SUPPORTED,
    hipFileScaleFluxCSDSupported,
    hipFileScaleFluxCSDSupported as CU_FILE_SCALEFLUX_CSD_SUPPORTED,
    hipFileNVMeshSupported,
    hipFileNVMeshSupported as CU_FILE_NVMESH_SUPPORTED,
    hipFileBeeGFSSupported,
    hipFileBeeGFSSupported as CU_FILE_BEEGFS_SUPPORTED,
    hipFileNVMeP2PSupported,
    hipFileNVMeP2PSupported as CU_FILE_NVME_P2P_SUPPORTED,
    hipFileScatefsSupported,
    hipFileScatefsSupported as CU_FILE_SCATEFS_SUPPORTED,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileDriverControlFlags  <->  CU_FILE_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileUsePollMode,
    hipFileUsePollMode as CU_FILE_USE_POLL_MODE,
    hipFileAllowCompatMode,
    hipFileAllowCompatMode as CU_FILE_ALLOW_COMPAT_MODE,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileFeatureFlags  <->  CU_FILE_*_SUPPORTED
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileDynRoutingSupported,
    hipFileDynRoutingSupported as CU_FILE_DYN_ROUTING_SUPPORTED,
    hipFileBatchIOSupported,
    hipFileBatchIOSupported as CU_FILE_BATCH_IO_SUPPORTED,
    hipFileStreamsSupported,
    hipFileStreamsSupported as CU_FILE_STREAMS_SUPPORTED,
    hipFileParallelIOSupported,
    hipFileParallelIOSupported as CU_FILE_PARALLEL_IO_SUPPORTED,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileFileHandleType  <->  CU_FILE_HANDLE_TYPE_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileHandleTypeOpaqueFD,
    hipFileHandleTypeOpaqueFD as CU_FILE_HANDLE_TYPE_OPAQUE_FD,
    hipFileHandleTypeOpaqueWin32,
    hipFileHandleTypeOpaqueWin32 as CU_FILE_HANDLE_TYPE_OPAQUE_WIN32,
    hipFileHandleTypeUserspaceFS,
    hipFileHandleTypeUserspaceFS as CU_FILE_HANDLE_TYPE_USERSPACE_FS,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileOpcode  <->  CUFILE_READ / CUFILE_WRITE
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileBatchRead,
    hipFileBatchRead as CUFILE_READ,
    hipFileBatchWrite,
    hipFileBatchWrite as CUFILE_WRITE,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileStatus  <->  CUFILE_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileWaiting,
    hipFileWaiting as CUFILE_WAITING,
    hipFilePending,
    hipFilePending as CUFILE_PENDING,
    hipFileInvalid,
    hipFileInvalid as CUFILE_INVALID,
    hipFileCanceled,
    hipFileCanceled as CUFILE_CANCELED,
    hipFileComplete,
    hipFileComplete as CUFILE_COMPLETE,
    hipFileTimeout,
    hipFileTimeout as CUFILE_TIMEOUT,
    hipFileFailed,
    hipFileFailed as CUFILE_FAILED,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileBatchMode  <->  CUFILE_BATCH
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileBatch,
    hipFileBatch as CUFILE_BATCH,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileSizeTConfigParameter_t  <->  CUFILE_PARAM_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileParamProfileStats,
    hipFileParamProfileStats as CUFILE_PARAM_PROFILE_STATS,
    hipFileParamExecutionMaxIOQueueDepth,
    hipFileParamExecutionMaxIOQueueDepth as CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH,
    hipFileParamExecutionMaxIOThreads,
    hipFileParamExecutionMaxIOThreads as CUFILE_PARAM_EXECUTION_MAX_IO_THREADS,
    hipFileParamExecutionMinIOThresholdSizeKB,
    hipFileParamExecutionMinIOThresholdSizeKB as CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB,
    hipFileParamExecutionMaxRequestParallelism,
    hipFileParamExecutionMaxRequestParallelism as CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM,
    hipFileParamPropertiesMaxDirectIOSizeKB,
    hipFileParamPropertiesMaxDirectIOSizeKB as CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB,
    hipFileParamPropertiesMaxDeviceCacheSizeKB,
    hipFileParamPropertiesMaxDeviceCacheSizeKB as CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB,
    hipFileParamPropertiesPerBufferCacheSizeKB,
    hipFileParamPropertiesPerBufferCacheSizeKB as CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB,
    hipFileParamPropertiesMaxDevicePinnedMemSizeKB,
    hipFileParamPropertiesMaxDevicePinnedMemSizeKB as CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB,
    hipFileParamPropertiesIOBatchsize,
    hipFileParamPropertiesIOBatchsize as CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE,
    hipFileParamPollthresholdSizeKB,
    hipFileParamPollthresholdSizeKB as CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB,
    hipFileParamPropertiesBatchIOTimeoutMs,
    hipFileParamPropertiesBatchIOTimeoutMs as CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileBoolConfigParameter_t  <->  CUFILE_PARAM_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileParamPropertiesUsePollMode,
    hipFileParamPropertiesUsePollMode as CUFILE_PARAM_PROPERTIES_USE_POLL_MODE,
    hipFileParamPropertiesAllowCompatMode,
    hipFileParamPropertiesAllowCompatMode as CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE,
    hipFileParamForceCompatMode,
    hipFileParamForceCompatMode as CUFILE_PARAM_FORCE_COMPAT_MODE,
    hipFileParamFsMiscApiCheckAggressive,
    hipFileParamFsMiscApiCheckAggressive as CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE,
    hipFileParamExecutionParallelIO,
    hipFileParamExecutionParallelIO as CUFILE_PARAM_EXECUTION_PARALLEL_IO,
    hipFileParamProfileNvtx,
    hipFileParamProfileNvtx as CUFILE_PARAM_PROFILE_NVTX,
    hipFileParamPropertiesAllowSystemMemory,
    hipFileParamPropertiesAllowSystemMemory as CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY,
    hipFileParamUsePcip2pdma,
    hipFileParamUsePcip2pdma as CUFILE_PARAM_USE_PCIP2PDMA,
    hipFileParamPreferIOUring,
    hipFileParamPreferIOUring as CUFILE_PARAM_PREFER_IO_URING,
    hipFileParamForceOdirectMode,
    hipFileParamForceOdirectMode as CUFILE_PARAM_FORCE_ODIRECT_MODE,
    hipFileParamSkipTopologyDetection,
    hipFileParamSkipTopologyDetection as CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION,
    hipFileParamStreamMemopsBypass,
    hipFileParamStreamMemopsBypass as CUFILE_PARAM_STREAM_MEMOPS_BYPASS,
)

# ---------------------------------------------------------------------------
# Enumerators: hipFileStringConfigParameter_t  <->  CUFILE_PARAM_*
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileParamLoggingLevel,
    hipFileParamLoggingLevel as CUFILE_PARAM_LOGGING_LEVEL,
    hipFileParamEnvLogfilePath,
    hipFileParamEnvLogfilePath as CUFILE_PARAM_ENV_LOGFILE_PATH,
    hipFileParamLogDir,
    hipFileParamLogDir as CUFILE_PARAM_LOG_DIR,
)

# ---------------------------------------------------------------------------
# Structs / opaque handle typedefs  <->  CUfile*_t
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileError,
    hipFileError_t,
    hipFileError as CUfileError,
    hipFileError_t as CUfileError_t,
    hipFileDriverProps,
    hipFileDriverProps_t,
    hipFileDriverProps as CUfileDrvProps,
    hipFileDriverProps_t as CUfileDrvProps_t,
    hipFileRDMAInfo,
    hipFileRDMAInfo_t,
    hipFileRDMAInfo as CUfileRDMAInfo,
    hipFileRDMAInfo_t as CUfileRDMAInfo_t,
    hipFileFSOps,
    hipFileFSOps_t,
    hipFileFSOps as CUfileFSOps,
    hipFileFSOps_t as CUfileFSOps_t,
    hipFileDescr,
    hipFileDescr_t,
    hipFileDescr as CUfileDescr,
    hipFileDescr_t as CUfileDescr_t,
    hipFileHandle_t,
    hipFileHandle_t as CUfileHandle_t,
    hipFileIOParams,
    hipFileIOParams_t,
    hipFileIOParams as CUfileIOParams,
    hipFileIOParams_t as CUfileIOParams_t,
    hipFileIOEvents,
    hipFileIOEvents_t,
    hipFileIOEvents as CUfileIOEvents,
    hipFileIOEvents_t as CUfileIOEvents_t,
    hipFileBatchHandle_t,
    hipFileBatchHandle_t as CUfileBatchHandle_t,
)

# ---------------------------------------------------------------------------
# Functions  <->  cuFile* / cufileop_status_error
# ---------------------------------------------------------------------------
from rocm.bindings.cyhipfile cimport (
    hipFileGetOpErrorString,
    hipFileGetOpErrorString as cufileop_status_error,
    hipFileHandleRegister,
    hipFileHandleRegister as cuFileHandleRegister,
    hipFileHandleDeregister,
    hipFileHandleDeregister as cuFileHandleDeregister,
    hipFileBufRegister,
    hipFileBufRegister as cuFileBufRegister,
    hipFileBufDeregister,
    hipFileBufDeregister as cuFileBufDeregister,
    hipFileRead,
    hipFileRead as cuFileRead,
    hipFileWrite,
    hipFileWrite as cuFileWrite,
    hipFileDriverOpen,
    hipFileDriverOpen as cuFileDriverOpen,
    hipFileDriverClose,
    hipFileDriverClose as cuFileDriverClose,
    hipFileUseCount,
    hipFileUseCount as cuFileUseCount,
    hipFileDriverGetProperties,
    hipFileDriverGetProperties as cuFileDriverGetProperties,
    hipFileDriverSetPollMode,
    hipFileDriverSetPollMode as cuFileDriverSetPollMode,
    hipFileDriverSetMaxDirectIOSize,
    hipFileDriverSetMaxDirectIOSize as cuFileDriverSetMaxDirectIOSize,
    hipFileDriverSetMaxCacheSize,
    hipFileDriverSetMaxCacheSize as cuFileDriverSetMaxCacheSize,
    hipFileDriverSetMaxPinnedMemSize,
    hipFileDriverSetMaxPinnedMemSize as cuFileDriverSetMaxPinnedMemSize,
    hipFileBatchIOSetUp,
    hipFileBatchIOSetUp as cuFileBatchIOSetUp,
    hipFileBatchIOSubmit,
    hipFileBatchIOSubmit as cuFileBatchIOSubmit,
    hipFileBatchIOGetStatus,
    hipFileBatchIOGetStatus as cuFileBatchIOGetStatus,
    hipFileBatchIOCancel,
    hipFileBatchIOCancel as cuFileBatchIOCancel,
    hipFileBatchIODestroy,
    hipFileBatchIODestroy as cuFileBatchIODestroy,
    hipFileReadAsync,
    hipFileReadAsync as cuFileReadAsync,
    hipFileWriteAsync,
    hipFileWriteAsync as cuFileWriteAsync,
    hipFileStreamRegister,
    hipFileStreamRegister as cuFileStreamRegister,
    hipFileStreamDeregister,
    hipFileStreamDeregister as cuFileStreamDeregister,
    hipFileGetVersion,
    hipFileGetVersion as cuFileGetVersion,
    hipFileGetParameterSizeT,
    hipFileGetParameterSizeT as cuFileGetParameterSizeT,
    hipFileGetParameterBool,
    hipFileGetParameterBool as cuFileGetParameterBool,
    hipFileGetParameterString,
    hipFileGetParameterString as cuFileGetParameterString,
    hipFileSetParameterSizeT,
    hipFileSetParameterSizeT as cuFileSetParameterSizeT,
    hipFileSetParameterBool,
    hipFileSetParameterBool as cuFileSetParameterBool,
    hipFileSetParameterString,
    hipFileSetParameterString as cuFileSetParameterString,
)
