# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
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

# Ported from ROCm/rocm-systems projects/hipfile/python/hipfile/enums.py
# (commit cbbf349092, "[hipFile] Polish the python project prior to early
# release on PyPI", #5089) — original author Riley Dixon
# <riley.dixon@amd.com>. The port re-routes the enum-value source from the
# upstream's _hipfile.pyx Cython extension to hip-python's auto-generated
# rocm.bindings.hipfile IntEnum classes.

__author__ = (
    "Riley Dixon <riley.dixon@amd.com> (original); "
    "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com> (port)"
)

# Re-expose the friendlier enum names the upstream package uses. The values
# are sourced from hip-python's auto-generated `rocm.bindings.hipfile` IntEnum
# classes (whose members carry the raw C names, e.g. `hipFileHandleTypeOpaqueFD`)
# so a rebuild against a newer hipfile.h picks up any value changes
# automatically; only the friendly aliases live here. Users of
# `from rocm.hipfile import OpError, FileHandleType` get the same shape they'd
# expect from upstream (`OpError.SUCCESS`, `FileHandleType.OPAQUE_FD`, ...).
from enum import IntEnum

from rocm.bindings.hipfile import (
    hipFileOpError as _hipFileOpError,
    hipFileFileHandleType as _hipFileFileHandleType,
)


class OpError(IntEnum):
    """Python enum mirroring ``hipFileOpError_t`` with upstream-friendly names."""

    SUCCESS = _hipFileOpError.hipFileSuccess
    DRIVER_NOT_INITIALIZED = _hipFileOpError.hipFileDriverNotInitialized
    DRIVER_INVALID_PROPS = _hipFileOpError.hipFileDriverInvalidProps
    DRIVER_UNSUPPORTED_LIMIT = _hipFileOpError.hipFileDriverUnsupportedLimit
    DRIVER_VERSION_MISMATCH = _hipFileOpError.hipFileDriverVersionMismatch
    DRIVER_VERSION_READ_ERROR = _hipFileOpError.hipFileDriverVersionReadError
    DRIVER_CLOSING = _hipFileOpError.hipFileDriverClosing
    PLATFORM_NOT_SUPPORTED = _hipFileOpError.hipFilePlatformNotSupported
    IO_NOT_SUPPORTED = _hipFileOpError.hipFileIONotSupported
    DEVICE_NOT_SUPPORTED = _hipFileOpError.hipFileDeviceNotSupported
    DRIVER_ERROR = _hipFileOpError.hipFileDriverError
    HIP_DRIVER_ERROR = _hipFileOpError.hipFileHipDriverError
    HIP_POINTER_INVALID = _hipFileOpError.hipFileHipPointerInvalid
    HIP_MEMORY_TYPE_INVALID = _hipFileOpError.hipFileHipMemoryTypeInvalid
    HIP_POINTER_RANGE_ERROR = _hipFileOpError.hipFileHipPointerRangeError
    HIP_CONTEXT_MISMATCH = _hipFileOpError.hipFileHipContextMismatch
    INVALID_MAPPING_SIZE = _hipFileOpError.hipFileInvalidMappingSize
    INVALID_MAPPING_RANGE = _hipFileOpError.hipFileInvalidMappingRange
    INVALID_FILE_TYPE = _hipFileOpError.hipFileInvalidFileType
    INVALID_FILE_OPEN_FLAG = _hipFileOpError.hipFileInvalidFileOpenFlag
    DIO_NOT_SET = _hipFileOpError.hipFileDIONotSet
    INVALID_VALUE = _hipFileOpError.hipFileInvalidValue
    MEMORY_ALREADY_REGISTERED = _hipFileOpError.hipFileMemoryAlreadyRegistered
    MEMORY_NOT_REGISTERED = _hipFileOpError.hipFileMemoryNotRegistered
    PERMISSION_DENIED = _hipFileOpError.hipFilePermissionDenied
    DRIVER_ALREADY_OPEN = _hipFileOpError.hipFileDriverAlreadyOpen
    HANDLE_NOT_REGISTERED = _hipFileOpError.hipFileHandleNotRegistered
    HANDLE_ALREADY_REGISTERED = _hipFileOpError.hipFileHandleAlreadyRegistered
    DEVICE_NOT_FOUND = _hipFileOpError.hipFileDeviceNotFound
    INTERNAL_ERROR = _hipFileOpError.hipFileInternalError
    GET_NEW_FD_FAILED = _hipFileOpError.hipFileGetNewFDFailed
    DRIVER_SETUP_ERROR = _hipFileOpError.hipFileDriverSetupError
    IO_DISABLED = _hipFileOpError.hipFileIODisabled
    BATCH_SUBMIT_FAILED = _hipFileOpError.hipFileBatchSubmitFailed
    GPU_MEMORY_PINNING_FAILED = _hipFileOpError.hipFileGPUMemoryPinningFailed
    BATCH_FULL = _hipFileOpError.hipFileBatchFull
    ASYNC_NOT_SUPPORTED = _hipFileOpError.hipFileAsyncNotSupported
    IO_MAX_ERROR = _hipFileOpError.hipFileIOMaxError


class FileHandleType(IntEnum):
    """Python enum mirroring ``hipFileFileHandleType_t`` with friendly names."""

    OPAQUE_FD = _hipFileFileHandleType.hipFileHandleTypeOpaqueFD
    OPAQUE_WIN32 = _hipFileFileHandleType.hipFileHandleTypeOpaqueWin32
    USERSPACE_FS = _hipFileFileHandleType.hipFileHandleTypeUserspaceFS


__all__ = ["OpError", "FileHandleType"]
