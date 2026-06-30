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

"""A ``pynvml`` (NVML) compatibility shim for ROCm.

This is NOT a port of the upstream ``pynvml`` / ``nvidia-ml-py`` source. It is a
fresh, MIT-licensed re-implementation of the subset of the NVML Python surface
used by HIP ports of RAPIDS projects (e.g. cuDF), backed entirely by AMD SMI via
the high-level ``rocm.bindings.amdsmi`` bindings.

Goal: code that does ``import pynvml`` and calls a handful of ``nvml*`` functions
keeps working on AMD GPUs without modification. Where concepts do not exist on
ROCm (e.g. MIG), the corresponding calls degrade gracefully.

Implemented surface:

* ``nvmlInit`` / ``nvmlInitWithFlags`` / ``nvmlShutdown``
* ``nvmlDeviceGetCount``
* ``nvmlDeviceGetHandleByIndex`` / ``nvmlDeviceGetHandleByUUID``
* ``nvmlDeviceIsMigDeviceHandle`` / ``nvmlDeviceGetDeviceHandleFromMigDeviceHandle``
* ``nvmlDeviceGetMemoryInfo`` / ``nvmlDeviceGetName`` / ``nvmlDeviceGetUUID``
* ``nvmlDeviceGetTemperature`` / ``nvmlDeviceGetPowerUsage``
* ``nvmlDeviceGetUtilizationRates``
* ``nvmlDeviceGetComputeRunningProcesses`` (best-effort, see note)
* ``NVMLError`` and the per-code subclasses consumers commonly catch.

NVML eventually migrates to ``cuda.core.system``; this shim covers the legacy
``pynvml`` entry points still in use until that migration completes.
"""

import ctypes

from rocm.bindings import amdsmi

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

_S = amdsmi.amdsmi_status_t
_OK = int(_S.AMDSMI_STATUS_SUCCESS)

# NVML status codes (values match the upstream nvml.h so consumers comparing the
# integer ``.value`` keep working).
NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_UNKNOWN = 999

# Temperature sensor selectors (NVML nvmlTemperatureSensors_t).
NVML_TEMPERATURE_GPU = 0

# Sentinel used by AMD SMI for unsupported scalar telemetry fields.
_UINT32_MAX = 0xFFFFFFFF
_UINT16_MAX = 0xFFFF


class NVMLError(Exception):
    """Base class mirroring ``pynvml.NVMLError``.

    Carries an integer ``value`` (an ``NVML_ERROR_*`` code) so existing code that
    inspects ``err.value`` continues to work. Concrete per-code subclasses are
    registered in ``_NVML_ERROR_SUBCLASSES`` and surfaced as module attributes
    (e.g. ``NVMLError_NotSupported``), matching upstream pynvml.
    """

    value = NVML_ERROR_UNKNOWN

    def __new__(cls, value=None, msg=None):
        # When raised as ``NVMLError(code)`` dispatch to the matching subclass,
        # exactly like upstream pynvml's ``NVMLError.__new__``.
        if cls is NVMLError and value is not None:
            subcls = _NVML_ERROR_SUBCLASSES.get(int(value))
            if subcls is not None:
                return super().__new__(subcls)
        return super().__new__(cls)

    def __init__(self, value=None, msg=None):
        if value is not None:
            self.value = int(value)
        self._msg = msg
        super().__init__(msg or f"NVML error {self.value}")

    def __str__(self):
        return self._msg or f"NVML error {self.value}"

    def __eq__(self, other):
        if isinstance(other, NVMLError):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


def _make_error_subclass(name, code):
    cls = type(name, (NVMLError,), {"value": code})
    globals()[name] = cls
    return cls


# Per-code subclasses consumers may catch by name.
NVMLError_Uninitialized = _make_error_subclass(
    "NVMLError_Uninitialized", NVML_ERROR_UNINITIALIZED)
NVMLError_InvalidArgument = _make_error_subclass(
    "NVMLError_InvalidArgument", NVML_ERROR_INVALID_ARGUMENT)
NVMLError_NotSupported = _make_error_subclass(
    "NVMLError_NotSupported", NVML_ERROR_NOT_SUPPORTED)
NVMLError_NoPermission = _make_error_subclass(
    "NVMLError_NoPermission", NVML_ERROR_NO_PERMISSION)
NVMLError_NotFound = _make_error_subclass(
    "NVMLError_NotFound", NVML_ERROR_NOT_FOUND)
NVMLError_InsufficientSize = _make_error_subclass(
    "NVMLError_InsufficientSize", NVML_ERROR_INSUFFICIENT_SIZE)
NVMLError_DriverNotLoaded = _make_error_subclass(
    "NVMLError_DriverNotLoaded", NVML_ERROR_DRIVER_NOT_LOADED)
NVMLError_Timeout = _make_error_subclass(
    "NVMLError_Timeout", NVML_ERROR_TIMEOUT)
NVMLError_GpuIsLost = _make_error_subclass(
    "NVMLError_GpuIsLost", NVML_ERROR_GPU_IS_LOST)
NVMLError_FunctionNotFound = _make_error_subclass(
    "NVMLError_FunctionNotFound", NVML_ERROR_FUNCTION_NOT_FOUND)
NVMLError_Unknown = _make_error_subclass(
    "NVMLError_Unknown", NVML_ERROR_UNKNOWN)

_NVML_ERROR_SUBCLASSES = {
    cls.value: cls
    for cls in (
        NVMLError_Uninitialized,
        NVMLError_InvalidArgument,
        NVMLError_NotSupported,
        NVMLError_NoPermission,
        NVMLError_NotFound,
        NVMLError_InsufficientSize,
        NVMLError_DriverNotLoaded,
        NVMLError_Timeout,
        NVMLError_GpuIsLost,
        NVMLError_FunctionNotFound,
        NVMLError_Unknown,
    )
}

# Map AMD SMI status codes onto the closest NVML error code.
_AMDSMI_TO_NVML = {
    int(_S.AMDSMI_STATUS_NOT_SUPPORTED): NVML_ERROR_NOT_SUPPORTED,
    int(_S.AMDSMI_STATUS_NOT_INIT): NVML_ERROR_UNINITIALIZED,
    int(_S.AMDSMI_STATUS_INVAL): NVML_ERROR_INVALID_ARGUMENT,
    int(_S.AMDSMI_STATUS_NO_PERM): NVML_ERROR_NO_PERMISSION,
    int(_S.AMDSMI_STATUS_NOT_FOUND): NVML_ERROR_NOT_FOUND,
    int(_S.AMDSMI_STATUS_OUT_OF_RESOURCES): NVML_ERROR_INSUFFICIENT_SIZE,
    int(_S.AMDSMI_STATUS_TIMEOUT): NVML_ERROR_TIMEOUT,
}


def _status_int(call_result):
    """Extract the integer AMD SMI status from a binding return value.

    ``rocm.bindings.amdsmi`` functions return either a bare ``amdsmi_status_t``
    or a tuple whose first element is the status.
    """
    status = call_result[0] if isinstance(call_result, (tuple, list)) else call_result
    return int(status)


def _check(call_result, what=""):
    """Raise the matching ``NVMLError`` on a non-success AMD SMI status."""
    status = _status_int(call_result)
    if status != _OK:
        code = _AMDSMI_TO_NVML.get(status, NVML_ERROR_UNKNOWN)
        raise NVMLError(code, msg=f"{what or 'amdsmi call'} failed (amdsmi status {status})")
    return call_result


# ---------------------------------------------------------------------------
# Return objects (mirror the ctypes structs pynvml hands back).
# ---------------------------------------------------------------------------


class c_nvmlMemory_t:
    """Mirror of ``nvmlMemory_t`` (bytes)."""

    __slots__ = ("total", "free", "used")

    def __init__(self, total, free, used):
        self.total = total
        self.free = free
        self.used = used

    def __repr__(self):
        return f"c_nvmlMemory_t(total={self.total}, free={self.free}, used={self.used})"


class c_nvmlUtilization_t:
    """Mirror of ``nvmlUtilization_t`` (percentages)."""

    __slots__ = ("gpu", "memory")

    def __init__(self, gpu, memory):
        self.gpu = gpu
        self.memory = memory

    def __repr__(self):
        return f"c_nvmlUtilization_t(gpu={self.gpu}, memory={self.memory})"


class c_nvmlProcessInfo_t:
    """Mirror of ``nvmlProcessInfo_t`` (subset used by consumers)."""

    __slots__ = ("pid", "usedGpuMemory")

    def __init__(self, pid, usedGpuMemory):
        self.pid = pid
        self.usedGpuMemory = usedGpuMemory

    def __repr__(self):
        return f"c_nvmlProcessInfo_t(pid={self.pid}, usedGpuMemory={self.usedGpuMemory})"


class _NvmlDevice:
    """Opaque NVML device handle backed by an AMD SMI processor handle."""

    __slots__ = ("index", "_amdsmi_handle")

    def __init__(self, index, amdsmi_handle):
        self.index = index
        # Stored as a raw integer; wrapped in ctypes.c_void_p when passed back
        # to amdsmi so it survives the Pointer.fromPyobj round-trip.
        self._amdsmi_handle = amdsmi_handle

    @property
    def _handle(self):
        return ctypes.c_void_p(self._amdsmi_handle)

    def __repr__(self):
        return f"<pynvml device index={self.index} (AMD SMI)>"


# ---------------------------------------------------------------------------
# Library state.
# ---------------------------------------------------------------------------

_init_count = 0
_devices = []  # ordered list of _NvmlDevice, index == NVML device ordinal


def _ensure_initialized():
    if _init_count <= 0:
        raise NVMLError(NVML_ERROR_UNINITIALIZED, msg="NVML was not successfully initialized")


def _enumerate_gpus():
    """Return the ordered list of AMD GPU processor handles (as raw ints).

    Drives AMD SMI's two-call (count -> allocate -> fill) enumeration:
    sockets -> processors -> filter ``AMDSMI_PROCESSOR_TYPE_AMD_GPU``.
    """
    gpu_type = int(amdsmi.processor_type_t.AMDSMI_PROCESSOR_TYPE_AMD_GPU)

    socket_count = (ctypes.c_uint * 1)()
    _check(amdsmi.amdsmi_get_socket_handles(socket_count, None), "amdsmi_get_socket_handles")
    n_sockets = socket_count[0]
    if n_sockets == 0:
        return []
    socket_arr = (ctypes.c_void_p * n_sockets)()
    socket_count[0] = n_sockets
    _check(amdsmi.amdsmi_get_socket_handles(socket_count, socket_arr), "amdsmi_get_socket_handles")

    handles = []
    for s in range(n_sockets):
        socket_handle = ctypes.c_void_p(socket_arr[s])
        proc_count = (ctypes.c_uint * 1)()
        _check(
            amdsmi.amdsmi_get_processor_handles(socket_handle, proc_count, None),
            "amdsmi_get_processor_handles",
        )
        n_procs = proc_count[0]
        if n_procs == 0:
            continue
        proc_arr = (ctypes.c_void_p * n_procs)()
        proc_count[0] = n_procs
        _check(
            amdsmi.amdsmi_get_processor_handles(socket_handle, proc_count, proc_arr),
            "amdsmi_get_processor_handles",
        )
        for p in range(n_procs):
            proc_handle = ctypes.c_void_p(proc_arr[p])
            _, ptype = _check(
                amdsmi.amdsmi_get_processor_type(proc_handle),
                "amdsmi_get_processor_type",
            )
            if int(ptype) == gpu_type:
                handles.append(proc_arr[p])
    return handles


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------


def nvmlInit():
    """Initialize NVML (maps to ``amdsmi_init`` for AMD GPUs)."""
    return nvmlInitWithFlags(0)


def nvmlInitWithFlags(flags):
    """Initialize NVML; ``flags`` is accepted for compatibility and ignored."""
    global _init_count
    if _init_count > 0:
        _init_count += 1
        return
    _check(
        amdsmi.amdsmi_init(amdsmi.amdsmi_init_flags_t.AMDSMI_INIT_AMD_GPUS),
        "amdsmi_init",
    )
    try:
        raw_handles = _enumerate_gpus()
    except Exception:
        amdsmi.amdsmi_shut_down()
        raise
    _devices[:] = [_NvmlDevice(i, h) for i, h in enumerate(raw_handles)]
    _init_count = 1


def nvmlShutdown():
    """Shut down NVML (refcounted, maps to ``amdsmi_shut_down``)."""
    global _init_count
    _ensure_initialized()
    _init_count -= 1
    if _init_count == 0:
        _devices[:] = []
        _check(amdsmi.amdsmi_shut_down(), "amdsmi_shut_down")


# ---------------------------------------------------------------------------
# Device enumeration / handles.
# ---------------------------------------------------------------------------


def nvmlDeviceGetCount():
    """Number of AMD GPUs visible to AMD SMI."""
    _ensure_initialized()
    return len(_devices)


def nvmlDeviceGetHandleByIndex(index):
    """Return the device handle for the given ordinal."""
    _ensure_initialized()
    index = int(index)
    if index < 0 or index >= len(_devices):
        raise NVMLError(NVML_ERROR_INVALID_ARGUMENT, msg=f"invalid device index {index}")
    return _devices[index]


def nvmlDeviceGetHandleByUUID(uuid):
    """Return the device whose UUID matches ``uuid`` (``str`` or ``bytes``)."""
    _ensure_initialized()
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii", "replace")
    target = uuid.strip()
    # Accept both bare and ``GPU-`` prefixed UUIDs.
    candidates = {target}
    if target.startswith("GPU-"):
        candidates.add(target[len("GPU-"):])
    else:
        candidates.add("GPU-" + target)
    for dev in _devices:
        dev_uuid = nvmlDeviceGetUUID(dev)
        if dev_uuid in candidates or dev_uuid[len("GPU-"):] in candidates:
            return dev
    raise NVMLError(NVML_ERROR_NOT_FOUND, msg=f"no device with UUID {uuid!r}")


def nvmlDeviceIsMigDeviceHandle(handle):
    """ROCm has no MIG; always reports ``False``."""
    _ensure_initialized()
    return False


def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle):
    """No MIG on ROCm; the handle already refers to a full device."""
    _ensure_initialized()
    return handle


# ---------------------------------------------------------------------------
# Device queries.
# ---------------------------------------------------------------------------


def _decode_cstr(value):
    """Decode a NUL-terminated byte field returned by AMD SMI structs."""
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return str(value)


def nvmlDeviceGetMemoryInfo(handle):
    """VRAM totals as an ``nvmlMemory_t``-like object (bytes)."""
    _ensure_initialized()
    info = _check(
        amdsmi.amdsmi_get_gpu_vram_usage(handle._handle),
        "amdsmi_get_gpu_vram_usage",
    )[1]
    # AMD SMI reports VRAM in MiB; NVML reports bytes.
    mib = 1024 * 1024
    total = int(info.vram_total) * mib
    used = int(info.vram_used) * mib
    free = max(total - used, 0)
    return c_nvmlMemory_t(total=total, free=free, used=used)


def nvmlDeviceGetName(handle):
    """Marketing/product name as a ``str`` (NVML returns ``str`` on py3)."""
    _ensure_initialized()
    info = _check(
        amdsmi.amdsmi_get_gpu_asic_info(handle._handle),
        "amdsmi_get_gpu_asic_info",
    )[1]
    name = _decode_cstr(info.market_name)
    if name:
        return name
    board = _check(
        amdsmi.amdsmi_get_gpu_board_info(handle._handle),
        "amdsmi_get_gpu_board_info",
    )[1]
    return _decode_cstr(board.product_name)


def nvmlDeviceGetUUID(handle):
    """Device UUID rendered as the conventional ``GPU-<uuid>`` string."""
    _ensure_initialized()
    size = int(amdsmi.AMDSMI_GPU_UUID_SIZE)
    length = (ctypes.c_uint * 1)()
    length[0] = size
    buf = bytearray(size + 1)
    _check(
        amdsmi.amdsmi_get_gpu_device_uuid(handle._handle, length, buf),
        "amdsmi_get_gpu_device_uuid",
    )
    uuid = bytes(buf).split(b"\x00", 1)[0].decode("ascii", "replace").strip()
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


def nvmlDeviceGetTemperature(handle, sensorType):
    """Current temperature in Celsius for the requested sensor."""
    _ensure_initialized()
    # NVML only defines NVML_TEMPERATURE_GPU; map it to the edge sensor and fall
    # back to the hotspot/junction sensor where edge is unavailable.
    sensors = (
        amdsmi.amdsmi_temperature_type_t.AMDSMI_TEMPERATURE_TYPE_EDGE,
        amdsmi.amdsmi_temperature_type_t.AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
    )
    last_status = None
    for sensor in sensors:
        temperature = (ctypes.c_int64 * 1)()
        result = amdsmi.amdsmi_get_temp_metric(
            handle._handle,
            sensor,
            amdsmi.amdsmi_temperature_metric_t.AMDSMI_TEMP_CURRENT,
            temperature,
        )
        if _status_int(result) == _OK:
            return int(temperature[0])
        last_status = result
    _check(last_status, "amdsmi_get_temp_metric")


def nvmlDeviceGetPowerUsage(handle):
    """Current board power draw in milliwatts."""
    _ensure_initialized()
    info = _check(
        amdsmi.amdsmi_get_power_info(handle._handle),
        "amdsmi_get_power_info",
    )[1]
    watts = int(info.current_socket_power)
    if watts in (_UINT32_MAX, _UINT16_MAX):
        watts = int(getattr(info, "average_socket_power", _UINT32_MAX))
    if watts in (_UINT32_MAX, _UINT16_MAX):
        raise NVMLError(NVML_ERROR_NOT_SUPPORTED, msg="power telemetry not supported")
    return watts * 1000


def nvmlDeviceGetUtilizationRates(handle):
    """GPU/memory engine utilization percentages."""
    _ensure_initialized()
    info = _check(
        amdsmi.amdsmi_get_gpu_activity(handle._handle),
        "amdsmi_get_gpu_activity",
    )[1]

    def _norm(value):
        value = int(value)
        return 0 if value in (_UINT32_MAX, _UINT16_MAX) else value

    return c_nvmlUtilization_t(
        gpu=_norm(info.gfx_activity),
        memory=_norm(info.umc_activity),
    )


def nvmlDeviceGetComputeRunningProcesses(handle):
    """Compute processes running on the device.

    Drives AMD SMI's two-call (count -> allocate -> fill) process query:
    ``amdsmi_get_gpu_process_list`` reports the running-process count when
    ``max_processes`` is 0 and ``list`` is ``None``; a caller-sized
    ``amdsmi_proc_info_t`` record array (used as a sequence adapter via its
    indexed ``get_*(i)`` accessors) is then filled on the second call.
    Returns an empty list when nothing is running or the platform does not
    support the query.
    """
    _ensure_initialized()
    count = (ctypes.c_uint * 1)()
    # First call: NULL list with a zero size asks AMD SMI for the count.
    count[0] = 0
    status = _status_int(
        amdsmi.amdsmi_get_gpu_process_list(handle._handle, count, None)
    )
    if status == int(_S.AMDSMI_STATUS_NOT_SUPPORTED):
        return []
    _check(status, "amdsmi_get_gpu_process_list")
    n = int(count[0])
    if n == 0:
        return []
    # Second call: caller-allocated array of n amdsmi_proc_info_t records.
    proc_list = amdsmi.amdsmi_proc_info_t.allocate(n)
    count[0] = n
    _check(
        amdsmi.amdsmi_get_gpu_process_list(handle._handle, count, proc_list),
        "amdsmi_get_gpu_process_list",
    )
    processes = []
    for i in range(int(count[0])):
        pid = int(proc_list.get_pid(i))
        if pid == 0:
            continue
        used = int(proc_list.get_mem(i))
        if used == 0:
            # Fall back to the per-engine VRAM accounting when ``mem`` is unset.
            mem_usage = proc_list.get_memory_usage(i)
            if mem_usage is not None:
                used = int(getattr(mem_usage, "vram_mem", 0))
        processes.append(c_nvmlProcessInfo_t(pid=pid, usedGpuMemory=used))
    return processes


__all__ = [
    "NVMLError",
    "NVMLError_Uninitialized",
    "NVMLError_InvalidArgument",
    "NVMLError_NotSupported",
    "NVMLError_NoPermission",
    "NVMLError_NotFound",
    "NVMLError_InsufficientSize",
    "NVMLError_DriverNotLoaded",
    "NVMLError_Timeout",
    "NVMLError_GpuIsLost",
    "NVMLError_FunctionNotFound",
    "NVMLError_Unknown",
    "NVML_SUCCESS",
    "NVML_ERROR_UNINITIALIZED",
    "NVML_ERROR_INVALID_ARGUMENT",
    "NVML_ERROR_NOT_SUPPORTED",
    "NVML_ERROR_NOT_FOUND",
    "NVML_ERROR_UNKNOWN",
    "NVML_TEMPERATURE_GPU",
    "c_nvmlMemory_t",
    "c_nvmlUtilization_t",
    "c_nvmlProcessInfo_t",
    "nvmlInit",
    "nvmlInitWithFlags",
    "nvmlShutdown",
    "nvmlDeviceGetCount",
    "nvmlDeviceGetHandleByIndex",
    "nvmlDeviceGetHandleByUUID",
    "nvmlDeviceIsMigDeviceHandle",
    "nvmlDeviceGetDeviceHandleFromMigDeviceHandle",
    "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetName",
    "nvmlDeviceGetUUID",
    "nvmlDeviceGetTemperature",
    "nvmlDeviceGetPowerUsage",
    "nvmlDeviceGetUtilizationRates",
    "nvmlDeviceGetComputeRunningProcesses",
]
