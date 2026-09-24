pynvml
======

.. py:module:: pynvml

.. autoapi-nested-parse::

   A ``pynvml`` (NVML) compatibility shim for ROCm.

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
   * ``nvmlDeviceGetMigMode`` / ``nvmlDeviceGetMaxMigDeviceCount`` / ``nvmlDeviceGetMigDeviceHandleByIndex`` (MIG unsupported on ROCm)
   * ``nvmlDeviceGetMemoryInfo`` / ``nvmlDeviceGetName`` / ``nvmlDeviceGetUUID``
   * ``nvmlDeviceGetTemperature`` / ``nvmlDeviceGetPowerUsage``
   * ``nvmlDeviceGetUtilizationRates``
   * ``nvmlDeviceGetCpuAffinity`` (Linux-only, like NVML)
   * ``nvmlDeviceGetComputeRunningProcesses`` (best-effort, see note)
   * ``NVMLError`` and the per-code subclasses consumers commonly catch.

   NVML eventually migrates to ``cuda.core.system``; this shim covers the legacy
   ``pynvml`` entry points still in use until that migration completes.



Attributes
----------

.. autoapisummary::

   pynvml.NVML_SUCCESS
   pynvml.NVML_ERROR_UNINITIALIZED
   pynvml.NVML_ERROR_INVALID_ARGUMENT
   pynvml.NVML_ERROR_NOT_SUPPORTED
   pynvml.NVML_ERROR_NOT_FOUND
   pynvml.NVML_ERROR_LIBRARY_NOT_FOUND
   pynvml.NVML_ERROR_FUNCTION_NOT_FOUND
   pynvml.NVML_ERROR_UNKNOWN
   pynvml.NVML_TEMPERATURE_GPU
   pynvml.NVML_DEVICE_MIG_DISABLE
   pynvml.NVML_DEVICE_MIG_ENABLE
   pynvml.NVMLError_Uninitialized
   pynvml.NVMLError_InvalidArgument
   pynvml.NVMLError_NotSupported
   pynvml.NVMLError_NoPermission
   pynvml.NVMLError_NotFound
   pynvml.NVMLError_InsufficientSize
   pynvml.NVMLError_DriverNotLoaded
   pynvml.NVMLError_Timeout
   pynvml.NVMLError_GpuIsLost
   pynvml.NVMLError_LibraryNotFound
   pynvml.NVMLError_FunctionNotFound
   pynvml.NVMLError_Unknown


Exceptions
----------

.. autoapisummary::

   pynvml.NVMLError


Classes
-------

.. autoapisummary::

   pynvml.c_nvmlMemory_t
   pynvml.c_nvmlUtilization_t
   pynvml.c_nvmlProcessInfo_t


Functions
---------

.. autoapisummary::

   pynvml.nvmlInit
   pynvml.nvmlInitWithFlags
   pynvml.nvmlShutdown
   pynvml.nvmlDeviceGetCount
   pynvml.nvmlDeviceGetHandleByIndex
   pynvml.nvmlDeviceGetHandleByUUID
   pynvml.nvmlDeviceGetIndex
   pynvml.nvmlDeviceIsMigDeviceHandle
   pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle
   pynvml.nvmlDeviceGetMigMode
   pynvml.nvmlDeviceGetMaxMigDeviceCount
   pynvml.nvmlDeviceGetMigDeviceHandleByIndex
   pynvml.nvmlDeviceGetMemoryInfo
   pynvml.nvmlDeviceGetName
   pynvml.nvmlDeviceGetUUID
   pynvml.nvmlDeviceGetTemperature
   pynvml.nvmlDeviceGetPowerUsage
   pynvml.nvmlDeviceGetUtilizationRates
   pynvml.nvmlDeviceGetCpuAffinity
   pynvml.nvmlDeviceGetComputeRunningProcesses


Package Contents
----------------

.. py:data:: NVML_SUCCESS
   :value: 0


.. py:data:: NVML_ERROR_UNINITIALIZED
   :value: 1


.. py:data:: NVML_ERROR_INVALID_ARGUMENT
   :value: 2


.. py:data:: NVML_ERROR_NOT_SUPPORTED
   :value: 3


.. py:data:: NVML_ERROR_NOT_FOUND
   :value: 6


.. py:data:: NVML_ERROR_LIBRARY_NOT_FOUND
   :value: 12


.. py:data:: NVML_ERROR_FUNCTION_NOT_FOUND
   :value: 13


.. py:data:: NVML_ERROR_UNKNOWN
   :value: 999


.. py:data:: NVML_TEMPERATURE_GPU
   :value: 0


.. py:data:: NVML_DEVICE_MIG_DISABLE
   :value: 0


.. py:data:: NVML_DEVICE_MIG_ENABLE
   :value: 1


.. py:exception:: NVMLError(value=None, msg=None)

   Bases: :py:obj:`Exception`


   Base class mirroring ``pynvml.NVMLError``.

   Carries an integer ``value`` (an ``NVML_ERROR_*`` code) so existing code that
   inspects ``err.value`` continues to work. Concrete per-code subclasses are
   registered in ``_NVML_ERROR_SUBCLASSES`` and surfaced as module attributes
   (e.g. ``NVMLError_NotSupported``), matching upstream pynvml.


   .. py:attribute:: value
      :value: 999



.. py:data:: NVMLError_Uninitialized

.. py:data:: NVMLError_InvalidArgument

.. py:data:: NVMLError_NotSupported

.. py:data:: NVMLError_NoPermission

.. py:data:: NVMLError_NotFound

.. py:data:: NVMLError_InsufficientSize

.. py:data:: NVMLError_DriverNotLoaded

.. py:data:: NVMLError_Timeout

.. py:data:: NVMLError_GpuIsLost

.. py:data:: NVMLError_LibraryNotFound

.. py:data:: NVMLError_FunctionNotFound

.. py:data:: NVMLError_Unknown

.. py:class:: c_nvmlMemory_t(total, free, used)

   NVML ``c_nvmlMemory_t`` return object (VRAM totals, in bytes).

   The class name intentionally matches upstream ``pynvml`` (nvidia-ml-py),
   whose ctypes wrapper for the C ``nvmlMemory_t`` struct is likewise named
   ``c_nvmlMemory_t`` -- so ported code that inspects the type keeps working.
   Returned by `~.nvmlDeviceGetMemoryInfo`.


   .. py:attribute:: total
      :type:  int

      Total installed VRAM, in bytes.



   .. py:attribute:: free
      :type:  int

      Free VRAM currently available, in bytes.



   .. py:attribute:: used
      :type:  int

      VRAM currently in use, in bytes.



.. py:class:: c_nvmlUtilization_t(gpu, memory)

   NVML ``c_nvmlUtilization_t`` return object (engine utilization, percent).

   The class name intentionally matches upstream ``pynvml`` (nvidia-ml-py),
   whose ctypes wrapper for the C ``nvmlUtilization_t`` struct is likewise
   named ``c_nvmlUtilization_t`` -- so ported code that inspects the type keeps
   working. Returned by `~.nvmlDeviceGetUtilizationRates`.


   .. py:attribute:: gpu
      :type:  int

      Percent of the last sampling period the GPU/compute engine was busy.



   .. py:attribute:: memory
      :type:  int

      Percent of the last sampling period the memory controller was busy.



.. py:class:: c_nvmlProcessInfo_t(pid, usedGpuMemory)

   NVML ``c_nvmlProcessInfo_t`` return object (subset used by consumers).

   The class name intentionally matches upstream ``pynvml`` (nvidia-ml-py),
   whose ctypes wrapper for the C ``nvmlProcessInfo_t`` struct is likewise
   named ``c_nvmlProcessInfo_t`` -- so ported code that inspects the type keeps
   working. Returned by `~.nvmlDeviceGetComputeRunningProcesses`.


   .. py:attribute:: pid
      :type:  int

      Operating-system process identifier.



   .. py:attribute:: usedGpuMemory
      :type:  int

      GPU memory used by the process, in bytes. (Upstream NVML may report
      ``None`` under the Windows WDDM driver; not applicable on ROCm.)



.. py:function:: nvmlInit()

   Initialize NVML (maps to ``amdsmi_init`` for AMD GPUs).

   Returns:
       ``None``.

   Raises:
       `~.NVMLError`:
           if AMD SMI initialization or GPU enumeration fails.


.. py:function:: nvmlInitWithFlags(flags)

   Initialize NVML; ``flags`` is accepted for compatibility and ignored.

   Returns:
       ``None``.

   Raises:
       `~.NVMLError`:
           if AMD SMI initialization or GPU enumeration fails.


.. py:function:: nvmlShutdown()

   Shut down NVML (refcounted, maps to ``amdsmi_shut_down``).

   Returns:
       ``None``.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetCount()

   Number of AMD GPUs visible to AMD SMI.

   Returns:
       ``int``:
           the count of AMD GPU devices.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetHandleByIndex(index)

   Return the device handle for the given ordinal.

   Returns:
       An opaque NVML device handle (pass to the ``nvmlDeviceGet*`` queries).

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError_InvalidArgument`:
           if ``index`` is out of range.


.. py:function:: nvmlDeviceGetHandleByUUID(uuid)

   Return the device whose UUID matches ``uuid`` (``str`` or ``bytes``).

   Returns:
       An opaque NVML device handle for the matching device.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError_NotFound`:
           if no device has the requested UUID.


.. py:function:: nvmlDeviceGetIndex(handle)

   NVML ordinal of the device handle.

   Returns:
       ``int``:
           the zero-based device ordinal.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceIsMigDeviceHandle(handle)

   ROCm has no MIG; always reports ``False``.

   Returns:
       ``bool``:
           always ``False`` on ROCm.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)

   No MIG on ROCm; the handle already refers to a full device.

   Returns:
       The same device handle that was passed in.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetMigMode(handle)

   MIG mode pair ``(current, pending)``.

   ROCm has no MIG, so MIG is always disabled. Returns a 2-tuple to match
   NVML, whose callers typically read ``[0]`` for the current mode.

   Returns:
       ``tuple[int, int]``:
           ``(current, pending)`` MIG mode, both always
           `~.NVML_DEVICE_MIG_DISABLE` on ROCm.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetMaxMigDeviceCount(handle)

   Maximum number of MIG devices; always ``0`` on ROCm (no MIG).

   Returns:
       ``int``:
           always ``0`` on ROCm.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.


.. py:function:: nvmlDeviceGetMigDeviceHandleByIndex(device, index)

   ROCm has no MIG instances; always unsupported.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError_NotSupported`:
           always, since ROCm has no MIG.


.. py:function:: nvmlDeviceGetMemoryInfo(handle)

   VRAM totals as an ``nvmlMemory_t``-like object (bytes).

   Returns:
       `~.c_nvmlMemory_t`:
           total/free/used VRAM, in bytes.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if the AMD SMI VRAM-usage query fails.


.. py:function:: nvmlDeviceGetName(handle)

   Marketing/product name as a ``str`` (NVML returns ``str`` on py3).

   Returns:
       ``str``:
           the device market name, falling back to the board product name.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if the AMD SMI ASIC/board queries fail.


.. py:function:: nvmlDeviceGetUUID(handle)

   Device UUID rendered as the conventional ``GPU-<uuid>`` string.

   Returns:
       ``str``:
           the device UUID, prefixed with ``GPU-``.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if the AMD SMI UUID query fails.


.. py:function:: nvmlDeviceGetTemperature(handle, sensorType)

   Current temperature in Celsius for the requested sensor.

   Returns:
       ``int``:
           temperature in degrees Celsius (edge sensor, falling back to the
           hotspot/junction sensor).

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if no supported temperature sensor could be read.


.. py:function:: nvmlDeviceGetPowerUsage(handle)

   Current board power draw in milliwatts.

   Returns:
       ``int``:
           board power draw in milliwatts.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError_NotSupported`:
           if power telemetry is unavailable on this device.


.. py:function:: nvmlDeviceGetUtilizationRates(handle)

   GPU/memory engine utilization percentages.

   Returns:
       `~.c_nvmlUtilization_t`:
           ``gpu`` and ``memory`` busy percentages
           (unsupported fields normalized to ``0``).

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if the AMD SMI activity query fails.


.. py:function:: nvmlDeviceGetCpuAffinity(handle, cpuSetSize)

   CPU affinity bitmask for the device's NUMA node.

   Mirrors NVML's ``nvmlDeviceGetCpuAffinity``: returns a list of
   ``cpuSetSize`` 64-bit words whose bits mark the CPUs local to the GPU.
   Backed by ``amdsmi_get_cpu_affinity_with_scope`` at NUMA-node scope.

   Like NVML, this is a Linux-only capability; on platforms (e.g. Windows) or
   builds where AMD SMI cannot provide it, ``NVMLError_NotSupported`` is
   raised so callers can fall back to a default affinity.

   Returns:
       ``list[int]``:
           ``cpuSetSize`` 64-bit words forming the CPU affinity bitmask.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError_InvalidArgument`:
           if ``cpuSetSize`` is not positive.

       `~.NVMLError_NotSupported`:
           on non-Linux platforms or builds where AMD SMI
           cannot provide CPU affinity.


.. py:function:: nvmlDeviceGetComputeRunningProcesses(handle)

   Compute processes running on the device.

   Drives AMD SMI's two-call (count -> allocate -> fill) process query:
   ``amdsmi_get_gpu_process_list`` reports the running-process count when
   ``max_processes`` is 0 and ``list`` is ``None``; a caller-sized
   ``amdsmi_proc_info_t`` record array (used as a sequence adapter via its
   indexed ``get_*(i)`` accessors) is then filled on the second call.
   Returns an empty list when nothing is running or the platform does not
   support the query.

   Returns:
       ``list`` of `~.c_nvmlProcessInfo_t`:
           one entry per running compute process
           (``pid`` and ``usedGpuMemory`` in bytes); empty if none or unsupported.

   Raises:
       `~.NVMLError_Uninitialized`:
           if NVML was not successfully initialized.

       `~.NVMLError`:
           if the AMD SMI process-list query fails for a reason other
           than being unsupported.


