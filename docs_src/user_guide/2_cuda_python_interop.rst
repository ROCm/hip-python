.. MIT License
..
.. Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

CUDA\ |reg| Python Interoperability
===================================

This chapter discusses HIP Python's CUDA\ |reg| Python interoperability layer
that is shipped in a separate package with the name ``hip-python-interop``.
In particular, we discuss how to run existing CUDA Python code on AMD GPUs,
and if localized modifications are required, how to detect HIP Python and
how to fall back to the underlying HIP Python Python and Cython modules.
Moreover, a technique named "enum constant hallucination" is presented that
allows HIP Python "invent" enum constants and their non-conflicting value
on-the-fly for enum error types.

.. note::

   All examples in this chapter have been tested with ROCm\ |trade| 7.13 on Ubuntu 22.04.
   The :ref:`ch_license` applies to all examples in this chapter.

Installation
------------

HIP Python's CUDA interoperability layer comes in a separate Python 3
package with the name ``hip-python-interop``. Its sole dependency is the
``rocm-bindings-hip`` package with the exact same version number.

After having identified the correct package for your ROCm\ |trade|
installation, type:

.. code-block:: shell

   python3 -m pip install hip-python-interop~=<rocm_version>

or, if you have a HIP Python wheel somewhere in your filesystem, type:

.. code-block:: shell

   python3 -m pip install <path/to/hip_python_interop>.whl

.. note::

   See :ref:`subsec_hip_python_versioning` for more details on the
   ``hip-python-interop`` version number.

Basic Usage (Python)
--------------------

.. admonition:: What will I learn?

   * How I can use HIP Python's CUDA Python interoperability modules in my Python
     code.

After installing ``hip-python-interop``, you can import the individual
modules that you need as shown below:

.. code-block:: py
   :linenos:
   :caption: Importing the CUDA interop modules

   from cuda.bindings import driver
   from cuda.bindings import runtime
   from cuda.bindings import nvrtc

.. note::

   ``cuda.bindings`` is the package; ``driver`` (CUDA Driver API),
   ``runtime`` (CUDA Runtime API), and ``nvrtc`` (NVRTC) are the
   modules. Importing them under shorter names (``from cuda import
   cuda`` etc., as some older CUDA Python releases used to support)
   is **not** provided by ``hip-python-interop``.

.. note::

   The ``cuda.bindings`` modules above mirror the surface CUDA Python
   itself exposes under ``cuda.bindings``. In addition, the
   ``hip-python-interop`` wheel ships two small convenience
   compatibility shims for common CUDA-ecosystem entry points:
   :ref:`sec_pynvml_shim` (an NVML / ``pynvml`` shim) and
   :ref:`sec_cuda_core_shim` (a minimal ``cuda.core.Device`` shim).

Python Example
--------------

.. admonition:: What will I learn?

   How I can run simple CUDA Python applications directly on AMD GPUs via HIP Python.

After installing ``hip-python-interop`` (and its ``rocm-bindings-hip``
dependency), you can run the :ref:`example below <cuda_stream>` directly
on AMD GPUs. There is nothing else to do. This works because all CUDA
Python functions, types and even enum constants are aliases of HIP
objects.

.. admonition:: See

   :py:obj:`~.cuda.bindings.runtime.cudaError_t`,
   :py:obj:`~.cuda.bindings.runtime.cudaStreamCreate`,
   :py:obj:`~.cuda.bindings.runtime.cudaMemcpyAsync`,
   :py:obj:`~.cuda.bindings.runtime.cudaMemsetAsync`,
   :py:obj:`~.cuda.bindings.runtime.cudaStreamSynchronize`,
   :py:obj:`~.cuda.bindings.runtime.cudaStreamDestroy`,
   :py:obj:`~.cuda.bindings.runtime.cudaFree`

.. literalinclude:: ../../examples/1_CUDA_Interop/cuda_stream.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :name: cuda_stream
   :caption: CUDA Python Example

.. admonition:: What is happening?

   See :ref:`sec_hip_streams` for an explanation of a similar HIP program's
   steps.

.. _sec_pynvml_shim:

NVML compatibility shim (``pynvml``)
------------------------------------

.. admonition:: What will I learn?

   * That ``import pynvml`` keeps working unchanged on AMD GPUs.
   * Which part of the NVML surface the shim implements and how it
     behaves for concepts that do not exist on ROCm\ |trade|.

The ``hip-python-interop`` wheel installs a top-level ``pynvml``
module: a compatibility shim for the NVML Python API backed by AMD
SMI via the high-level :py:obj:`rocm.bindings.amdsmi` bindings. Code
that already speaks NVML --- for example HIP ports of RAPIDS-style
projects --- can keep calling ``pynvml`` unmodified on AMD hardware.

.. note::

   This shim is a fresh, MIT-licensed re-implementation of the subset
   of the NVML Python surface that HIP consumers commonly use. It is
   **not** a fork of the upstream ``pynvml`` / ``nvidia-ml-py``
   package. Because it is backed by AMD SMI, the interop wheel pulls in
   ``rocm-bindings-systems`` (which provides ``rocm.bindings.amdsmi``)
   in addition to ``rocm-bindings-hip``.

The implemented surface covers the entry points HIP ports rely on:

* Lifecycle: ``nvmlInit`` / ``nvmlInitWithFlags`` / ``nvmlShutdown``
  (reference-counted, mapping to ``amdsmi_init`` / ``amdsmi_shut_down``).
* Device enumeration and handles: ``nvmlDeviceGetCount``,
  ``nvmlDeviceGetHandleByIndex``, ``nvmlDeviceGetHandleByUUID``,
  ``nvmlDeviceGetIndex``.
* Device queries: ``nvmlDeviceGetMemoryInfo``, ``nvmlDeviceGetName``,
  ``nvmlDeviceGetUUID``, ``nvmlDeviceGetTemperature``,
  ``nvmlDeviceGetPowerUsage``, ``nvmlDeviceGetUtilizationRates``,
  ``nvmlDeviceGetComputeRunningProcesses``.
* ``nvmlDeviceGetCpuAffinity`` --- like NVML, a Linux-only capability;
  it raises ``NVMLError_NotSupported`` where AMD SMI cannot provide it.
* ``NVMLError`` and the per-code subclasses consumers commonly catch
  (e.g. ``NVMLError_NotSupported``), plus the ``NVML_*`` status and
  selector constants.

.. note::

   ROCm has no Multi-Instance GPU (MIG) concept. The MIG-related
   entry points (``nvmlDeviceIsMigDeviceHandle``,
   ``nvmlDeviceGetMigMode``, ``nvmlDeviceGetMaxMigDeviceCount``, ...)
   are present and degrade gracefully: MIG always reports disabled and
   requests for MIG device handles raise ``NVMLError_NotSupported``.

.. literalinclude:: ../../examples/1_CUDA_Interop/pynvml_query_devices.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :name: pynvml_query_devices
   :caption: Querying AMD GPUs through the NVML (pynvml) API

.. seealso::

   The full API reference for the shim is at :doc:`/python_api/pynvml/index`.

.. _sec_cuda_core_shim:

``cuda.core`` compatibility shim
--------------------------------

.. admonition:: What will I learn?

   * That a minimal ``cuda.core.Device`` is available on AMD GPUs and
     what part of its surface is implemented.

The wheel also provides a minimal ``cuda.core`` shim, backed by the
high-level :py:obj:`rocm.bindings.hip` HIP runtime API. It implements
just enough of the high-level ``cuda.core`` surface for HIP ports of
CUDA Python consumers: a ``Device`` class exposing ``device_id`` and
``uuid``.

.. code-block:: python
   :linenos:
   :caption: Using the cuda.core.Device shim
   :name: cuda_core_device

   from cuda.core import Device

   dev = Device()          # current device; or Device(device_id)
   print(dev.device_id)    # HIP device ordinal
   print(dev.uuid)         # "GPU-<uuid>" string (via hipDeviceGetUuid)

.. caution::

   This is **not** a full port of NVIDIA's ``cuda.core`` /
   ``cuda.core.experimental`` package. Only ``Device.device_id`` and
   ``Device.uuid`` are implemented today; higher-level abstractions
   (``Stream``, ``Buffer``, ``Program``, ...) are intentionally out of
   scope and can be added on demand.

.. seealso::

   The full API reference for the shim is at :doc:`/python_api/cuda/core/index`.

Enum Constant Hallucination
---------------------------

.. admonition:: What will I learn?

   * How I can let HIP Python's enum error types in the CUDA Python
     interoperability layer "invent" values for undefined enum constants (that do
     not conflict with the values of the defined constants).

We use :ref:`the example below <cuda_error_hallucinate_enums>` to demonstrate how
you can deal with scenarios where a CUDA Python program, which we want to run
on AMD GPUs, performs an error check that involves enum constants that are
not relevant for HIP programs and/or AMD GPUs. As HIP Python's routines will
never return these enum constants, it is safe to generate values for them on
the fly. Such behavior can be enabled selectively for CUDA Python
interoperability layer enums --- either via the respective environment
variable ``HIP_PYTHON_{myenumtype}_HALLUCINATE`` and/or at runtime via the
module variable with the same name in :py:obj:`cuda.bindings.driver`,
:py:obj:`cuda.bindings.runtime`, or :py:obj:`cuda.bindings.nvrtc`.

:ref:`The example below <cuda_error_hallucinate_enums>` fails because there are no
HIP analogues to the following constants:

* ``cudaError_t.cudaErrorStartupFailure``
* ``cudaError_t.cudaErrorNotPermitted``
* ``cudaError_t.cudaErrorSystemNotReady``
* ``cudaError_t.cudaErrorSystemDriverMismatch``
* ``cudaError_t.cudaErrorCompatNotSupportedOnDevice``
* ``cudaError_t.cudaErrorTimeout``
* ``cudaError_t.cudaErrorApiFailureBase``

However, the example will run successfully if you set the environment
variable ``HIP_PYTHON_cudaError_t_HALLUCINATE`` to ``1``, ``yes``, ``y``, or ``true``
(case does not matter). Alternatively, you could set the module variable
:py:obj:`cuda.bindings.runtime.HIP_PYTHON_cudaError_t_HALLUCINATE` to :py:obj:`True`;
see :ref:`sec_hip_python_specific_code_modifications` on different ways
to detect HIP Python in order to introduce such a modification to your code.

.. literalinclude:: ../../examples/1_CUDA_Interop/cuda_error_hallucinate_enums.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 8,12,14-16,18,20
   :linenos:
   :name: cuda_error_hallucinate_enums
   :caption: CUDA Python Enum Constant Hallucination

.. caution::

   Enum constant hallucination should only be used for
   error return values and not for enum constants that are passed
   as argument to one of the CUDA Python interoperability layer's functions.

Basic Usage (Cython)
--------------------

.. admonition:: What will I learn?

   * How I can use the CUDA Python interoperability layer's Cython and Python
     modules in my code.

You can import the Python objects that you need into your ``*.pyx`` file as shown below:

.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Modules into Cython ``*.pyx`` file

   from cuda.bindings import driver  # enum types, enum aliases, fields
   from cuda.bindings import runtime
   from cuda.bindings import nvrtc
   # ...

In the same file, you can **also or alternatively** ``cimport`` the ``cdef`` entities
as shown below:

.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Cython declaration files (``*.pxd``) into a Cython ``*.pxd`` or ``*.pyx`` file

   # The cy*-prefixed modules expose the C declarations and the lazy
   # function loaders that delegate to the underlying HIP/HIPRTC C API.
   from cuda.bindings cimport cydriver
   from cuda.bindings cimport cyruntime
   from cuda.bindings cimport cynvrtc
   # ...

   # The non-prefixed modules expose the `cdef class` and `ctypedef`
   # types created per C struct/union/typedef.
   from cuda.bindings cimport driver
   from cuda.bindings cimport runtime
   from cuda.bindings cimport nvrtc
   # ...

Cython Example
--------------

.. admonition:: What will I learn?

   * That I can port CUDA Python Cython code to AMD GPUs with minor
     modifications.
   * How I can introduce different compilation paths for HIP Python's CUDA
     interoperability layer and CUDA Python.

:ref:`The example below <ccuda_stream_pyx>` shows a CUDA Python example that can be
compiled for and run on AMD GPUs. To do so, it is necessary to define the
Cython compile-time flag ``HIP_PYTHON`` from within the ``setup.py``
script (see the script below). This will replace the qualified
``C++``-like enum constant expression
``cyruntime.cudaError_t.cudaSuccess`` by the ``C``-like expression
``cyruntime.cudaSuccess`` (the HIP/HIPRTC Cython enums expose enum
constants directly at module scope, while CUDA Cython enums nest them
inside the enum class).

In the example, the ``DEF`` statement and the ``IF`` and ``ELSE`` statements are
Cython `compile time definitions
<https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#compile-time-definitions>`__
and `conditional statements
<https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#compile-time-definitions>`__,
respectively.

.. literalinclude:: ../../examples/1_CUDA_Interop/cyruntime_cuda_stream.pyx
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 7, 14-17, 22
   :name: ccuda_stream_pyx
   :caption: CUDA Python Cython Program (cyruntime variant)

.. admonition:: What is happening?

   See :ref:`sec_hip_streams` for an explanation of a similar HIP Python
   program's steps.

The example can be compiled for AMD GPUs via the following
:ref:`setup.py script <cuda_cython_setup_py>`,
which specifies ``compile_time_env=dict(HIP_PYTHON=True)`` as keyword parameter
of the :py:obj:`~.cythonize` call in line

.. literalinclude:: ../../examples/1_CUDA_Interop/setup.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 39
   :name: cuda_cython_setup_py
   :caption: Setup Script

For your convenience, you can use the :ref:`Makefile below <cuda_cython_makefile>`
to build a Cython module in-place (via ``make build``) and run the code (by
importing the module via ``make run``).

.. literalinclude:: ../../examples/1_CUDA_Interop/Makefile
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :name: cuda_cython_makefile
   :caption: Makefile

.. _sec_hip_python_specific_code_modifications:

HIP Python-Specific Code Modifications
--------------------------------------

.. admonition:: What will I learn?

   * That I can use HIP objects (via member variables) when ``import``\ ing the CUDA
     Python interoperability layer's Python modules.
   * That I can access HIP enum constants also via their CUDA interoperability
     layer type.
   * That I can directly use HIP definitions too when ``cimport``\ ing the CUDA
     Python interoperability layer's Cython modules.

In scenarios where the HIP Python Python or Cython code will need to diverge
from the original CUDA Python code, e.g. due to differences in a signature,
we can directly access the underlying HIP Python Python modules from the
CUDA interoperability layer's Python modules. Each of
:py:obj:`cuda.bindings.driver`, :py:obj:`cuda.bindings.runtime`, and
:py:obj:`cuda.bindings.nvrtc` exposes the following module-level
attributes when served by HIP Python's interop layer:

* ``HIP_PYTHON`` — the literal :py:obj:`True`. Absent when the user is
  running NVIDIA's stock ``cuda-python`` package instead.
* ``hip_python_mod`` — a reference to the underlying HIP Python module
  (:py:obj:`rocm.bindings.hip` for ``driver`` / ``runtime``,
  :py:obj:`rocm.bindings.hiprtc` for ``nvrtc``).
* ``hip`` — same module as ``hip_python_mod``, exposed under the short
  name on the ``driver`` and ``runtime`` modules. *Not* exposed on
  ``nvrtc`` (which uses ``hiprtc`` instead — see below).
* ``hiprtc`` — same module as ``hip_python_mod``, exposed under the
  short name on ``nvrtc``. Not exposed on ``driver`` / ``runtime``.

Use :py:func:`hasattr` (or ``__dict__`` membership) to test for these.
Module-level :py:obj:`in` does **not** work — Python modules are not
iterable, so ``"HIP_PYTHON" in driver`` raises :py:obj:`TypeError`.

.. code-block:: python
   :linenos:
   :caption: Detecting HIP Python's CUDA interoperability layer at runtime
   :name: detecting_hip_python

   from cuda.bindings import driver  # or runtime; or nvrtc

   # Preferred test:
   if getattr(driver, "HIP_PYTHON", False):
       # We are running on HIP Python's interop layer.
       hip = driver.hip            # or driver.hip_python_mod  (same module)
       # ... use HIP-only APIs as needed ...

   # Equivalent forms:
   if hasattr(driver, "HIP_PYTHON"):
       ...
   if "HIP_PYTHON" in driver.__dict__:
       ...

   # For `nvrtc`, the underlying module is exposed as `hiprtc` (no
   # `hip` attribute), but `hip_python_mod` works on all three modules:
   from cuda.bindings import nvrtc
   if getattr(nvrtc, "HIP_PYTHON", False):
       hiprtc = nvrtc.hiprtc       # or nvrtc.hip_python_mod  (same module)

Moreover, the interoperability layer's Python enum types also contain all the
enum constants of their HIP analogue as shown in the
:ref:`snippet below <snippet_cuda_enum>`.

.. code-block:: cython
   :linenos:
   :caption: Python enum class in cuda/bindings/driver.pyx
   :emphasize-lines: 3,5,7,9,11,13
   :name: snippet_cuda_enum

   # [...]
   class CUmemorytype(hip._hipMemoryType__Base,metaclass=_CUmemorytype_EnumMeta):
      hipMemoryTypeHost = rocm.bindings.cyhip.hipMemoryTypeHost
      CU_MEMORYTYPE_HOST = rocm.bindings.cyhip.hipMemoryTypeHost
      cudaMemoryTypeHost = rocm.bindings.cyhip.hipMemoryTypeHost
      hipMemoryTypeDevice = rocm.bindings.cyhip.hipMemoryTypeDevice
      CU_MEMORYTYPE_DEVICE = rocm.bindings.cyhip.hipMemoryTypeDevice
      cudaMemoryTypeDevice = rocm.bindings.cyhip.hipMemoryTypeDevice
      hipMemoryTypeArray = rocm.bindings.cyhip.hipMemoryTypeArray
      CU_MEMORYTYPE_ARRAY = rocm.bindings.cyhip.hipMemoryTypeArray
      hipMemoryTypeUnified = rocm.bindings.cyhip.hipMemoryTypeUnified
      CU_MEMORYTYPE_UNIFIED = rocm.bindings.cyhip.hipMemoryTypeUnified
      hipMemoryTypeManaged = rocm.bindings.cyhip.hipMemoryTypeManaged
      cudaMemoryTypeManaged = rocm.bindings.cyhip.hipMemoryTypeManaged
   # [...]

In the ``cy``-prefixed Cython declaration files
(``cuda.bindings.cydriver.pxd``, ``cuda.bindings.cyruntime.pxd``, and
``cuda.bindings.cynvrtc.pxd``), you will further find that the
:ref:`HIP functions and union/struct types are directly included too <ccuda_hip_names>`:

.. code-block:: cython
   :linenos:
   :emphasize-lines: 2, 5
   :caption: Excerpt from cuda/bindings/cydriver.pxd
   :name: ccuda_hip_names

   # [...]
   from rocm.bindings.cyhip cimport hipDeviceProp_t
   from rocm.bindings.cyhip cimport hipDeviceProp_t as cudaDeviceProp
   # [...]
   from rocm.bindings.cyhip cimport hipMemcpy
   from rocm.bindings.cyhip cimport hipMemcpy as cudaMemcpy
   # [...]

In the Cython declaration files without ``cy``-prefix
(``cuda.bindings.driver.pxd``, ``cuda.bindings.runtime.pxd``, and
``cuda.bindings.nvrtc.pxd``), you will discover that the original HIP types
(only those derived from unions and structs) are ``cimport``\ ed too and
that the CUDA interoperability layer types are made subclasses of the
respective HIP type; see :ref:`the example below <cuda_hip_names>`. This allows
to pass them to the CUDA interoperability layer's Python functions, i.e., the
aliased HIP Python functions.

.. code-block:: cython
   :linenos:
   :caption: Excerpt from cuda/bindings/driver.pxd
   :emphasize-lines: 2,3,5,7,9
   :name: cuda_hip_names

   # [...]
   from rocm.bindings.hip cimport hipKernelNodeParams # here
   cdef class CUDA_KERNEL_NODE_PARAMS(hipKernelNodeParams):
      pass
   cdef class CUDA_KERNEL_NODE_PARAMS_st(hipKernelNodeParams):
      pass
   cdef class CUDA_KERNEL_NODE_PARAMS_v1(hipKernelNodeParams):
      pass
   cdef class cudaKernelNodeParams(hipKernelNodeParams):
      pass
   # [...]

.. |reg| unicode:: U+000AE .. REGISTERED SIGN
   :ltrim:
.. |trade| unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
