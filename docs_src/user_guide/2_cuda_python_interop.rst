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
that is shipped in a separate package with the name ``hip-python-as-cuda``. In
particular, we discuss how to run existing CUDA Python code on AMD GPUs, and if
localized modifications are required, how to detect HIP Python and how to fall
back to the underlying HIP Python Python and Cython modules. Moreover, a
technique named "enum constant hallucination" is presented that allows HIP
Python "invent" enum constants and their non-conflicting value on-the-fly for
enum error types.

.. note::

   All examples in this chapter have been tested with ROCm\ |trade| 5.4.3 on Ubuntu 22.
   The :ref:`ch_license` applies to all examples in this chapter.

Installation
------------

HIP Python's CUDA interoperability layer comes in a separate Python 3 package
with the name ``hip-python-as-cuda``. Its sole dependency is the ``hip-python``
package with the exact same version number.

After having identified the correct package for your ROCm\ |trade| installation,
type:

.. code-block:: shell

   python3 -m pip install hip-python-as-cuda-<hip_version>.<hip_python_version>

or, if you have a HIP Python wheel somewhere in your filesystem, type:

.. code-block:: shell

   python3 -m pip install <path/to/hip_python_as_cuda>.whl

.. note::

   The first option will only be available after the public release on PyPI.

.. note::

   See :ref:`subsec_hip_python_versioning` for more details on the ``hip-python``
   and ``hip-python-as-cuda`` version number.

Basic Usage (Python)
--------------------

.. admonition:: What will I learn?

   * How I can use HIP Python's CUDA Python interoperability modules in my Python
     code.

.. note::

   Most links in this tutorial to the CUDA Python interoperability layer API are
   broken. Until we find a way to index the respective Python modules, you must
   unfortunately use the search function for CUDA Python interoperability layer
   symbols.

After installing the HIP Python package ``hip-python-as-cuda``, you can import
the individual modules that you need as shown below:

.. code-block:: py
   :linenos:
   :caption: Importing HIP Python CUDA Interop Modules

   from cuda import cuda
   from cuda import cudart
   from cuda import nvrtc

.. note::

   The legacy ``from cuda import cuda`` (and friends) imports continue
   to work as a compatibility shim. New code should prefer
   ``from cuda.bindings import driver, runtime, nvrtc`` directly,
   matching the modern per-package layout. Both styles are supported
   and not deprecated; the ``cuda.bindings.*`` style is more explicit
   about which package supplies the symbol.

   This mirrors the analogous guidance for the HIP side
   (``from hip import hip`` → ``from rocm.bindings import hip``); see
   the corresponding note in :doc:`1_usage`.

.. note::

   When writing this documentation, only Python and Cython modules for the
   libraries ``cuda`` (CUDA Driver), ``cudart`` (CUDA runtime), and ``nvrtc`` (NVRTC)
   were shipped by CUDA Python. Therefore, HIP Python only provides
   interoperability modules for them and no other CUDA library.

Python Example
--------------

.. admonition:: What will I learn?

   How I can run simple CUDA Python applications directly on AMD GPUs via HIP Python.

After installing the HIP Python package ``hip-python-interop``, you can run the
:ref:`example below <cuda_stream>` directly on AMD GPUs. There is nothing else
to do. This works because all CUDA Python functions, types and even enum
constants are aliases of HIP objects.

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
* ``cudaError_t.cudaError_t.cudaErrorNotPermitted``
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

   from cuda import cuda # enum types, enum aliases, fields
   from cuda import nvrtc
   # ...

In the same file, you can **also or alternatively** ``cimport`` the ``cdef`` entities
as shown below:

.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Cython declaration files (``*.pxd``) into a Cython ``*.pxd`` or ``*.pyx`` file

   from cuda cimport ccuda   # direct access to C interfaces and lazy function loaders
   from cuda cimport ccudart
   from cuda cimport cnvrtc
   ...

   from cuda cimport cuda # access to `cdef class` and `ctypedef` types
                          # that have been created per C struct/union/typedef
   from cuda cimport cudart
   from cuda cimport nvrtc
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
compiler flag ``HIP_Python`` from within the ``setup.py`` script. (We will
discuss how to do so in short). This will replace the qualified ``C++``-like
enum constant expression ``ccudart.cudaError_t.cudaSuccess`` by the ``C``-like
expression ``ccudart.cudaSuccess``.

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
   :emphasize-lines: 38
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
from the original CUDA Python code, e.g. due
to differences in a signature, we can directly access the underlying HIP Python
Python modules from the CUDA interoperability layer's Python modules as shown
in :ref:`the example below <detecting_hip_python>`.

.. code-block:: python
   :linenos:
   :caption: Various ways to determine if we are working with HIP Python's CUDA Python interoperability layer in Python code.
   :name: detecting_hip_python

   from cuda.bindings import driver # or runtime, or nvrtc
   # [...]
   if "HIP_PYTHON" in driver:
      # do something (with driver.hip.<...> or driver.hip_python_mod.<...>)
   if "hip" in driver: # or "hiprtc" for nvrtc
      # do something with driver.hip.<...> (or driver.hip_python_mod.<...>)
   if hasattr(driver,"hip"): # or "hiprtc" for nvrtc
      # do something with driver.hip.<...> (or driver.hip_python_mod.<...>)
   if "hip_python_mod" in driver:
      # do something with driver.hip_python_mod.<...> (or driver.hip.<...>) # or nvrtc.<...> for nvrtc
   if hasattr(driver,"hip_python_mod"):
      # do something with driver.hip_python_mod.<...> (or driver.hip.<...>) # or nvrtc.<...> for nvrtc

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
