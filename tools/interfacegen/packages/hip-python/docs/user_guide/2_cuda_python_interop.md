# CUDA Python Interoperability

This chapter discusses HIP Python's CUDA Python interoperability layer that is shipped
in a separate package with the name `hip-python-as-cuda`.
In particular, we discuss how to run existing CUDA Python code on AMD GPUs, and 
if localized modifications are required, how to detect HIP Python and how to fall
back to the underlying HIP Python Python and Cython modules.
Moreover, a technique named "enum constant hallucination" is presented
that allows HIP Python "invent" enum constants and their non-conflicting value
on-the-fly for enum error types.

## Installation

HIP Python's CUDA interoperability layer comes in a separate Python 3 package with the name `hip-python-as-cuda`.
Its sole dependency is the `hip-python` package with the exact same version number.

After having identified the correct package for your ROCm installation, type:

```shell
python3 -m pip install hip-python-as-cuda-<hip_version>.<hip_python_version>
```

or, if you have a HIP Python wheel somewhere in your filesystem, type:

```shell
python3 -m pip install <path/to/hip_python_as_cuda>.whl
```

:::{note}

The first option will only be available after the public release on PyPI.
:::

:::{note}

See <project:#subsec_hip_python_versioning> for more details on the `hip-python`
and `hip-python-as-cuda` version number.
:::

## Basic Usage (Python)

:::{admonition} What will I learn?
* How I can use HIP Python CUDA interoperability modules in my Python code.
:::

After installing the HIP Python package `hip-python-as-cuda`, you can import the individual
modules that you need as shown below:

```{eval-rst}
.. code-block:: py
   :linenos:
   :caption: Importing HIP Python CUDA Interop Modules

    from cuda import cuda
    from cuda import cudart
    from cuda import nvrtc
```

:::{note}
When writing this documentation, only Python and Cython modules for the libraries `cuda` (CUDA Driver), `cudart` (CUDA runtime), and `nvrtc` (NVRTC) were shipped by CUDA Python. Therefore, HIP Python only provides interoperability modules for them and no other CUDA library.
:::

## Python Example

:::{admonition} What will I learn?
How I can run simple CUDA Python application directly on AMD GPUs via HIP Python.
:::

After installing the HIP Python package `hip-python-as-cuda`, you can run the [below example](cuda_stream)
directly on AMD GPUs. There is nothing else to do. 
This works because all CUDA Python functions, types and even enum constants are aliases
of HIP objects. 

:::{admonition} See
{py:obj}`~.cuda.cudaError_t`, {py:obj}`~.cuda.cudaError_t`, {py:obj}`~.cuda.cudaStreamCreate`, {py:obj}`~.cuda.cudaMemcpyAsync`,
{py:obj}`~.cuda.cudaMemsetAsync`,{py:obj}`~.cuda.cudaStreamSynchronize`,{py:obj}`~.cuda.cudaStreamDestroy`,{py:obj}`~.cuda.cudaFree`
:::

```{eval-rst}
.. literalinclude:: ../../examples/1_CUDA_Interop/cuda_stream.py
   :language: python
   :lines: 5-
   :linenos:
   :name: cuda_stream
   :caption: CUDA Python Example
```

:::{admonition} What is happening?
See <project:#sec_hip_streams> for an explanation of the program's steps.
:::

## Enum Constant Hallucination

:::{admonition} What will I learn?

* How I can let HIP Python's enum error types in the CUDA Python interoperability layer "invent"
values for undefined enum constants (that do not conflict with the values of the defined constants).
:::

We use [the example below](cuda_error_hallucinate_enums) to demonstrate how you can deal with scenarios where a CUDA Python program,
which we want to run on AMD GPUs, performs an error check that involves enum constants that are not relevant for HIP programs and/or AMD GPUs.
As HIP Python's routines will never return these enum constants, it is safe to generate values for them on the fly.
Such behavior can be enabled selectively for CUDA Python interop layer enums --- either via the
respective environment variable `HIP_PYTHON_{myenumtype}_HALLUCINATE` and/or at runtime
via `{myenumtype}.hallucinate = True`.

[The example below](cuda_error_hallucinate_enums) fails because there are no HIP analogues to the following constants:

* `cudaError_t.cudaErrorStartupFailure`
* `cudaError_t.cudaError_t.cudaErrorNotPermitted`
* `cudaError_t.cudaErrorSystemNotReady`
* `cudaError_t.cudaErrorSystemDriverMismatch`
* `cudaError_t.cudaErrorCompatNotSupportedOnDevice`
* `cudaError_t.cudaErrorTimeout`
* `cudaError_t.cudaErrorApiFailureBase`

However, the example will run succesfully if you set the environment variable `HIP_PYTHON_cudaError_t_HALLUCINATE` to `1`, `yes`, `y`, or `true` (case does not matter). Alternatively, you could set the class variable {py:obj}`cuda.cudart.cudaError_t.hallucinate` to {py:obj}`True`; 
see <project:#sec_hip_python_specific_code_modifications> on different ways to detect HIP Python in
order to introduce such a modification to your code.

```{eval-rst}
.. literalinclude:: ../../examples/1_CUDA_Interop/cuda_error_hallucinate_enums.py
   :language: python
   :lines: 22-
   :linenos:
   :name: cuda_error_hallucinate_enums
   :caption: CUDA Python Enum Constant Hallucination
```

:::{caution}

Enum constant hallucination should only be used for
error return values and not for enum constants that are passed
as argument to one of the CUDA Python interoperability layer's functions.
:::

## Basic Usage (Cython)

:::{admonition} What will I learn?
* How I can use the CUDA Python interoperability layer's Cython and Python modules in my code.
:::

You can import the Python objects that you need into your `*.pyx` file as shown below:

```{eval-rst}
.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Modules into Cython `*.pyx` file

    from cuda import cuda # enum types, enum aliases, fields
    from cuda import nvrtc
    # ...
```

In the same file, you can **also or alternatively** `cimport` the `cdef` entities
as shown below:

```{eval-rst}
.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Cython declaration files (`*.pxd`) into a Cython `*.pxd` or `*.pyx` file

   from cuda cimport ccuda   # direct access to C interfaces and lazy function loaders
   from cuda cimport ccudart
   from cuda cimport cnvrtc
   ...

    from cuda cimport cuda # access to `cdef class` and `ctypedef` types 
                           # that have been created per C struct/union/typedef
    from cuda cimport cudart
    from cuda cimport nvrtc
    # ...
```

## Cython Example

```{eval-rst}
.. literalinclude:: ../../examples/1_CUDA_Interop/ccuda_stream.pyx
   :language: python
   :lines: 23-
   :linenos:
   :name: ccuda_stream.pyx
   :caption: CUDA Python Cython Program
```

```{eval-rst}
.. literalinclude:: ../../examples/1_CUDA_Interop/setup.py
   :language: python
   :lines: 23-
   :linenos:
   :name: cuda_cython_setup_py
   :caption: Setup Script
```

```{eval-rst}
.. literalinclude:: ../../examples/1_CUDA_Interop/Make
   :language: python
   :lines: 23-
   :linenos:
   :name: cuda_cython_makefile
   :caption: Makefile
```

(sec_hip_python_specific_code_modifications)=
## HIP Python-Specific Code Modifications

:::{admonition} What will I learn?

* That I can use HIP objects (via member variables) when `import`ing the CUDA Python interoperability layer's Python modules.
* That I can access HIP enum constants also via their CUDA interoperability layer type.
* That I can directly use HIP definitions too when `cimport`ing the CUDA Python interoperability layer's Cython modules.
:::

In scenarios where the HIP Python Python or Cython code will need to diverge from the original CUDA Python code, e.g. due
to differences in a signature, we can directly access the underlying HIP Python Python modules
from the CUDA interoperability layer's Python modules as shown in [the example below](detecting_hip_python).

```{eval-rst} 

.. code-block:: python
   :linenos:
   :caption: Various ways to determine if we are working with HIP Python's CUDA Python interoperability layer in Python code.
   :name: detecting_hip_python

   from cuda import cuda # or cudart, or nvrtc
   # [...]
   if "HIP_PYTHON" in cuda:
      # do something (with cuda.hip.<...> or cuda.hip_python_mod.<...>)
   if "hip" in cuda: # or "hiprtc" for nvrtc
      # do something with cuda.hip.<...> (or cuda.hip_python_mod.<...>)
   if hasattr(cuda,"hip"): # or "hiprtc" for nvrtc
      # do something with cuda.hip.<...> (or cuda.hip_python_mod.<...>)
   if "hip_python_mod" in cuda:
      # do something with cuda.hip_python_mod.<...> (or cuda.hip.<...>) # or nvrtc.<...> for nvrtc
   if hasattr(cuda,"hip_python_mod"):
      # do something with cuda.hip_python_mod.<...> (or cuda.hip.<...>) # or nvrtc.<...> for nvrtc
```

Moreover, the interoperability layer's Python enum types also contain all the enum constants of their HIP analogue
as shown

```{eval-rst} 

.. code-block:: cython
   :linenos:
   :caption: Python enum class in cuda.pyx
   :emphasize-lines: 3,5,7,9,11,13,15,17
   
   # [...]
   class CUmemorytype(hip._hipMemoryType__Base,metaclass=_CUmemorytype_EnumMeta):
      hipMemoryTypeHost = hip.chip.hipMemoryTypeHost
      CU_MEMORYTYPE_HOST = hip.chip.hipMemoryTypeHost
      cudaMemoryTypeHost = hip.chip.hipMemoryTypeHost
      hipMemoryTypeDevice = hip.chip.hipMemoryTypeDevice
      CU_MEMORYTYPE_DEVICE = hip.chip.hipMemoryTypeDevice
      cudaMemoryTypeDevice = hip.chip.hipMemoryTypeDevice
      hipMemoryTypeArray = hip.chip.hipMemoryTypeArray
      CU_MEMORYTYPE_ARRAY = hip.chip.hipMemoryTypeArray
      hipMemoryTypeUnified = hip.chip.hipMemoryTypeUnified
      CU_MEMORYTYPE_UNIFIED = hip.chip.hipMemoryTypeUnified
      hipMemoryTypeManaged = hip.chip.hipMemoryTypeManaged
      cudaMemoryTypeManaged = hip.chip.hipMemoryTypeManaged
   # [...]
```




```{eval-rst} 

.. code-block:: cython
   :linenos:
   :emphasize-lines: 2
   :caption: Excerpt from ccuda.pxd

   # [...]
   from hip.chip cimport hipMemcpy
   from hip.chip cimport hipMemcpy as cudaMemcpy
   # [...]
```

```{eval-rst} 

.. code-block:: cython
   :linenos:
   :caption: Excerpt from cuda.pxd
   :emphasize-lines: 2
   
   # [...]
   from hip.hip cimport hipKernelNodeParams # here
   cdef class CUDA_KERNEL_NODE_PARAMS(hip.hip.hipKernelNodeParams):
      pass
   cdef class CUDA_KERNEL_NODE_PARAMS_st(hip.hip.hipKernelNodeParams):
      pass
   cdef class CUDA_KERNEL_NODE_PARAMS_v1(hip.hip.hipKernelNodeParams):
      pass
   cdef class cudaKernelNodeParams(hip.hip.hipKernelNodeParams):
      pass
   # [...]
```





