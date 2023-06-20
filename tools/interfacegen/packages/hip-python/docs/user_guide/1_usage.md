# Basic Usage and Examples

This chapter explains how to use HIP Python's main interfaces. 
The usage of the CUDA interoperability layer is discussed in a separate chapter.
We first aim to give an introduction to the *Python* API of HIP Python 
by means of basic examples before discussing the *Cython* API in the 
last sections of this chapter.

:::{note}
All examples in this chapter have been tested with ROCm 5.4.3.
:::

## Basic Usage (Python)

:::{admonition} What will I learn?

* How to use HIP Python modules in your Python code.
:::

After installing the HIP Python package `hip-python`, you can import the individual
modules that you need as shown below:

```{eval-rst}
.. code-block:: py
   :linenos:
   :caption: Importing HIP Python Modules

    from hip import hip
    from hip import hiprtc
    # ...
```

And you are ready to go!

(sec_obtaining_device_properties)=
## Obtaining Device Properties

:::{admonition} What will I learn?

* How I can obtain device attributes/properties via {py:obj}`~.hipGetDeviceProperties`.
* How I can obtain device attributes/properties via {py:obj}`~.hipDeviceGetAttribute`.
:::

Obtaining device properties such as the architecture or the number of compute units
is important for many applications.

### Via {py:obj}`~.hipGetDeviceProperties`

A number of device properties can be obtained via the 
{py:obj}`~.hipDeviceProp_t` object. After creation (line 16) this object must
be passed to the {py:obj}`~.hipGetDeviceProperties` routine (line 17).
The second argument (`0`) is the device number.

Running the {ref}`example below <hip_deviceproperties>` will print out the values of all queried device properties 
before the program eventually prints `"ok"` and quits.

:::{note}
The `hip_check` routine in the snippet unpacks the result tuple -- HIP Python routines always return a tuple, then checks the therein contained error code (first argument),
and finally returns the rest of the tuple -- either as single value or tuple sans error code. Such error check routines will be used throughout this and the following sections.
:::

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hip_deviceproperties.py
   :language: python
   :lines: 5-
   :emphasize-lines: 12-13
   :linenos:
   :name: hip_deviceproperties
   :caption: Obtaining Device Properties via hipGetDeviceProperties
```

### Via {py:obj}`~.hipDeviceGetAttribute`

You can also obtain some of the properties that appeared in the 
{ref}`previous example <hip_deviceproperties>` plus a number of additional properties via the {py:obj}`~.hipDeviceGetAttribute` routine as shown in the {ref}`example below <hip_deviceattributes>` (line 130).
In the example below, we query integer-type device attributes/properties.
Therefore, we supply the address of a {py:obj}`ctypes.c_int` variable
as first argument. The respective property, the second argument, is passed
as enum constant of type {py:obj}`~.hipDeviceAttribute_t`.

Running this example will print out the values of all queried device
attributes before the program prints `"ok"` and quits.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hip_deviceattributes.py
   :language: python
   :lines: 5-
   :emphasize-lines: 125-126
   :linenos:
   :name: hip_deviceattributes
   :caption: Obtaining Device Properties via hipDeviceGetAttribute
```

(sec_hip_streams)=
## HIP Streams

:::{admonition} What will I learn?

* How I can use HIP Python's {py:obj}`~.hipStream_t` objects and the associated HIP Python routines.
* That I can directly pass Python 3 {py:obj}`~array.array` objects to HIP runtime routines such
  as {py:obj}`hipMemcpy` and {py:obj}`hipMemcpyAsync`.
:::

An important concept in HIP are streams. They allow
to overlap host and device work as well as device computations
with data movement to or from that same device.

The {ref}`below example <hip_stream>` showcases how to use HIP Python's {py:obj}`~.hipStream_t` objects and the associated HIP Python routines.
The example further demonstrates that you can pass Python 3 {py:obj}`array.array` types
directly to HIP Python interfaces that expect an host buffer. One example of such
interfaces is {py:obj}`~.hipMemcpyAsync` (lines 27 and 28).

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hip_stream.py
   :language: python
   :lines: 5-
   :linenos:
   :name: hip_stream
   :caption: HIP Streams
```

:::{admonition} What is happening?

1. A host buffer will be filled with random numbers before
it is asynchronously copied to the device (line 27), where a asynchronous {py:obj}`~.hipMemsetAsync` (same stream) resets all bytes to `0` (line 28). 
1. An asynchronous memcpy (same stream) is then issued to copy the device data back to the host (line 29).
All operations within the stream are executed in order.
1. As the `~Async` operations are non-blocking, the host waits via {py:obj}`~.hipStreamSynchronize`
until operations in the stream have been completed (line 30) before deleting
the stream (line 31). 
1. Eventually the program deallocates device data via
{py:obj}`~.hipFree` and checks if all bytes in the host buffer are now set to `0`.
If so, it quits with an "ok".
:::

(sec_launching_kernels)=
## Launching Kernels

:::{admonition} What will I learn?

* How I can compile a HIP C++ kernel at runtime via {py:obj}`~.hiprtcCompileProgram`.
* How I can launch kernels via {py:obj}`~.hipModuleLaunchKernel`.
:::

HIP Python does not provide the necessary infrastructure to express device code
in native Python. However, you can compile and launch kernels from within Python code 
via the just-in-time (JIT) compilation interface provided by HIP Python module {py:obj}`hip.hiprtc` together with
the kernel launch routines provided by HIP Python module {py:obj}`hip.hip`.
The {ref}`example below <hiprtc_launch_kernel_no_args>` demonstrates how to do so.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hiprtc_launch_kernel_no_args.py
   :language: python
   :lines: 5-
   :linenos:
   :emphasize-lines: 20, 25, 34, 42-44, 47-55
   :name: hiprtc_launch_kernel_no_args
   :caption: Compiling and Launching Kernels
```

:::{admonition} What is happening?

1. In the example, the kernel `print_tid` defined within the string `source` simply prints the
block-local thread ID (`threadIDx.x`) for every thread running the kernel (line 20).
1. A program `prog` is then created in line 25 via {py:obj}`~.hiprtcCreateProgram`, where we pass `source` as first argument,
we further give the kernel a name (note the `b".."`), specify zero headers and include names (last three arguments).
1. Next we query the architecture name via {py:obj}`~.hipGetDeviceProperties` (more details: <project:#sec_obtaining_device_properties>)
and use it in lines 33-34, where we specify compile flags (`cflags`) and
compile `prog` via {py:obj}`~.hiprtcCompileProgram`.
In case of a failure, we obtain the program log and raise it as {py:obj}`~.RuntimeError`.
1. In case of success, we query the code size via {py:obj}`~.hiprtcGetCodeSize`, create a buffer with that information,
and then copy the code into this buffer via {py:obj}`~.hiprtcGetCode`.
Afterwards, we load the code as `module` via {py:obj}`~.hipModuleLoadData` and then
obtain our device kernel with name `"print_tid"` from it via {py:obj}`~.hipModuleGetFunction`.
1. This object is then passed as first argument to the {py:obj}`~.hipModuleLaunchKernel` routine,
followed by the usual grid and block dimension triples, the size of the required shared memory,
and stream to use (`None` means the null stream).
The latter two arguments, `kernelParams` and `extra`, can be used for passing kernel arguments.
We will take a look how to pass kernel arguments via `extra` in the next section.
1. After the kernel launch, the host waits on completion via {py:obj}`~.hipDeviceSynchronize`
and then unloads the code module again via  {py:obj}`hipModuleUnload` before quitting
with an `"ok"`.
:::

## Kernels with Arguments

:::{admonition} What will I learn?

How I can pass arguments to {py:obj}`~.hipModuleLaunchKernel`.
:::

One of the difficulties that programmers face when attempting to launch 
kernels via {py:obj}`~.hipModuleLaunchKernel` is passing arguments to the kernels.
When using the `extra` argument, the kernel arguments must be aligned in a certain way.
In C/C++ programs, one can simply put all arguments into a struct and let the 
compiler take care of the argument alignment. Similarly, 
one could create a {py:obj}`ctypes.Structure` in python to do the same.

However, we do not want to oblige HIP Python users with creating such glue code.
Instead, users can directly pass a {py:obj}`list` or {py:obj}`tuple` of arguments to 
the {py:obj}`~.hipModuleLaunchKernel`. The entries of these objects must 
either be of type {py:obj}`~.DeviceArray` (or can be converted to {py:obj}`~.DeviceArray`)
or one of the {py:obj}`ctypes` types.

The former are typically the result of a {py:obj}`~.hipMalloc` call (or similar memory allocation routines).
Please also see <project:#ch_datatypes> for details on what other types can be converted to {py:obj}`~.DeviceArray`.
The latter are typically used to convert a scalar of the python {py:obj}`bool`, {py:obj}`int`, and {py:obj}`float` scalar types 
to a fixed precision.

The [below example](hiprtc_launch_kernel_args) demonstrates the usage of {py:obj}`~.hipModuleLaunchKernel`
by means of a simple kernel, which scales a vector by a factor.
Here, We pass multiple arguments that require different alignments to the aforementioned routine
in lines 80-95. We insert some additional `unused*` arguments into the `extra` {py:obj}`tuple` to stress the
argument buffer allocator. Note the {py:obj}`ctypes` object construction for scalars and the direct passing of the device array `x_d`. 
Compare the argument list with the signature of the kernel defined in line 23.
The example also introduces HIP Python's {py:obj}`~.dim3` struct (default value per dimension is 1), which can be unpacked just
like a {py:obj}`tuple` or {py:obj}`list`.
:::

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
   :language: python
   :lines: 5-
   :emphasize-lines: 85-90, 72-73, 23
   :linenos:
   :name: hiprtc_launch_kernel_args
   :caption: Compiling and Launching Kernels With Arguments
```

:::{admonition} What is happening?

See the previous section [Launching Kernels](sec_launching_kernels) for 
a textual description of the main steps.
:::

## hipBLAS and NUMPY Interoperability

:::{admonition} What will I learn?

* How I can use HIP Python's {py:obj}`~.hipblas` module.
* That I can pass {py:obj}`numpy` arrays to HIP runtime routines such
  as {py:obj}`hipMemcpy` and {py:obj}`hipMemcpyAsync`.
:::

[This example](hipblas_with_numpy) demonstrates how to initialize and use HIP Python's {py:obj}`~.hipblas`
module. Furthermore, it shows that you can simply pass {py:obj}`numpy` arrays to HIP runtime routines such
as {py:obj}`hipMemcpy` and {py:obj}`hipMemcpyAsync`. This works because some of HIP Python's interfaces 
support automatic conversion from various different types---in particular such types that implement the Python [buffer protocol](https://docs.python.org/3/c-api/buffer.html). The {py:obj}`~numpy.numpy` arrays implement the Python buffer protocol
and thus can be directly passed to those interfaces.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hipblas_with_numpy.py
   :language: python
   :lines: 5-
   :emphasize-lines: 23-24, 35-36, 39-41, 44
   :linenos:
   :name: hipblas_with_numpy
   :caption: hipBLAS and NUMPY Interoperability 
```

:::{admonition} What is happening?

1. We initialize two `float32`-typed {py:obj}`numpy` arrays `x_h` and `y_h` on the host and fill them with random data (lines 23-24).
1. We compute the expected result on the host via {py:obj}`numpy` array operations (line 27).
1. We allocate device analogues for `x_h` and `y_h` (lines 31-32) and copy the host data over (line 35-36).
   Note that we can directly pass the {py:obj}`numpy` arrays `x_h` and `y_h` to {py:obj}`~.hipMemcpy`.
1. Before being able to call one of the compute routines of {py:obj}`~.hipblas`, it's necessary to create a {py:obj}`~.hipblas` handle via
{py:obj}`~.hipblasCreate` that will be passed to every {py:obj}`~.hipblas` routine as first argument (line 39).
1. In line 40 follows the call to {py:obj}`~.hipblasSaxpy`, where we pass the handle as first argument and the address of host
{py:obj}`ctypes.c_float` variable `alpha` as third argument.
1. In line 41 the handle is destroyed via {py:obj}`~.hipblasDestroy` because it is not needed anymore.
1. The device data is downloaded in line 44. where we pass `numpy` array `y_h` as destination array.
1. We compare the expected host result with the downloaded device result (lines 47-50) and print `"ok"` if all is fine.
1. Finally, we deallocate the device arrays in lines 55-56.
:::

## HIP Python Device Arrays

:::{admonition} What will I learn?

* How I can change the shape and datatype of HIP Python's {py:obj}`~.DeviceArray` objects.
* How I can obtain subarrays from HIP Python's {py:obj}`~.DeviceArray` objects --- which are again of that type --- via array subscript.
:::

[This example](hip_python_device_array) demonstrates how to {py:obj}`~.DeviceArray.configure` the shape and data type
of a {py:obj}`~.DeviceArray` returned by {py:obj}`~.hipMalloc` (and related routines).
It further shows how to retrieve single elements / contiguous subarrays
with respect to the specified type and shape information.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hip_python_device_array.py
   :language: python
   :lines: 16-
   :linenos:
   :emphasize-lines: 18, 19, 23-25, 31
   :name: hip_python_device_array
   :caption: Configuring and Slicing HIP Python's DeviceArray
```

:::{admonition} What is happening?

1. A two-dimensional row-major array of size `(3,20)` (line 18) is created on the host.
   All elements are initialized to `1` (line 19).
1. A device array with the same number of bytes is created on the device (line 23).
1. The device array is reconfigured to have `float32` type and the shape of
   the host array (line 23-25).
1. The host data is copied to the device array (line 26).
1. Within a loop over the row indices (index: `r`):
   1. A pointer to row with index `r` is created via array subscript (line 31). This yields `row`.
   1. `row` is passed to a {py:obj}`hipblasSscal` call that
      writes index `r` to all elements of the row (line 34).
1. Data is copied back from the device to the host array.
1. Finally, a check is performed on the host if the row values equal the respective row index (lines 44-50).
:::

:::{note}
Please also see <project:#ch_datatypes> for more details on the capabilities of type {py:obj}`~.DeviceArray`
and the [CUDA Array interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) 
that it implements.
:::

## Monte Carlo with hipRAND

:::{admonition} What will I learn?
* How I can create an {py:obj}`~.hiprand` random number generator via {py:obj}`hiprandCreateGenerator`.
* How I can generate uniformly-distributed random numbers via {py:obj}`~.hiprandGenerateUniformDouble`.
:::

This [example](hiprand_monte_carlo_pi) uses {py:obj}`~/.hiprand` to estimate {math}`\pi` by means of the Monte-Carlo method.

:::{admonition} Background

The unit square has the area {math}`1^2`, while the unit circle has the area
{math}`\pi\,(\frac{1}{2})^2`. Therefore, the ratio between the latter and the former area is {math}`\frac{\pi}{4}`.
Using the Monte-Carlo method, we randomly choose {math}`N` {math}`(x,y)`-coordinates in the unit square.
We then estimate the ratio of areas as the ratio between the number of samples located within the unit circle and
the total number of samples {math}`N`. The accuracy of the approach increases with {math}`N`.
:::

:::{note}
This example was derived from a similar example in the [rocRAND repository on Github](https://github.com/ROCmSoftwarePlatform/rocRAND/tree/develop). See this repository for another higher-level interface to hiprand/rocrand ({py:obj}`ctypes`-based, no Cython interfaces).
:::

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hiprand_monte_carlo_pi.py
   :language: python
   :lines: 21-
   :emphasize-lines: 22, 24, 27
   :linenos:
   :name: hiprand_monte_carlo_pi
   :caption: Monte Carlo with hipRAND
```

:::{admonition} What is happening?
Within a loop that per iteration multiplies the problem size `n` by `10` (line 37-38), we call a function 
`calculate_pi` with  `n` as argument, in which:

1. We first create a two-dimensional host array `xy` of type `double` with `n` elements (line 21).
1. We then create a {py:obj}`~.hiprandCreateGenerator` generator of type
{py:obj}`~.hiprandRngType.HIPRAND_RNG_PSEUDO_DEFAULT` (line 22).
1. We create a device array `xy_d` that stores the same number of bytes as `xy`.
1. We fill `xy_d` with random data via {py:obj}`~.hiprandGenerateUniformDouble` (line 24).
1. We then copy to `xy` from `xy_d` and free `x_d` (lines 25-26) and destroy the generator (line 27).
1. We use `numpy` array operations to count the number of random-generated :math:`x-y`-coordinates within the unit circle
   (lines 29-30).
1. Finally, we compute the ratio estimate for the given `n` and return it (lines 31-32).
:::

## A simple complex FFT with hipFFT

:::{admonition} What will I learn?
* How I can create an {py:obj}`~.hipfft` 1D plan via {py:obj}`~.hipfftPlan1d`.
* How I can run a complex in-place forward FFT via {py:obj}`~.hipfftExecZ2Z`.
:::

[This example](hipfft) demonstrates the usage of HIP Python's {py:obj}`~.hipfft` library.

We perform a double-complex-to-double-complex in-place forward FFT
of a constant time signal {math}`f(t) = 1-1j` of which we have {math}`N` samples.
The resulting FFT coefficients are all zero --- aside from the first one, which has the value {math}`N-Nj`.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/hipfft.py
   :language: python
   :lines: 14-
   :linenos:
   :emphasize-lines: 25,28
   :name: hipfft
   :caption: A simple complex FFT with hipFFT
```

:::{admonition} What is happening?

1. We start with creating the initial data in lines 17-18, where we use {py:obj}`~numpy.numpy` for convenience.
1. We then create a device array of the same size and copy the device data over (lines 21-22).
1. We create a plan in line 25, where we specify the number of samples `N` and the the type
   of the FFT as *double-complex-to-double-complex*, {py:obj}`~.hipfftType.HIPFFT_Z2Z`.
1. Afterwards, we execute the FFT in-place (`idata=dx` and `odata=dx`) and specify that 
   we run an forward FFT, {py:obj}`hipfft.HIPFFT_FORWARD` (line 28).
1. The host then waits for completion of all activity on the device before copying data back to the
   host and freeing the device array (lines 29-33).
1. Finally, we check if the result is as expected and print `"ok"` if that's the case (lines 35-42).
:::

## A multi-GPU broadcast with RCCL

:::{admonition} What will I learn?
* How I can create a multi-GPU communicator via {py:obj}`~.ncclCommInitAll`.
* How I can destroy a communicator again via {py:obj}`~.ncclCommDestroy`.
* How I can open and close a communication group via  {py:obj}`~.ncclGroupStart` and
  {py:obj}`~.ncclGroupEnd`, respectively.
* How I can perform a broadcast via {py:obj}`~.ncclBcast`.
:::

[This example](rccl_comminitall_bcast) implements a single-node multi-GPU broadcast of a small array
from one GPU's device buffer to that of the other ones.

```{eval-rst}
.. literalinclude:: ../../examples/0_Basic_Usage/rccl_comminitall_bcast.py
   :language: python
   :lines: 5-
   :linenos:
   :emphasize-lines: 17-19, 36-37, 34,39, 55
   :name: rccl_comminitall_bcast
   :caption: A multi-GPU broadcast with RCCL
```

:::{admonition} What is happening?
1. In line 17, we use the device count `num_gpus` (via {py:obj}`~.hiphipGetDeviceCount`) to create an array
of pointers (same size as `unsigned long`, `dtype="uint64"`). This array named `comms` 
is intended to store a pointer to each device's communicator.
1. We then create an array of device identifiers (line 18).
1. We pass both arrays to {py:obj}`~.ncclCommInitAll`  as first and last argument, respectively (line 19).
   The second element is the device count.
   The aforementioned routine initializes all communicators and writes their address to the `comms` array.
1. In lines 22-28, we create an array `dx` on each device of size `N` that is initialized with zeros on all devices
except device `0`. The latter's array is filled with ones.
1. We start a communication group in line 34, and then call {py:obj}`~.ncclBcast` per device in line 37.
   The first argument of the call is per-device `dx`, the second the size of `dx`.
   Then follows the {py:obj}`~.ncclDataType_t`, the root (device `0`), then the communicator (`int(comms[dev])`)
   and finally the stream ({py:obj}`None`).
   Casting `comms[dev]` is required as the result is otherwise interpreted as single-element `Py_buffer`
   by HIP Python's {py:obj}:`~.ncclBcast` instead of as an address.
1. In line 39, we close the communication group again.
1. We download all data to the host per device and check if the elements are set to `1` (lines 42-50).
   Otherwise, a runtime error is emitted.
1. Finally, we clean up by deallocating all device memory and destroying the per-device communicators
   via {py:obj}:`~.ncclCommDestroy` in line 55. Note that here again the `comm` must be converted to `int` before passing it to the HIP Python routine.
:::

:::{note}
Please also see <project:#ch_datatypes> for more details on automatic type conversions
supported by HIP Python's datatypes.
:::

## Basic Usage (Cython)

:::{admonition} What will I learn?
* How I can use HIP Python's Cython modules in my Cython code.
* How to compile my Cython code that uses HIP Python's Cython modules.
:::

In this section, we show how to use HIP Python's [Cython](https://cython.org/) modules 
and how to compile projects that use them.

### Cython Recap

:::{note}
This section expects that the user has at least some basic knowledge about
the programming language Cython.
If you are unfamiliar with the language, we refer to the
[Cython tutorials](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html)
and the [Language Basics page](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html).
:::

Cython modules are often split into a `*.pxd` and a `*.pyx` file, which are a Cython module's
declaration and implementation part respectively.
While the former files are to some degree comparable to header files in C/C++,
the latter can be compared to sources files.
The declaration part may only contain `cdef` fields, variables, and function prototypes
while the implementation part may contain the implementation of those entities
as well as Python fields, variables, and functions.

The implementation part is the interface between the C/C++ and the Python world.
Here, you can import Python code via Python's `import` statements,
you can *C-import* `cdef` declarations from other Cython 
declaration files (`*.pxd`) via `cimport` statements,
and you can include C/C++ declarations from C/C++ header files as `cdef` declarations.

To build a Python module from a Cython module, 
the implementation part must be first "cythonized", i.e. converted into a C/C++ file
and then compiled with a compiler. It is recommended to use the compiler that
was used for compiling the used python interpreter.
Most people don't do this manually but instead prefer to use the build infrastructure
provided by {py:obj}`setuptools`. They then write a `setup.py` script 
that contains the code that performs the aforementioned two tasks.

### Cython modules in HIP Python

Per Python module {py:obj}`hip.hip`, {py:obj}`hip.hiprtc`, ... ,
HIP Python ships an additional `c`-prefixed `hip.c<pkg_name>` module.

* The module *without* the `c` prefix is compiled into the interface for HIP Python's
Python users. However, all `cdef` declarations therein can also be `cimport`ed by Cython users (typically `cdef class` declarations) and all Python objects therein can be `import`ed by Cython users too (typically enum and function objects).
* The module *with* the `c` prefix builds the bridge to the underlying HIP C library by including C definitions from the corresponding header files. This code is located in the declaration part. This part further declares
runtime function loader prototypes. The definition of these function loaders in the implementation part
first try to load the underlying C library and then if successful, try to load the function symbol from
that shared object.

:::{note}
The lazy-loading of functions at runtime can, under some circumstances, allow to use a HIP Python version that 
covers a superset or only a subset of the functions available within the respective library of a ROCm installation.
:::

### Using the Cython API

You can import the Python objects that you need as shown below:

```{eval-rst}
.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Modules into Cython `*.pyx` file

    from hip import hip # enum types, enum aliases, fields
    from hip import hiprtc
    # ...
```

In the same file, you can **also or alternatively** `cimport` the `cdef` entities
as shown below:

```{eval-rst}
.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Cython declaration files (`*.pxd`) into a Cython `*.pxd` or `*.pyx` file

   from hip cimport chip   # direct access to C interfaces and lazy function loaders
   from hip cimport chiprtc 
   ...

    from hip cimport hip # access to `cdef class` and `ctypedef` types 
                         # that have been created per C struct/union/typedef
    from hip cimport hiprtc
    # ...
```

### Compiling a Cython module

After having written your own `mymodule.pyx` file that uses HIP Python's Cython API,
you can compile the result using a `setup.py` script as [shown below](cython_setup_py). In the `setup.py` script, we only assume that HIP or HIPRTC is used. Therefore, only `amdhip64` is put into the `rocm_libs` list. 
It is further important to specify the HIP Platform as the header files from which we include the C interfaces will be included at compile time by the underlying C/C++ compiler. The compilation path must include all these interfaces.

```{eval-rst}
.. code-block:: python
   :linenos:
   :caption: Compiling a Cython module that uses HIP Python's Cython API.
   :name: cython_setup_py
   
   import os, sys

   mymodule = "mymodule"

   from setuptools import Extension, setup
   from Cython.Build import cythonize

   ROCM_PATH=os.environ.get("ROCM_PATH", "/opt/rocm")
   HIP_PLATFORM = os.environ.get("HIP_PLATFORM", "amd")

   if HIP_PLATFORM not in ("amd", "hcc"):
      raise RuntimeError("Currently only HIP_PLATFORM=amd is supported")

   def create_extension(name, sources):
      global ROCM_PATH
      global HIP_PLATFORM
      rocm_inc = os.path.join(ROCM_PATH,"include")
      rocm_lib_dir = os.path.join(ROCM_PATH,"lib")
      platform = HIP_PLATFORM.upper()
      cflags = ["-D", f"__HIP_PLATFORM_{platform}__"]
   
      return Extension(
         name,
         sources=sources,
         include_dirs=[rocm_inc],
         library_dirs=[rocm_lib_dir],
         libraries=rocm_libs,
         language="c",
         extra_compile_args=cflags,
      )

   setup(
      ext_modules = cythonize(
         [create_extension(mymodule, [f"{mymodule}.pyx"]),],
         compiler_directives=dict(language_level=3),
         compile_time_env=dict(HIP_PYTHON=True),
      )
   )
```