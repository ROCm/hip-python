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

Basic Usage and Examples
========================

This chapter explains how to use HIP Python's main interfaces. The usage of
the CUDA\ |reg| interoperability layer is discussed in a separate chapter. We
first aim to give an introduction to the *Python* API of HIP Python by means
of basic examples before discussing the *Cython* API in the last sections of
this chapter.

.. note::

   All examples in this chapter have been tested with ROCm\ |trade| 7.13 on Ubuntu
   22.04. The :ref:`ch_license` applies to all examples in this chapter.

Basic Usage (Python)
--------------------

.. admonition:: What will I learn?

   * How to use HIP Python modules in your Python code.

After installing the HIP Python package ``hip-python``, you can import the individual
modules that you need as shown below:

.. code-block:: py
   :linenos:
   :caption: Importing HIP Python Modules

   from hip import hip
   from hip import hiprtc
   # ...

And you are ready to go!

.. note::

   The ``from hip import hip`` (and friends) imports continue to
   work — they are served by the ``hip-python`` package, which
   provides the ``hip.*`` namespace as an alias of the
   ``rocm.bindings.*`` modules. New code should prefer
   ``from rocm.bindings import hip`` (or
   ``import rocm.bindings.hip as hip``) directly: the modern style
   is more explicit about which package supplies the symbol and
   matches the per-package layout used throughout the rest of the
   documentation. The CUDA interop layer similarly prefers
   ``from cuda.bindings import driver, runtime, nvrtc``.

   .. code-block:: py
      :linenos:
      :caption: Preferred imports for new code

      from rocm.bindings import hip
      from rocm.bindings import hiprtc
      # or
      import rocm.bindings.hip as hip
      import rocm.bindings.hiprtc as hiprtc

   Both styles are supported and not deprecated; the
   ``rocm.bindings.*`` style is more explicit about which package
   supplies the symbol and matches the modern per-package layout
   used by every example in this guide.

.. _sec_obtaining_device_properties:

Obtaining Device Properties
---------------------------

.. admonition:: What will I learn?

   * How I can obtain device attributes/properties via :py:obj:`~.hipGetDeviceProperties`.
   * How I can obtain device attributes/properties via :py:obj:`~.hipDeviceGetAttribute`.

Obtaining device properties such as the architecture or the number of compute
units is important for many applications.

Via :py:obj:`~.hipGetDeviceProperties`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A number of device properties can be obtained via the
:py:obj:`~.hipDeviceProp_t` object. The example below obtains it as the return
value of the :py:obj:`~.hipGetDeviceProperties` call (line 21); the second
argument (``0``) is the device number.

Running the :ref:`example below <hip_deviceproperties>` will print out the
values of all queried device properties before the program eventually prints
``"ok"`` and quits.

.. note::

   The ``hip_check`` routine in the snippet unpacks the result tuple -- HIP Python
   routines always return a tuple, then checks the therein contained error code
   (first argument), and finally returns the rest of the tuple -- either as
   single value or tuple sans error code. Such error check routines will be used
   throughout this and the following sections.

.. literalinclude:: ../../examples/0_Basic_Usage/hip_deviceproperties.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 21
   :linenos:
   :name: hip_deviceproperties
   :caption: Obtaining Device Properties via hipGetDeviceProperties

Via :py:obj:`~.hipDeviceGetAttribute`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also obtain some of the properties that appeared in the
:ref:`previous example <hip_deviceproperties>` plus a number of additional
properties via the :py:obj:`~.hipDeviceGetAttribute` routine as shown in the
:ref:`example below <hip_deviceattributes>` (line 32). In the example below,
we query integer-type device attributes/properties. Therefore, we supply the
address of a :py:obj:`ctypes.c_int` variable as first argument. The respective
property, the second argument, is passed as enum constant of type
:py:obj:`~.hipDeviceAttribute_t`.

Running this example will print out the values of all queried device
attributes before the program prints ``"ok"`` and quits.

.. literalinclude:: ../../examples/0_Basic_Usage/hip_deviceattributes.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 32
   :linenos:
   :name: hip_deviceattributes
   :caption: Obtaining Device Properties via hipDeviceGetAttribute

.. _sec_hip_streams:

HIP Streams
-----------

.. admonition:: What will I learn?

   * How I can use HIP Python's :py:obj:`~.hipStream_t` objects and the
     associated HIP Python routines.
   * That I can directly pass Python 3 :py:obj:`~array.array` objects to HIP
     runtime routines such as :py:obj:`~.hipMemcpy` and
     :py:obj:`~.hipMemcpyAsync`.

An important concept in HIP are streams. They allow to overlap host and device
work as well as device computations with data movement to or from that same
device.

The :ref:`below example <hip_stream>` showcases how to use HIP Python's
:py:obj:`~.hipStream_t` objects and the associated HIP Python routines. The
example further demonstrates that you can pass Python 3 :py:obj:`array.array`
types directly to HIP Python interfaces that expect an host buffer. One
example of such interfaces is :py:obj:`~.hipMemcpyAsync` (lines 23 and 29).

.. literalinclude:: ../../examples/0_Basic_Usage/hip_stream.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :name: hip_stream
   :caption: HIP Streams

.. admonition:: What is happening?

   1. A host buffer is filled with random numbers (line 18) before
      it is asynchronously copied to the device (line 23), where a asynchronous
      :py:obj:`~.hipMemsetAsync` (same stream) resets all bytes to ``0`` (line 28).
   2. An asynchronous memcpy (same stream) is then issued to copy the device data
      back to the host (line 29). All operations within the stream are executed in
      order.
   3. As the ``~Async`` operations are non-blocking, the host waits via
      :py:obj:`~.hipStreamSynchronize` until operations in the stream have been
      completed (line 34) before destroying the stream (line 35).
   4. Eventually the program deallocates device data via :py:obj:`~.hipFree` and
      checks if all bytes in the host buffer are now set to ``0``.
      If so, it quits with an "ok".

.. _sec_launching_kernels:

Launching Kernels
-----------------

.. admonition:: What will I learn?

   * How I can compile a HIP C++ kernel at runtime via :py:obj:`~.hiprtcCompileProgram`.
   * How I can launch kernels via :py:obj:`~.hipModuleLaunchKernel`.

HIP Python does not provide the necessary infrastructure to express device code
in native Python. However, you can compile and launch kernels from within
Python code via the just-in-time (JIT) compilation interface provided by HIP
Python module :py:obj:`~.rocm.bindings.hiprtc` together with the kernel launch routines
provided by HIP Python module :py:obj:`~.rocm.bindings.hip`. The
:ref:`example below <hiprtc_launch_kernel_no_args>` demonstrates how to do so.

.. literalinclude:: ../../examples/0_Basic_Usage/hiprtc_launch_kernel_no_args.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 27-30, 32, 39-40, 46-50, 52-62
   :name: hiprtc_launch_kernel_no_args
   :caption: Compiling and Launching Kernels

.. admonition:: What is happening?

   1. In the example, the kernel ``print_tid`` defined within the string ``source``
      simply prints the block-local thread ID (``threadIDx.x``) for every thread
      running the kernel (lines 27-30).
   2. A program ``prog`` is then created in line 32 via
      :py:obj:`~.hiprtcCreateProgram`, where we pass ``source`` as first argument,
      we further give the program a name (note the ``b".."``), specify zero headers
      and include names (last three arguments).
   3. Next we query the architecture name via :py:obj:`~.hipGetDeviceProperties`
      (more details: :ref:`sec_obtaining_device_properties`) and use it in
      lines 39-40, where we specify compile flags (``cflags``) and compile ``prog``
      via :py:obj:`~.hiprtcCompileProgram`.
      In case of a failure, we obtain the program log and raise it as
      :py:obj:`~.RuntimeError`.
   4. In case of success, we query the code size via
      :py:obj:`~.hiprtcGetCodeSize`, create a buffer with that information, and
      then copy the code into this buffer via :py:obj:`~.hiprtcGetCode`.
      Afterwards, we load the code as ``module`` via :py:obj:`~.hipModuleLoadData`
      and then obtain our device kernel with name ``"print_tid"`` from it via
      :py:obj:`~.hipModuleGetFunction`.
   5. This object is then passed as first argument to the
      :py:obj:`~.hipModuleLaunchKernel` routine, followed by the usual grid and
      block dimension triples, the size of the required shared memory, and stream
      to use (``None`` means the null stream). The latter two arguments,
      ``kernelParams`` and ``extra``, can be used for passing kernel arguments.
      We will take a look how to pass kernel arguments via ``extra`` in the next
      section.
   6. After the kernel launch, the host waits on completion via
      :py:obj:`~.hipDeviceSynchronize` and then unloads the code module again
      via  :py:obj:`~.hipModuleUnload` before quitting with an ``"ok"``.

.. _sec_launching_kernels_with_args:

Kernels with Arguments
----------------------

.. admonition:: What will I learn?

   How I can pass arguments to :py:obj:`~.hipModuleLaunchKernel`.

One of the difficulties that programmers face when attempting to launch
kernels via :py:obj:`~.hipModuleLaunchKernel` is passing arguments to the
kernels. When using the ``extra`` argument, the kernel arguments must be aligned
in a certain way. In C/C++ programs, one can simply put all arguments into a
struct and let the compiler take care of the argument alignment. Similarly,
one could create a :py:obj:`ctypes.Structure` in python to do the same.

However, we do not want to oblige HIP Python users with creating such glue
code. Instead, users can directly pass a :py:obj:`list` or :py:obj:`tuple` of
arguments to the :py:obj:`~.hipModuleLaunchKernel`. The entries of these
objects must either be of type :py:obj:`~.DeviceArray` (or can be converted to
:py:obj:`~.DeviceArray`) or one of the :py:obj:`ctypes` types.

The former are typically the result of a :py:obj:`~.hipMalloc` call (or
similar memory allocation routines). Please also see :ref:`ch_datatypes`
for details on what other types can be converted to :py:obj:`~.DeviceArray`.
The :py:obj:`ctypes` types are typically used to convert a scalar of the
python :py:obj:`bool`, :py:obj:`int`, and :py:obj:`float` scalar types to a
fixed precision.

The :ref:`below example <hiprtc_launch_kernel_args>` demonstrates the usage of
:py:obj:`~.hipModuleLaunchKernel` by means of a simple kernel, which scales a
vector by a factor. Here, we pass multiple arguments that require different
alignments to the aforementioned routine in lines 95-102. We insert some
additional ``unused*`` arguments into the ``extra`` :py:obj:`tuple` to stress the
argument buffer allocator. Note the :py:obj:`ctypes` object construction for
scalars and the direct passing of the device array ``x_d``. Compare the argument
list with the signature of the kernel defined in line 31. The example also
introduces HIP Python's :py:obj:`~.dim3` struct (default value per dimension
is 1), which can be unpacked just like a :py:obj:`tuple` or :py:obj:`list`.

.. literalinclude:: ../../examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 31, 83-84, 95-102
   :linenos:
   :name: hiprtc_launch_kernel_args
   :caption: Compiling and Launching Kernels With Arguments

.. admonition:: What is happening?

   See the previous section :ref:`sec_launching_kernels` for
   a textual description of the main steps.

hipBLAS and NumPy Interoperability
----------------------------------

.. admonition:: What will I learn?

   * How I can use HIP Python's :py:obj:`~.hipblas` module.
   * That I can pass :py:obj:`numpy` arrays to HIP runtime routines such
     as :py:obj:`~.hipMemcpy` and :py:obj:`~.hipMemcpyAsync`.

:ref:`This example <hipblas_with_numpy>` demonstrates how to initialize and use HIP
Python's :py:obj:`~.hipblas` module. Furthermore, it shows that you can simply
pass :py:obj:`numpy` arrays to HIP runtime routines such as
:py:obj:`~.hipMemcpy` and :py:obj:`~.hipMemcpyAsync`. This works because some
of HIP Python's interfaces support automatic conversion from various different
types --- in particular such types that implement the
`Python buffer protocol <https://docs.python.org/3/c-api/buffer.html>`__.
The :py:obj:`~numpy.numpy` arrays implement the Python buffer protocol and
thus can be directly passed to those interfaces.

.. literalinclude:: ../../examples/0_Basic_Usage/hipblas_with_numpy.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 34-35, 46-51, 54-60, 63-65
   :linenos:
   :name: hipblas_with_numpy
   :caption: hipBLAS and NumPy Interoperability

.. admonition:: What is happening?

   1. We initialize two ``float32``-typed :py:obj:`numpy` arrays ``x_h`` and ``y_h`` on
      the host and fill them with random data (lines 34-35).
   2. We compute the expected result on the host via :py:obj:`numpy` array
      operations (line 38).
   3. We allocate device analogues for ``x_h`` and ``y_h`` (lines 42-43) and copy the
      host data over (lines 46-51). Note that we can directly pass the
      :py:obj:`numpy` arrays ``x_h`` and ``y_h`` to :py:obj:`~.hipMemcpy`.
   4. Before being able to call one of the compute routines of
      :py:obj:`~.hipblas`, it's necessary to create a :py:obj:`~.hipblas` handle
      via :py:obj:`~.hipblasCreate` that will be passed to every
      :py:obj:`~.hipblas` routine as first argument (line 54).
   5. In lines 55-59 follows the call to :py:obj:`~.hipblasSaxpy`, where we pass the
      handle as first argument and the address of host :py:obj:`ctypes.c_float`
      variable ``alpha`` as third argument.
   6. In line 60 the handle is destroyed via :py:obj:`~.hipblasDestroy` because
      it is not needed anymore.
   7. The device data is downloaded in lines 63-65, where we pass ``numpy`` array
      ``y_h`` as destination array.
   8. We compare the expected host result with the downloaded device result
      (lines 68-71) and print ``"ok"`` if all is fine.

Linear Algebra with hipSOLVER
-----------------------------

.. admonition:: What will I learn?

   * How I can create a :py:obj:`~.hipsolver` handle via
     :py:obj:`~.hipsolverCreate`.
   * How I can query the workspace size and compute an LU factorization on
     the GPU via :py:obj:`~.hipsolverDgetrf_bufferSize` and
     :py:obj:`~.hipsolverDgetrf`.

:ref:`This example <hipsolver_getrf>` computes the LU factorization of a
small dense matrix on the GPU using HIP Python's :py:obj:`~.hipsolver`
module. Like LAPACK, hipSOLVER expects **column-major** matrices, so the
input is a Fortran-ordered :py:obj:`numpy` array that we pass directly to
:py:obj:`~.hipMemcpy`. The factorization computes :math:`PA = LU`; the
example reconstructs :math:`LU` on the host with :py:obj:`numpy` and checks
it against the row-pivoted input, printing ``"ok"`` on success.

Note that :py:obj:`~.hipsolverDgetrf_bufferSize` returns the required
workspace size (``lwork``, in bytes) *directly* as a second return value
next to the status --- HIP Python models such callee-written scalar output
pointers as return values rather than as caller-supplied buffers.

.. literalinclude:: ../../examples/0_Basic_Usage/hipsolver_getrf.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 26-33, 37-39, 42-44, 48-50, 53-57, 60-75, 80-91
   :linenos:
   :name: hipsolver_getrf
   :caption: Linear Algebra with hipSOLVER

.. admonition:: What is happening?

   1. We build the input matrix ``A`` as a Fortran-ordered (column-major)
      ``float64`` :py:obj:`numpy` array and keep a copy ``A_orig`` for the
      later check (lines 26-33).
   2. We allocate device memory for the factored matrix ``dA``, the pivot
      indices ``dIpiv`` and the info flag ``dInfo`` (lines 37-39), then copy
      the input matrix over --- passing the :py:obj:`numpy` array directly to
      :py:obj:`~.hipMemcpy` (lines 42-44).
   3. We create a :py:obj:`~.hipsolver` handle via
      :py:obj:`~.hipsolverCreate` (line 48) and query the workspace size via
      :py:obj:`~.hipsolverDgetrf_bufferSize` (line 49). The size ``lwork`` is
      *returned* by the call, so we allocate the workspace ``dWork`` from it
      (line 50).
   4. In lines 53-57 we compute the LU factorization in place with
      :py:obj:`~.hipsolverDgetrf`, passing the handle, the matrix, the
      workspace and its size, and the pivot / info outputs.
   5. We download the factored matrix, the pivots and the info flag back to
      the host (lines 60-75).
   6. We rebuild the unit-lower ``L`` and upper ``U`` factors, replay the
      LAPACK 1-based row pivots on a copy of the original matrix to form
      ``PA``, and print ``"ok"`` if :math:`PA = LU` holds (lines 80-91).
   7. Finally we free the device buffers and destroy the handle via
      :py:obj:`~.hipsolverDestroy`.

.. _sec_example_hip_python_device_arrays:

HIP Python Device Arrays
------------------------

.. admonition:: What will I learn?

   * How I can change the shape and datatype of HIP Python's
     :py:obj:`~.DeviceArray` objects.
   * How I can obtain subarrays from HIP Python's :py:obj:`~.DeviceArray`
     objects --- which are again of that type --- via array subscript.

:ref:`This example <hip_python_device_array>` demonstrates how to
:py:obj:`~.DeviceArray.configure` the shape and data typ of a
:py:obj:`~.DeviceArray` returned by :py:obj:`~.hipMalloc` (and related
routines). It further shows how to retrieve single elements / contiguous
subarrays with respect to the specified type and shape information.

.. literalinclude:: ../../examples/0_Basic_Usage/hip_python_device_array.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 32-33, 37-39, 46-52, 57-59
   :name: hip_python_device_array
   :caption: Configuring and Slicing HIP Python's DeviceArray

.. admonition:: What is happening?

   1. A two-dimensional row-major array of size ``(3,20)`` is created on the host.
      All elements are initialized to ``1`` (lines 32-33).
   2. A device array with the same number of bytes is created on the device, then
      reconfigured to have ``float32`` type and the shape of the host array via
      :py:obj:`~.DeviceArray.configure` (lines 37-39).
   3. The host data is copied to the device array (lines 40-42).
   4. Within a loop over the row indices (index: ``r``):

      1. A pointer to row with index ``r`` is created via array subscript (line
         47). This yields ``row``.
      2. ``row`` is passed to a :py:obj:`~.hipblasSscal` call that writes index ``r``
         to all elements of the row (lines 50-52).
   5. Data is copied back from the device to the host array (lines 57-59).
   6. The device data is deallocated via :py:obj:`~.hipFree` (line 62).
   7. Finally, a check is performed on the host if the row values equal the
      respective row index (lines 64-68). The program quits with ``"ok"`` if all
      went well.

.. note::

   Please also see :ref:`ch_datatypes` for more details on the capabilities
   of type :py:obj:`~.DeviceArray` and the
   `CUDA Array interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`__
   that it implements.

Monte Carlo with hipRAND
------------------------

.. admonition:: What will I learn?

   * How I can create an :py:obj:`~.hiprand` random number generator via :py:obj:`~.hiprandCreateGenerator`.
   * How I can generate uniformly-distributed random numbers via :py:obj:`~.hiprandGenerateUniformDouble`.

:ref:`This example <hiprand_monte_carlo_pi>` uses :py:obj:`~.hiprand` to estimate
:math:`\pi` by means of the Monte-Carlo method.

.. admonition:: Background

   The unit square has the area :math:`1^2`, while the unit circle has the area
   :math:`\pi\,(\frac{1}{2})^2`. Therefore, the ratio between the latter and the
   former area is :math:`\frac{\pi}{4}`. Using the Monte-Carlo method, we
   randomly choose :math:`N` :math:`(x,y)`-coordinates in the unit square.
   We then estimate the ratio of areas as the ratio between the number of samples
   located within the unit circle and the total number of samples :math:`N`. The
   accuracy of the approach increases with :math:`N`.

.. note::

   This example was derived from a similar example in the
   `rocRAND repository on Github <https://github.com/ROCm/rocRAND/tree/develop>`__.
   See this repository for another higher-level interface to hipran and rocrand
   (:py:obj:`ctypes`-based, no Cython interfaces).

.. literalinclude:: ../../examples/0_Basic_Usage/hiprand_monte_carlo_pi.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 32-36, 40-42, 52
   :linenos:
   :name: hiprand_monte_carlo_pi
   :caption: Monte Carlo with hipRAND

.. admonition:: What is happening?

   Within a loop that per iteration multiplies the problem size ``n`` by ``10``
   (lines 64-65), we call a function ``calculate_pi`` with  ``n`` as argument, in which:

   1. We first create a two-dimensional host array ``xy`` of type ``double`` with ``n``
      elements (line 31).
   2. We then create a :py:obj:`~.hiprandCreateGenerator` generator of type
      :py:obj:`~.hiprandRngType.HIPRAND_RNG_PSEUDO_DEFAULT` (lines 32-36).
   3. We create a device array ``xy_d`` that stores the same number of bytes as
      ``xy`` (lines 37-39).
   4. We fill ``xy_d`` with random data via :py:obj:`~.hiprandGenerateUniformDouble`
      (lines 40-42).
   5. We then copy to ``xy`` from ``xy_d`` and free ``xy_d`` (lines 43-51) and destroy
      the generator (line 52).
   6. We use ``numpy`` array operations to count the number of random-generated
      :math:`x-y`-coordinates within the unit circle (lines 54-55).
   7. Finally, we compute the ratio estimate for the given ``n`` and return it
      (lines 56-57).

A simple complex FFT with hipFFT
--------------------------------

.. admonition:: What will I learn?

   * How I can create an :py:obj:`~.hipfft` 1D plan via :py:obj:`~.hipfftPlan1d`.
   * How I can run a complex in-place forward FFT via :py:obj:`~.hipfftExecZ2Z`.

:ref:`This example <hipfft_py>` demonstrates the usage of HIP Python's
:py:obj:`~.hipfft` library.

We perform a double-complex-to-double-complex in-place forward FFT of a
constant time signal :math:`f(t) = 1-1j` of which we have :math:`N` samples.
The resulting FFT coefficients are all zero --- aside from the first one,
which has the value :math:`N-Nj`.

.. literalinclude:: ../../examples/0_Basic_Usage/hipfft.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 39, 42-46
   :name: hipfft_py
   :caption: A simple complex FFT with hipFFT

.. admonition:: What is happening?

   1. We start with creating the initial data in lines 28-30, where we use
      :py:obj:`~numpy.numpy` for convenience.
   2. We then create a device array of the same size and copy the device data
      over (lines 33-36).
   3. We create a plan in line 39, where we specify the number of samples ``N`` and
      the the type of the FFT as *double-complex-to-double-complex*, :py:obj:`~.hipfftType.HIPFFT_Z2Z`.
   4. Afterwards, we execute the FFT in-place (``idata=dx`` and ``odata=dx``) and
      specify that we run an forward FFT, :py:obj:`~.HIPFFT_FORWARD` (lines 42-46).
   5. The host then waits for completion of all activity on the device before
      copying data back to the host and freeing the device array (lines 47-53).
   6. Finally, we check if the result is as expected and print ``"ok"`` if that's
      the case (lines 55 onward).

A multi-GPU broadcast with RCCL
-------------------------------

.. admonition:: What will I learn?

   * How I can create a multi-GPU communicator via :py:obj:`~.ncclCommInitAll`.
   * How I can destroy a communicator again via :py:obj:`~.ncclCommDestroy`.
   * How I can open and close a communication group via
     :py:obj:`~.ncclGroupStart` and :py:obj:`~.ncclGroupEnd`, respectively.
   * How I can perform a broadcast via :py:obj:`~.ncclBcast`.

:ref:`This example <rccl_comminitall_bcast>` implements a single-node multi-GPU
broadcast of a small array from one GPU's device buffer to that of the other
ones.

.. literalinclude:: ../../examples/0_Basic_Usage/rccl_comminitall_bcast.py
   :language: python
   :start-after: [literalinclude-begin]
   :linenos:
   :emphasize-lines: 28-33, 50, 53-62, 64, 81-83
   :name: rccl_comminitall_bcast
   :caption: A multi-GPU broadcast with RCCL

.. admonition:: What is happening?

   1. In line 28, we use the device count ``num_gpus`` (via
      :py:obj:`~.hipGetDeviceCount`) to create an array of pointers (same size as
      ``unsigned long``, ``dtype="uint64"``). This array named ``comms`` is intended to
      store a pointer to each device's communicator.
   2. We then create an array of device identifiers (line 32).
   3. We pass both arrays to :py:obj:`~.ncclCommInitAll` as first and last
      argument, respectively (line 33). The second element is the device count.
      The aforementioned routine initializes all communicators and writes their
      address to the ``comms`` array.
   4. In lines 37-47, we create an array ``dx`` on each device of size ``N`` that is
      initialized with zeros on all devices except device ``0``. The latter's array
      is filled with ones.
   5. We start a communication group in line 50, and then call
      :py:obj:`~.ncclBcast` per device in lines 53-62. The first argument of the call
      is per-device ``dx``, the second the size of ``dx``. Then follows the
      :py:obj:`~.ncclDataType_t`, the root (device ``0``), then the communicator
      (``int(comms[dev])``) and finally the stream (:py:obj:`None`). Casting
      ``comms[dev]`` to :py:obj:`int` is required as the result is otherwise
      interpreted as single-element ``Py_buffer`` by HIP Python's
      :py:obj:`~.ncclBcast` instead of as an address.
   6. In line 64, we close the communication group again.
   7. We download all data to the host per device and check if the elements are
      set to ``1`` (lines 67-76). Otherwise, a runtime error is emitted.
   8. Finally, we clean up by deallocating all device memory (lines 78-79) and
      destroying the per-device communicators via :py:obj:`~.ncclCommDestroy`
      (lines 81-83). Note that here again the ``comm`` must be converted to
      ``int`` before passing it to the HIP Python routine.

.. note::

   Please also see :ref:`ch_datatypes` for more details on automatic type
   conversions supported by HIP Python's datatypes.

.. _sec_hipfile_copy:

Copying a File through GPU Memory with hipFile
----------------------------------------------

.. admonition:: What will I learn?

   * How I can use the high-level :py:obj:`rocm.hipfile` wrapper classes
     ``Driver``, ``Buffer``, and ``FileHandle`` as context managers.
   * How I can read a file directly into device memory and write it back
     out via ``FileHandle.read`` / ``FileHandle.write``.

hipFile (Accelerated I/O Storage) moves data directly between storage and
GPU memory. HIP Python ships two layers for it: the auto-generated
low-level :py:obj:`rocm.bindings.hipfile` bindings and the high-level,
Pythonic :py:obj:`rocm.hipfile` wrapper. The :ref:`example below
<hipfile_copy>` uses the high-level wrapper to copy a file through a device
buffer and then verifies the round-trip by comparing SHA256 hashes.

.. note::

   hipFile requires the ``libhipfile.so`` shared library, which may not be
   part of a standard ROCm\ |trade| installation (see the note in
   :doc:`the installation chapter <0_install>`). It also issues its
   transfers with ``O_DIRECT``, so the scratch files must live on an
   ``O_DIRECT``-capable filesystem; set the ``HIPFILE_TMPDIR`` environment
   variable to such a mount (e.g. an ext4 mount) if the default temp
   directory is a ``tmpfs``.

.. literalinclude:: ../../examples/0_Basic_Usage/hipfile_copy.py
   :language: python
   :start-after: [literalinclude-begin]
   :emphasize-lines: 16, 41, 46, 48, 49-57, 59, 61
   :linenos:
   :name: hipfile_copy
   :caption: Copying a File through GPU Memory with hipFile

.. admonition:: What is happening?

   1. We query the hipFile version via ``get_version`` (line 16) and create
      a 2 MiB random input file in a temporary directory (line 34). The size
      is block-aligned so the ``O_DIRECT`` transfers are valid.
   2. We allocate a device buffer via :py:obj:`~.hipMalloc` and take its
      address from the returned :py:obj:`~.DeviceArray` (line 41).
   3. We open the hipFile ``Driver`` (line 46) and register the device
      buffer as a ``Buffer`` (line 48), both as context managers so they are
      deregistered/closed automatically on scope exit.
   4. We register the input and output files as ``FileHandle`` context
      managers (lines 49-57). ``FileHandleType.OPAQUE_FD`` selects the POSIX
      file-descriptor handle type (the default).
   5. We read the input file into the device buffer via
      ``FileHandle.read`` (line 59) and write it back out to the output file
      via ``FileHandle.write`` (line 61).
   6. After the ``with`` blocks tear down the handles, buffer, and driver,
      we free the device memory via :py:obj:`~.hipFree` and compare the
      SHA256 hashes of the input and output files.

.. note::

   For a version that drives the auto-generated :py:obj:`rocm.bindings.hipfile`
   functions directly — showing the raw ``(retval, errno, hip_drv_err)``
   result tuples and the manual driver/buffer/handle lifecycle that the
   high-level classes encapsulate — see
   ``examples/0_Basic_Usage/hipfile_copy_lowlevel.py``.

Basic Usage (Cython)
--------------------

.. admonition:: What will I learn?

   * How I can use HIP Python's Cython modules in my Cython code.
   * How to compile my Cython code that uses HIP Python's Cython modules.

In this section, we show how to use HIP Python's `Cython <https://cython.org/>`__
modules and how to compile projects that use them.

Cython Recap
^^^^^^^^^^^^

.. note::

   This section expects that the user has at least some basic knowledge about
   the programming language Cython.
   If you are unfamiliar with the language, we refer to the
   `Cython tutorials <https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html>`__
   and the `Language Basics page <https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html>`__.

Cython modules are often split into a ``*.pxd`` and a ``*.pyx`` file, which are a
Cython module's declaration and implementation part respectively. While the
former files are to some degree comparable to header files in C/C++, the
latter can be compared to sources files. The declaration part may only contain
``cdef`` fields, variables, and function prototypes while the implementation
part may contain the implementation of those entities as well as Python
fields, variables, and functions.

The implementation part is the interface between the C/C++ and the Python
world. Here, you can import Python code via Python's ``import`` statements,
you can *C-import* ``cdef`` declarations from other Cython declaration files
(``*.pxd``) via ``cimport`` statements, and you can include C/C++ declarations
from C/C++ header files as ``cdef`` declarations.

To build a Python module from a Cython module, the implementation part must be
first "cythonized", i.e. converted into a C/C++ file and then compiled with a
compiler. It is recommended to use the compiler that was used for compiling
the used python interpreter. Most people don't do this manually but instead
prefer to use the build infrastructure provided by :py:obj:`setuptools`. They
then write a ``setup.py`` script that contains the code that performs the
aforementioned two tasks.

Cython modules in HIP Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per Python module :py:obj:`rocm.bindings.hip`,
:py:obj:`rocm.bindings.hiprtc`, ..., HIP Python ships an additional
``cy``-prefixed ``rocm.bindings.cy<pkg_name>`` module.

* The module *without* the ``c`` prefix is compiled into the interface for HIP
  Python's Python users. However, all ``cdef`` declarations therein can also be
  ``cimport``\ ed by Cython users (typically ``cdef class`` declarations) and all
  Python objects therein can be ``import``\ ed by Cython users too (typically enum
  and function objects).
* The module *with* the ``cy`` prefix builds the bridge to the underlying HIP C
  library by including C definitions from the corresponding header files. This
  code is located in the declaration part. This part further declares runtime
  function loader prototypes. The definition of these function loaders in the
  implementation part first try to load the underlying C library and then if
  successful, try to load the function symbol from that shared object.

.. note::

   The lazy-loading of functions at runtime can, under some circumstances, allow
   to use a HIP Python version that covers a superset or only a subset of the
   functions available within the respective library of a ROCm\ |trade|
   installation.

Using the Cython API
^^^^^^^^^^^^^^^^^^^^

You can import the Python objects that you need as shown below:

.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Modules into Cython ``*.pyx`` file

   from rocm.bindings import hip # enum types, enum aliases, fields
   from rocm.bindings import hiprtc
   # ...

In the same file, you can **also or alternatively** ``cimport`` the ``cdef``
entities as shown below:

.. code-block:: cython
   :linenos:
   :caption: Importing HIP Python Cython declaration files (``*.pxd``) into a Cython ``*.pxd`` or ``*.pyx`` file

   from rocm.bindings cimport cyhip   # direct access to C interfaces and lazy function loaders
   from rocm.bindings cimport cyhiprtc
   # ...

   from rocm.bindings cimport hip # access to `cdef class` and `ctypedef` types
                                  # that have been created per C struct/union/typedef
   from rocm.bindings cimport hiprtc
   # ...

Compiling a Cython module
^^^^^^^^^^^^^^^^^^^^^^^^^

After having written your own ``mymodule.pyx`` file that uses HIP Python's
Cython API, you can compile the result using a ``setup.py`` script as
:ref:`shown below <cython_setup_py>`. In the ``setup.py`` script, we only assume that
HIP or HIPRTC is used. Therefore, only ``amdhip64`` is put into the ``rocm_libs``
list. It is further important to specify the HIP Platform as the header files
from which we include the C interfaces will be included at compile time by the
underlying C/C++ compiler. The compilation path must include all these
interfaces.

.. code-block:: python
   :linenos:
   :caption: Compiling a Cython module that uses HIP Python's Cython API.
   :name: cython_setup_py

   import os, sys

   mymodule = "mymodule"

   # We only assume HIP/HIPRTC is used, so only `amdhip64` is linked.
   rocm_libs = ["amdhip64"]

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

.. |reg| unicode:: U+000AE .. REGISTERED SIGN
   :ltrim:
.. |trade| unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
