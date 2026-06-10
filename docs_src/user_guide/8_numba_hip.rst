.. MIT License
..
.. Copyright (c) 2026 Advanced Micro Devices, Inc.
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

.. _numba_hip:

Numba HIP
=========

`Numba <https://numba.pydata.org/>`__ is a just-in-time (JIT) compiler that
turns a subset of Python and NumPy into fast machine code. **Numba HIP** is the
ROCm\ |trade| HIP backend for Numba: it lets you write GPU kernels in Python and
run them on AMD GPUs.

Numba HIP intentionally mirrors the ``numba.cuda`` API, so there are two
equivalent ways to use it.

**Natively, through the** ``numba.hip`` **package (recommended for new code).**
Every name lives under ``numba.hip`` instead of ``numba.cuda``; the only
difference is the module prefix:

.. code-block:: python

   from numba import hip

   @hip.jit
   def f(a, b, c):
       tid = hip.grid(1)
       size = len(c)
       if tid < size:
           c[tid] = a[tid] + b[tid]

**By posing as** ``numba.cuda`` **(mainly for porting).** A single call re-binds
the ``numba.cuda`` module onto the HIP backend, so existing Numba CUDA kernels
run unchanged. This is intended primarily for porting existing Numba CUDA code:

.. code-block:: python

   from numba import hip

   hip.pose_as_cuda()

   from numba import cuda  # now backed by HIP

.. note::

   The examples in this chapter are the hipified versions of the upstream Numba
   CUDA documentation examples. They are exercised as part of the Numba HIP test
   suite under ``tests/numba-hip/doc_examples/`` and run on AMD GPUs via the
   ``hip.pose_as_cuda()`` shim shown above. Each example below begins with the
   import + ``pose_as_cuda()`` preamble (omitted from the snippets for brevity).

   **All examples can equally be run natively through the** ``numba.hip``
   **package.** Drop the ``pose_as_cuda()`` call, use ``from numba import hip``,
   and replace the ``cuda.`` prefix with ``hip.`` throughout (e.g. ``@hip.jit``,
   ``hip.grid(1)``, ``hip.to_device``, ``hip.device_array_like``,
   ``hip.shared.array``, ``hip.syncthreads()``, ``hip.declare_device``).
   ``hip.pose_as_cuda()`` is mainly intended for porting existing Numba CUDA
   code.

Vector addition
---------------

A minimal element-wise kernel. ``cuda.grid(1)`` returns the global thread index;
each thread adds one pair of elements:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_vecadd.py
   :language: python
   :start-after: # ex_vecadd.kernel.begin
   :end-before: # ex_vecadd.kernel.end
   :dedent:

Allocate device arrays with ``cuda.to_device`` / ``cuda.device_array_like``:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_vecadd.py
   :language: python
   :start-after: # ex_vecadd.allocate.begin
   :end-before: # ex_vecadd.allocate.end
   :dedent:

The simplest launch lets ``forall`` pick the grid configuration for you:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_vecadd.py
   :language: python
   :start-after: # ex_vecadd.forall.begin
   :end-before: # ex_vecadd.forall.end
   :dedent:

Or specify the block/grid dimensions explicitly:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_vecadd.py
   :language: python
   :start-after: # ex_vecadd.launch.begin
   :end-before: # ex_vecadd.launch.end
   :dedent:

Reduction
---------

A shared-memory tree reduction. ``cuda.shared.array`` allocates per-block shared
memory and ``cuda.syncthreads()`` synchronizes the threads in a block:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_reduction.py
   :language: python
   :start-after: # ex_reduction.kernel.begin
   :end-before: # ex_reduction.kernel.end
   :dedent:

Launch it over a single block covering the whole input array:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_reduction.py
   :language: python
   :start-after: # ex_reduction.launch.begin
   :end-before: # ex_reduction.launch.end
   :dedent:

Matrix multiplication
---------------------

A naive 2-D kernel where each thread computes one output element via
``cuda.grid(2)``:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_matmul.py
   :language: python
   :start-after: # magictoken.ex_matmul.begin
   :end-before: # magictoken.ex_matmul.end
   :dedent:

Run it with a 2-D block/grid:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_matmul.py
   :language: python
   :start-after: # magictoken.ex_run_matmul.begin
   :end-before: # magictoken.ex_run_matmul.end
   :dedent:

A faster, shared-memory tiled variant blocks the dot product into
``TPB``-by-``TPB`` tiles:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_matmul.py
   :language: python
   :start-after: # magictoken.ex_fast_matmul.begin
   :end-before: # magictoken.ex_fast_matmul.end
   :dedent:

Universal functions (ufuncs)
----------------------------

NumPy ufuncs such as ``np.sin`` can be called directly inside a kernel:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_ufunc.py
   :language: python
   :start-after: # ex_cuda_ufunc.begin
   :end-before: # ex_cuda_ufunc.end
   :dedent:

CPU/GPU compatibility
---------------------

A function decorated with ``@numba.jit`` can be reused unchanged inside a
``@cuda.jit`` kernel, so the same business logic runs on both CPU and GPU.

Allocate the data on the device:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_cpu_gpu_compat.py
   :language: python
   :start-after: # ex_cpu_gpu_compat.allocate.begin
   :end-before: # ex_cpu_gpu_compat.allocate.end
   :dedent:

Define plain CPU logic with ``@numba.jit``:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_cpu_gpu_compat.py
   :language: python
   :start-after: # ex_cpu_gpu_compat.define.begin
   :end-before: # ex_cpu_gpu_compat.define.end
   :dedent:

The same function can be called on the host:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_cpu_gpu_compat.py
   :language: python
   :start-after: # ex_cpu_gpu_compat.cpurun.begin
   :end-before: # ex_cpu_gpu_compat.cpurun.end
   :dedent:

...and reused verbatim from a device kernel:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_cpu_gpu_compat.py
   :language: python
   :start-after: # ex_cpu_gpu_compat.usegpu.begin
   :end-before: # ex_cpu_gpu_compat.usegpu.end
   :dedent:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_cpu_gpu_compat.py
   :language: python
   :start-after: # ex_cpu_gpu_compat.launch.begin
   :end-before: # ex_cpu_gpu_compat.launch.end
   :dedent:

Foreign Function Interface (FFI)
--------------------------------

Numba HIP can link external device functions written in HIP/CUDA C++ into a
kernel. Declare the foreign function with ``cuda.declare_device`` and link the
source file via the ``link=[...]`` argument of ``@cuda.jit``.

The C++ device function (note the ``extern "C"`` linkage and the
caller-provided return slot):

.. literalinclude:: ../../tests/numba-hip/doc_examples/ffi/functions.cu
   :language: c++
   :start-after: // magictoken.ex_mul_f32_f32.begin
   :end-before: // magictoken.ex_mul_f32_f32.end

Declaring and linking it from Python:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_ffi.py
   :language: python
   :start-after: # magictoken.ex_linking_cu.begin
   :end-before: # magictoken.ex_linking_cu.end
   :dedent:

Arrays can be passed to a foreign function as raw pointers using
``cffi``'s ``ffi.from_buffer``. The matching C++ prototype:

.. literalinclude:: ../../tests/numba-hip/doc_examples/ffi/functions.cu
   :language: c++
   :start-after: // magictoken.ex_sum_reduce_proto.begin
   :end-before: // magictoken.ex_sum_reduce_proto.end

Declare the device function with a ``CPointer`` signature:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_ffi.py
   :language: python
   :start-after: # magictoken.ex_from_buffer_decl.begin
   :end-before: # magictoken.ex_from_buffer_decl.end
   :dedent:

...and pass the array buffer pointer from the kernel:

.. literalinclude:: ../../tests/numba-hip/doc_examples/test_ffi.py
   :language: python
   :start-after: # magictoken.ex_from_buffer_kernel.begin
   :end-before: # magictoken.ex_from_buffer_kernel.end
   :dedent:

.. |trade| unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
