*********
Numba HIP
*********

This repository provides a HIP backend for Numba that can be patched into
an existing Numba installation.

About Numba: A Just-In-Time Compiler for Numerical Functions in Python
######################################################################

Numba is an open source, NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc.  It uses the LLVM compiler project to generate machine code
from Python syntax.

Numba can compile a large subset of numerically-focused Python, including many
NumPy functions.  Additionally, Numba has support for automatic
parallelization of loops, generation of GPU-accelerated code, and creation of
ufuncs and C callbacks.

For more information about Numba, see the Numba homepage:
https://numba.pydata.org and the online documentation:
https://numba.readthedocs.io/en/stable/index.html

Numba HIP: Basic Usage
======================

Numba HIP's programming interfaces follow Numba CUDA's design very closely.
Aside from the module name ``hip``, there is often no difference between
Numba CUDA to Numba HIP code.

**Example 1 (Numba HIP):**

.. code-block:: python

   from numba import hip

   @hip.jit
   def f(a, b, c):
      # like threadIdx.x + (blockIdx.x * blockDim.x)
      tid = hip.grid(1)
      size = len(c)

      if tid < size:
          c[tid] = a[tid] + b[tid]


**Example 2 (Numba CUDA):**

.. code-block:: python

   from numba import cuda

   @cuda.jit
   def f(a, b, c):
      # like threadIdx.x + (blockIdx.x * blockDim.x)
      tid = cuda.grid(1)
      size = len(c)

      if tid < size:
          c[tid] = a[tid] + b[tid]

Numba HIP: Posing As Numba CUDA
===============================

As Numba HIP allows to use syntax that is so similar to that of Numba CUDA and
there are already many projects that use Numba CUDA, we have introduced a
feature to the Numba HIP backend that allows it to pose as the Numba CUDA
backend to dependent applications. We demonstrate the usage of this feature in
the example below:

**Example 3 (Numba HIP posing as Numba CUDA):**

.. code-block:: python

   from numba import hip
   hip.pose_as_cuda() # now 'from numba import cuda'
                      # and `numba.cuda` delegate to Numba HIP.

   # unchanged Numba CUDA snippet (Example 2)

   from numba import cuda

   @cuda.jit
   def f(a, b, c):
      # like threadIdx.x + (blockIdx.x * blockDim.x)
      tid = cuda.grid(1)
      size = len(c)

      if tid < size:
          c[tid] = a[tid] + b[tid]


Numba HIP: Limitations
======================

Generally, we aim for feature parity with Numba CUDA.

As of May 2024, the following Numba CUDA features are missing in
Numba HIP:

* Cooperative groups support (ex: ``cg.this_grid()``,
  ``cg.this_grid().sync()``)
* Atomic operations for tuple and array types,
* Runtime kernel debugging functionality,
* Device code printf,
* HIP Simulator equivalent to CUDA Simulator (low priority, users can
  potentially reuse CUDA simulator),
* Half precision (fp16) operations.

Note that so far only limited effort has been spent on optimizing the
performance of the just-in-time compilation infrastructure.

Numba HIP: Design Differences vs. Numba CUDA
============================================

* While Numba CUDA utilizes the ``nvvm`` IR library, Numba HIP generates
  an architecture-specific LLVM bitcode library from a HIP C++ header file
  at startup of a Numba HIP program. However, a filesystem cache ensures that
  this needs to be done only once for a given session. The presence of such an
  additional caching mechanism must be considered when benchmarking.

* While Numba CUDA manually/semi-automatically creates basic device function signatures and the respective lowering
  procedures, Numba HIP does this fully-automatically from the aforementioned HIP C++ header file via the LLVM ``clang`` Python bindings.

* Furthermore, Numba HIP automatically links the HIP device library functions with the ``math`` module and uses a
  mechanism for recursive attribute resolution.

Installation
============

.. note:: Supported Numba versions

   The Numba HIP backend has been tested with the following Numba versions:

   * 0.58.*
   * 0.59.*
   * 0.60.0

   Other versions have not been tested; using the Numba HIP Backend with these versions might work or not.

Preliminaries
-------------

Make sure that your ``pip`` is upgraded by running

.. code-block:: bash

   pip install --upgrade pip

Dependencies of Numba HIP are currently partially distributed via Test PyPI.
Therefore, you need to specify it as extra index URL in your ``pip`` config as
shown below:

.. code-block:: bash

   pip config set global.extra-index-url https://test.pypi.org/simple

Install via Github URL
----------------------

The easiest way to install Numba HIP is by passing the repository URL and
optionally the branch that you want to build directly to ``pip``:

.. code-block:: bash

   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   # syntax: pip install git+<URL>@<branch>
   pip install git+https://github.com/ROCm/numba-hip.git
     # alternatively: checkout a branch like 'dev':
     # pip install git+https://github.com/ROCm/numba-hip.git@dev

Install with optional test dependencies:

.. code-block:: bash

   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   # syntax: pip install git+<URL>@<branch>[test]
   pip install git+https://github.com/ROCm/numba-hip.git[test]
     # alternatively: checkout a branch like 'dev':
     # pip install git+https://github.com/ROCm/numba-hip.git@dev[test]

Install via pip install
-----------------------

After cloning the repository, you can also install the package via ``pip install``:

.. code-block:: bash

   git clone https://github.com/ROCm/numba-hip.git
     # alternatively: checkout a branch like 'dev':
     # pip clone https://github.com/ROCm/numba-hip.git -b branch
   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   python3 -m pip install .
     # alternatively: install optional test dependencies:
     # python3 -m pip install .[test]

Create a wheel via PyPA build
-----------------------------

After cloning the repository, you can also build a Python wheel
and then distribute it (or install it):

.. code-block:: bash

   git clone https://github.com/ROCm/numba-hip.git
     # alternatively: checkout a branch like 'dev':
     # pip clone https://github.com/ROCm/numba-hip.git -b branch
   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   pip install build venv # install PyPA build and venv
   python3 -m build install .
     # alternatively: install optional test dependencies:
     # python3 -m build install .[test]
   # optional: install the wheel:
   pip install dist/*.whl

Contact
=======

Numba has a discourse forum for discussions:

* https://numba.discourse.group
