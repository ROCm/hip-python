*********
Numba HIP
*********

This repository provides a ROCm™ HIP backend for Numba.

.. admonition:: **For AMD GPUs on Linux**

    Numba HIP is for AMD Radeon™ and AMD Instinct™ accelerators.
    CUDA® devices are not supported by Numba HIP.

    As Numba HIP generates LLVM bitcode device libraries on-the-fly via the
    ROCm compiler and delegates runtime tasks to the HIP runtime via the HIP
    Python bindings, it does itself not pose any limitation on the
    supported AMD GPU devices.

    The ROCm on Radeon QA team has tested Numba HIP 0.1.4 on systems 
    with AMD Radeon™ RDNA3, and AMD Radeon™ RDNA4 accelerators ROCm 7.1.0.
    The QA team's testing focused on the following AMD Radeon™ cards:
 
    * AMD Radeon™ RX 7900 XTX
    * AMD Radeon™ RX 7900 XT
    * AMD Radeon™ PRO W7900D
    * AMD Radeon™ PRO V710
    * AMD Radeon™ RX 9060 XT
    * AMD Radeon™ AI PRO R9700

    The authors have tested all versions of Numba HIP before version 0.1.4
    (so for ROCM versions before 7.1.1) predominantly on systems with
    AMD Instinct™ MI210X (gfx90a).
    
    The authors have tested Numba HIP 0.1.6 on ROCm 7.1.1 and ROCm 7.2.0
    systems with the following architectures:

    * gfx1030v (AMD Radeon™ PRO V620)
    * gfx1100p (AMD Radeon™ PRO W7800)
    * gfx1102 (AMD Radeon™ RX 7600 XT)
    * gfx1201 (AMD Radeon™ RX 9070 XT)
    * gfx90a (AMD Instinct™ MI210X/MI250
    * gfx942 (AMD Instinct™ MI300A/MI300X/MI308X/MI325X)

.. admonition:: **Experimental project**
    
    This project primarily aims to support the AMD ROCm™ Data Science toolkit
    (`ROCm-DS <https://rocm.docs.amd.com/projects/rocm-ds/en/latest/index.html>`_)
    Most features that have been implemented were driven by ROCm-DS.
    
    However, we are also happy to get feedback from early adopters on their experience with the new Numba HIP backend.
    So if you give Numba HIP a try, let us know about your experience. We are looking forward to receiving 
    your suggestions, issue reports, and pull requests.

About Numba: A Just-In-Time Compiler for Numerical Functions in Python
======================================================================

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

The following Numba CUDA features are not available via Numba HIP:

* Cooperative groups support (ex: ``cg.this_grid()``,
  ``cg.this_grid().sync()``)
* Atomic operations for tuple and array types,
* Runtime kernel debugging functionality,
* Device code printf,
* HIP Simulator equivalent to CUDA Simulator (low priority, users can
  potentially reuse CUDA simulator),
* Half precision (fp16) operations.

Note further that so far only limited effort has been spent on optimizing the
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
   * 0.61.2 (Numba HIP 0.1.5+)

   Other versions have not been tested; using the Numba HIP backend with these versions might work or not.

Important things to know before installing
------------------------------------------

Make sure that your ``pip`` is upgraded by running

.. code-block:: bash

   pip install --upgrade pip

Dependencies of Numba HIP are currently partially distributed via Test PyPI.
Therefore, you need to specify an extra index URL in your ``pip`` config
as shown below:

.. code-block:: bash

   pip config set global.extra-index-url https://test.pypi.org/simple

Those dependencies further are depending on a particular ROCm release.
We use optional dependency lists to make this configurable; see the
``pyproject.toml`` file for more details.
To install dependencies for a ROCm release of a particular version, you need
to specify an dependency key in the format
``rocm-<major>-<minor>-<patch>`` (example: ``rocm-7-2-0``) when building
the Numba HIP package. If you leave the key aside, ``pip`` will either use
already installed versions of the dependencies or install the latest release
of these dependencies, which are compatible with the most recent release of ROCm
but potentially not with older ROCm releases.

Install via Github URL
----------------------

The easiest way to install Numba HIP is by passing the repository URL and
optionally the branch that you want to build directly to ``pip``:

.. code-block:: bash

   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   # syntax 1: pip install git+<URL>@<branch>
   # syntax 2: pip install "numba-hip[rocm-<major>-<minor>-<patch>] @ git+<URL>@<branch>"
   pip install "numba-hip[rocm-7-2-0] @ git+https://github.com/ROCm/numba-hip.git"
     # alternatively: checkout a branch like 'dev':
     # pip install "numba-hip[rocm-7-2-0] @ git+https://github.com/ROCm/numba-hip.git@dev"

.. note:: ROCm key must agree with your environment

   Do not forget to change the ROCm version ``rocm-7-2-0``
   (format: ``rocm-<major>-<minor>-<patch>``) to a key that agrees with your
   ROCm installation so that dependency versions compatible with your
   ROCm installation are installed by ``pip``.

Install with optional test dependencies:

.. code-block:: bash

   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   # syntax 1: pip install "numba-hip[test] @  git+<URL>@<branch>"
   # syntax 2: pip install "numba-hip[rocm-<major>-<minor>-<patch>,test] @ git+<URL>@<branch>"
   pip install "numba-hip[rocm-7-2-0,test] @ git+https://github.com/ROCm/numba-hip.git"
     # alternatively: checkout a branch like 'dev':
     # pip install "numba-hip[rocm-7-2-0,test] @ git+https://github.com/ROCm/numba-hip.git@dev"

Install via pip install
-----------------------

After cloning the repository, you can also install the package via ``pip install``:

.. code-block:: bash

   git clone https://github.com/ROCm/numba-hip.git
     # alternatively: checkout a branch like 'dev':
     # pip clone https://github.com/ROCm/numba-hip.git -b branch
   pip install --upgrade pip
   pip config set global.extra-index-url https://test.pypi.org/simple
   python3 -m pip install .[rocm-7-2-0]
     # alternatively: install optional test dependencies:
     # variant 1: python3 -m pip install .[test]
     # variant 2: python3 -m pip install .[rocm-7-2-0,test]

.. note:: ROCm key must agree with your environment

   Do not forget to change the ROCm version ``rocm-7-2-0``
   (format: ``rocm-<major>-<minor>-<patch>``) to a key that agrees with your
   ROCm installation so that dependency versions compatible with your
   ROCm installation are installed by ``pip``.

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
   pip install build # install PyPA build 
   python3 -m build --wheel .

   # optional: install the wheel:
   pip install dist/*.whl
   # alternatively: install optional test dependencies:
   # pip3 install dist/numba_hip-0.1-py3-none-any.whl[rocm-7-2-0]

.. note:: ROCm key must agree with your environment

   Do not forget to change the ROCm version ``rocm-7-2-0``
   (format: ``rocm-<major>-<minor>-<patch>``) to a key that agrees with your
   ROCm installation so that dependency versions compatible with your
   ROCm installation are installed by ``pip``.

Contact
=======

Numba has a discourse forum for discussions:

* https://numba.discourse.group
