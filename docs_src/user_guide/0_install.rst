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

Installation
============

Supported Hardware
------------------

Currently, only AMD GPUs are supported.

* See the ROCm\ |trade| `Hardware_and_Software_Support
  <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`__
  page for a list of supported AMD GPUs.

Supported Operation Systems
---------------------------

Currently, only Linux is supported by the HIP Python interfaces's library
loader. The next section lists additional constraints with respect to the
required ROCm\ |trade| installation.

Software Requirements
---------------------

You must install a HIP Python version that is compatible with your
ROCm\ |trade| HIP SDK installation, or vice versa -- in particular, if you want
to use the Cython interfaces. See the
`ROCm documentation <https://rocm.docs.amd.com/en/latest/index.html>`__
for more details on how to install the ROCm\ |trade| HIP SDK.

.. _subsec_hip_python_versioning:

HIP Python Versioning
^^^^^^^^^^^^^^^^^^^^^

The ROCm\ |trade| HIP SDK is versioned according to the below scheme:

``ROCM_VERSION_MAJOR.ROCM_VERSION_MINOR.ROCM_VERSION_PATCH[...]``:

While HIP Python packages are versioned according to:

``ROCM_VERSION_MAJOR.ROCM_VERSION_MINOR.ROCM_VERSION_PATCH.HIP_PYTHON_VERSION``

where ``HIP_PYTHON_VERSION`` is the independently incremented hip-python
package version (tracked in the repo-root ``HIP_PYTHON_VERSION`` file, e.g.
``0.0.1``).

Any version of HIP Python that matches the first three numbers is suitable
for your ROCm\ |trade| HIP SDK installation.

.. admonition:: Example

   If you have the ROCm\ |trade| HIP SDK 7.13.0 installed, any
   HIP Python package with version ``7.13.0.*`` can be used.

.. tip::

   The pip commands below pin the first three components (the ROCm
   version) using PEP 440's compatible-release operator
   ``~=7.13.0.0``. This is equivalent to
   ``>=7.13.0.0, ==7.13.0.*`` — pip is allowed to pick newer
   ``HIP_PYTHON_VERSION`` updates within the ``7.13.0`` line but is
   forbidden from sliding forward to ``7.14.0``, where the pinned
   ROCm SDK might no longer be compatible.

.. note::

   The HIP Python Python packages load HIP SDK functions in a lazy manner.
   Therefore, you will likely "get away" with using "incompatible" ROCm\ |trade|
   and HIP Python pairs if the following assumptions apply:

   * You are only using Python code,
   * the definitions of the types that you use have not changed between the
     respective ROCm\ |trade| releases, and
   * you are using a subset of functions that is present in both
     ROCm\ |trade| releases.

   Both assumptions often apply.

Installation Commands
---------------------

.. important::

   Especially on older operating systems, ensure that your ``pip`` is upgraded to
   the latest version. You can upgrade it, e.g., as follows:

   .. code-block:: shell

      python3 -m pip install --upgrade pip

HIP Python ships as **seven separate wheels**, so that you only install
the runtime dependencies you actually need. The sections below cover
the three common installation shapes; pick one that matches your use
case.

Full installation (everything)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A plain ``hip-python`` install pulls in only the core and HIP
bindings plus the ``hip.*`` alias shim:

.. code-block:: shell

   python3 -m pip install hip-python~=7.13.0.0

This installs ``rocm-bindings-core``, ``rocm-bindings-hip``, and the
``hip-python`` alias. The math (``libraries``), system (``systems``),
and compiler (``compiler``) bindings are optional extras, and the CUDA
interop layer is a separate wheel. To pull in everything, request the
extras and add the interop wheel:

.. code-block:: shell

   python3 -m pip install \
       "hip-python[libraries,systems,compiler]~=7.13.0.0" \
       hip-python-interop~=7.13.0.0

Use this when you don't yet know which subset of ROCm libraries
you'll be calling.

If you have a HIP Python wheel somewhere in your filesystem:

.. code-block:: shell

   python3 -m pip install <path/to/hip_python>.whl

.. _subsec_install_wrapper_module:

Wrapper module (``hip-python``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hip-python`` wheel is a thin **alias package**. It exposes
``hip``, ``hiprtc``, ``hipblas``, etc. as re-exports of the
canonical ``rocm.bindings.*`` modules, so legacy code that does
``from hip import hip, hiprtc, hipblas`` keeps working unchanged.
It contains no bindings of its own — just a tiny shim — and
declares the binding wheels as runtime dependencies. Installing
``hip-python`` therefore drags in the bindings transitively:

.. code-block:: shell

   python3 -m pip install hip-python~=7.13.0.0

Pick this if your existing code uses the ``from hip import ...``
import style. New code should prefer importing from
``rocm.bindings.*`` directly and skip this wrapper.

.. _subsec_install_interop_module:

CUDA interoperability layer (``hip-python-interop``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hip-python-interop`` wheel ships the
``cuda.bindings.{driver,runtime,nvrtc}`` modules — drop-in
replacements for the corresponding ``cuda-python`` modules,
implemented on top of HIP. Use this to port CUDA Python code to
AMD GPUs with minimal source changes (see
:ref:`/user_guide/2_cuda_python_interop` for the porting guide):

.. code-block:: shell

   python3 -m pip install \
       hip-python-interop~=7.13.0.0

The interop wheel depends on ``rocm-bindings-hip`` (it forwards
calls into it), so pip will install that automatically.

.. _subsec_install_individual_bindings:

Individual binding components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For minimal install footprints (containerised builds, embedded
deployments) install only the binding wheels you actually need:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Wheel
     - When to install
   * - ``rocm-bindings-core``
     - Always — it provides the DLL loader and shared types every
       other binding depends on.
   * - ``rocm-bindings-hip``
     - You call ``hip.*`` (HIP runtime) or ``hiprtc.*``
       (just-in-time kernel compilation) directly.
   * - ``rocm-bindings-libraries``
     - You call into the math/FFT/random/sparse libraries
       (``hipblas``, ``hipsolver``, ``hiprand``, ``hipfft``,
       ``hipsparse``, plus the experimental ``hipblaslt``,
       ``hipsparselt``, ``hiptensor``, ``hipdnn_backend``).
   * - ``rocm-bindings-systems``
     - You call the system-level libraries: ``rccl`` (collective
       communication), ``roctx`` (profiling/tracing),
       :py:obj:`rocm.hipfile` (accelerated file I/O), ``amdsmi``
       (system management), or ``hsa`` (HSA runtime + AMD
       extensions).
   * - ``rocm-bindings-compiler``
     - You call AMD COMGR (``amd_comgr`` or the higher-level
       :py:obj:`rocm.comgr`) or the LLVM-C bindings
       (``rocm.bindings.llvm.c.*``).

Example — install only HIP + HIPRTC + the math libraries:

.. code-block:: shell

   python3 -m pip install \
       rocm-bindings-core~=7.13.0.0 \
       rocm-bindings-hip~=7.13.0.0 \
       rocm-bindings-libraries~=7.13.0.0

The transitive dependency chain (``rocm-bindings-core`` is pulled
in automatically by every other wheel) is declared in each
wheel's metadata, so you can equivalently just install the leaves
you want and let pip figure out the dependencies.

.. note::

   Some bindings (``hipfile``, ``hipblaslt``, ``hipsparselt``,
   ``hiptensor``, ``hipdnn_backend``) require shared libraries that may
   not be part of a standard ROCm installation. See the project
   README for build-from-source instructions if ``dlopen`` of the
   corresponding ``.so`` fails on your system.

.. note::

   HIP Python is published on the public PyPI under the
   ``hip-python`` / ``hip-python-interop`` /
   ``rocm-bindings-{core,hip,libraries,systems,compiler}`` project
   names. The commands above use pip's default index — no
   ``--extra-index-url`` / ``-i`` is required.

   Older releases were only available on TestPyPI for internal
   testing. Documentation that still recommends
   ``-i https://test.pypi.org/simple`` is out of date — prefer the
   default-index commands shown here.

.. |trade| unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
