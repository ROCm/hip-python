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
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

.. _building_from_source:

Building from Source
====================

This page summarises the hip-python build system for users who want to
build the wheels (or just the docs) themselves. It is a brief tour;
the full design — including CMake helper APIs, the
generator-emitted CMake includes, sdist build mechanics, and the
Cython version requirement — lives in
:download:`share/design/BUILDING.md <../../share/design/BUILDING.md>`
in the source tree.

Components
----------

The build produces **eight Python wheels**, one per package directory
under ``packages/``. The binding wheels contribute to the same two
PEP 420 implicit namespace packages at runtime — ``rocm.bindings.*`` and
``cuda.bindings.*`` — so a downstream user installs only the wheels
they need.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Wheel
     - Provides

   * - ``rocm-bindings-core``
     - DLL loader, shared Cython types (``Pointer``, ``CStr``,
       ``NDBuffer``, …), lazy ROCm library lookup. **Handcoded**, not
       generator output. Required by every other binding wheel.

   * - ``rocm-bindings-hip``
     - ``hip`` and ``hiprtc`` bindings (high-level + ``cy*`` C-level
       pairs). Helpers (``_hip_helpers``, ``_hiprtc_helpers``) are
       handcoded.

   * - ``rocm-bindings-libraries``
     - Math/FFT/random/sparse libraries: ``hipblas``, ``hipblaslt``,
       ``hipsolver``, ``hiprand``, ``hipfft``, ``hipsparse``,
       ``hipsparselt``. Module list is generator-managed, and modules
       the ROCm installation cannot supply are dropped at configure
       time.

   * - ``rocm-bindings-systems``
     - System-level libraries: ``rccl``, ``roctx``, ``hipfile``,
       ``amdsmi``. Includes the high-level
       :py:obj:`rocm.hipfile` Pythonic wrapper. Optional bundling of
       ``libhipfile.so``.

   * - ``rocm-bindings-compiler``
     - LLVM-C bindings, AMD COMGR bindings (with the high-level
       :py:obj:`rocm.comgr` wrapper). Optional bundling of
       ``libLLVM.so``.

   * - ``hip-python-interop``
     - CUDA interop layer: ``cuda.bindings.{driver,runtime,nvrtc}``.

   * - ``hip-python``
     - Pure-Python alias package: ``from hip import hip, hiprtc, ...``
       re-exports ``rocm.bindings.*``.

   * - ``numba-hip``
     - The ROCm HIP backend for Numba (``numba.hip``). Pure
       Python, versioned independently of the binding wheels; see
       :doc:`/user_guide/4_numba_hip`.

The unified build orchestrator at ``packages/CMakeLists.txt`` builds
all eight via the ``all_wheels`` aggregate target. A development loop
can also build one package at a time (``cd packages/<pkg> && python3
-m build --wheel --no-isolation``) without touching the others.

Quick start
-----------

Two minimal recipes — one for wheels, one for docs:

**All wheels at once.** From a checkout with ``ROCM_PATH`` pointing
at a ROCm installation (default ``/opt/rocm``):

.. code-block:: bash

   cd packages
   cmake -B build
   cmake --build build --target all_wheels -j$(nproc)

Wheels land in ``packages/build/dist/`` (override with
``-DHIP_PYTHON_WHEEL_OUTPUT_DIR=...``).

**Single package only** — useful during development:

.. code-block:: bash

   cd packages/rocm-bindings-core
   python3 -m build --wheel --no-isolation

Frequently used build options
-----------------------------

Two options decide what ``all_wheels`` actually contains beyond the
binding wheels:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Option
     - Default
     - Effect
   * - ``HIP_PYTHON_BUNDLE_LIBLLVM``
     - ``ON`` (Linux), ``OFF`` (Windows)
     - Bundles a shared LLVM into the ``rocm-bindings-compiler`` wheel.
       The ``rocm.bindings.llvm.*`` modules resolve a shared LLVM on
       their first call; ROCm ships one on Linux but only static
       archives on Windows, where the build links an ``LLVM.dll`` from
       them. Off by default there because it adds about 75 MB to the
       wheel — without it those modules import but raise on first use.
       The published Windows wheel *is* built with it on, so switch it
       on to match: a plain source build is a downgrade.
   * - ``HIP_PYTHON_BUILD_NUMBA_HIP``
     - ``ON``
     - Builds the pure-Python ``numba-hip`` wheel. It carries its own
       version and depends on the compiler bindings, so on Windows pair
       it with ``HIP_PYTHON_BUNDLE_LIBLLVM=ON`` — otherwise the
       ``numba.hip`` you build is less capable than the published one.

For example, a Windows build with both:

.. code-block:: powershell

   cd packages
   cmake -B build -DHIP_PYTHON_BUNDLE_LIBLLVM=ON -DHIP_PYTHON_BUILD_NUMBA_HIP=ON
   cmake --build build --target all_wheels

The full option list — build type, ROCm path, ``auditwheel`` repair,
configure-time codegen, stub generation — is in
:download:`share/design/BUILDING.md <../../share/design/BUILDING.md>`.

Documentation build
-------------------

Sphinx HTML docs are an **independent CMake target** — they do not
depend on ``all_wheels`` and can be built before, after, or in
parallel with the wheels. Sphinx parses ``.py`` / ``.pyi`` source
files directly via ``sphinx-autoapi``, so no compiled extension
needs to be installed first.

The target is gated on ``HIP_PYTHON_BUILD_DOCS=ON``:

.. code-block:: bash

   cd packages
   cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
   cmake --build build --target docs
   # open ../docs/index.html

Relevant CMake options:

.. list-table::
   :header-rows: 1
   :widths: 36 18 46

   * - Option
     - Default
     - Effect
   * - ``HIP_PYTHON_BUILD_DOCS``
     - ``OFF``
     - Gates target creation. Fails fast if ``python -m sphinx``
       isn't importable.
   * - ``HIP_PYTHON_DOCS_OUTPUT_DIR``
     - ``<repo>/docs``
     - Destination for the rendered HTML. Override for per-version
       layouts (``docs/rocm-rel-7.13.0``) or direct hosting paths
       (``/var/www/...``).
   * - ``HIP_PYTHON_DOCS_DOCTREE_DIR``
     - ``<build>/docs/_doctrees``
     - Sphinx intermediate cache.

Wheels and docs can build concurrently:

.. code-block:: bash

   cmake --build build --target all_wheels docs -j$(nproc)

``docs_src/sphinx/_toc.yml.in`` is itself generator output: the codegen
renders it from the handcoded ``_toc.yml.in.in`` template on every run,
substituting the discovered module list into one subtree per wheel. The
handwritten user-guide pages are listed inline in the template, so they
are the part to edit by hand; per-module API pages slot in without any
TOC edit.

.. _sec_build_windows:

Building on Windows
-------------------

Wheels are published for Windows alongside the Linux ones, so building from
source here is for development or for a configuration the published wheels do
not carry. ``ci\internal\build-wheels.ps1`` is the PowerShell counterpart to the
Linux ``ci/internal/build-wheels.sh`` and drives the same ``all_wheels`` target:

.. code-block:: powershell

   . ci\internal\env-rocm.ps1            # sets ROCM_PATH and PATH
   python -m pip install -r ci\internal\requirements-build.txt
   ci\internal\build-wheels.ps1          # add -Light for core + hip + compiler
   ci\internal\test.ps1                  # examples + unit-test suites

The MSVC environment is imported automatically, so the script behaves the same
from a plain PowerShell prompt, a CI runner, or a Developer shell.

``test.ps1`` takes ROCm from ``ROCM_PATH`` by default, which suits a system
install or an unpacked tarball. Pass ``-UseRocmSdkWheels`` to install ROCm into
the test venv from the ``rocm_sdk`` wheels instead, the way most users get it:

.. code-block:: powershell

   ci\internal\test.ps1 -UseRocmSdkWheels          # target read from the GPU
   ci\internal\test.ps1 -UseRocmSdkWheels -GfxArch gfx1100

The ROCm version defaults to the one the bindings were generated against, and
the ``rocm-sdk-device-*`` wheel to the target of the GPU in the machine. Those
wheels spread ROCm over several trees, so do not point ``ROCM_PATH`` at one of
them by hand: ``_rocm_sdk_devel`` in particular holds a second copy of
``libhipblaslt.dll`` next to an incomplete set of Tensile kernels, and hipBLASLt
looks for kernels beside whichever copy it loaded, so that one crashes inside the
algorithm search. Let the bindings resolve libraries through
``rocm_sdk.find_libraries``, which knows which tree owns each library.

Where the Windows build differs from Linux:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Aspect
     - Windows behaviour
   * - Host compiler
     - MSVC (``cl``) by default, because the host CPython is built with it and
       the extension modules must match its CRT and ABI. ``-UseRocmClang``
       switches to ``amdclang-cl``, the MSVC-compatible clang driver: it keeps
       the same ABI and diagnoses portability problems MSVC stays quiet about.
   * - Build type
     - ``CMAKE_BUILD_TYPE=Release`` is set explicitly. A multi-config MSVC
       default of ``Debug`` links against the debug CRT, which the release
       CPython does not provide.
   * - Generator
     - Ninja. The Linux script forces ``make`` only to dodge a GCC jobserver
       bug on the largest generated ``.c`` files.
   * - ``auditwheel``
     - Not used; it is a Linux ELF retagger. The wheels are emitted with a
       native ``win_amd64`` tag.
   * - Module set
     - Smaller. The components ROCm does not ship for Windows are detected and
       skipped at configure time; see the table in :ref:`sec_install`.
   * - Library naming
     - ROCm's Windows DLLs are version-suffixed (``amdhip64_7.dll``,
       ``hiprtc0714.dll``) or keep a Unix ``lib`` prefix
       (``libhipblaslt.dll``). :py:obj:`rocm.bindings.util.paths` resolves
       these by scanning ``ROCM_PATH`` rather than by assuming a flat name.

``manylinux`` repair via ``auditwheel``
---------------------------------------

For distribution-friendly ``manylinux_*`` wheels, enable
``HIP_PYTHON_AUDITWHEEL_REPAIR=ON``. Each per-package wheel target
then runs ``auditwheel repair`` after the underlying ``python -m
build`` step:

.. code-block:: bash

   cd packages
   cmake -B build -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON
   cmake --build build --target all_wheels -j$(nproc)

What this does in practice:

- Each freshly-built wheel is inspected for the externally-loaded
  shared libraries it depends on.
- Libraries that match the ``manylinux`` allowlist (the C runtime
  and a small standard set) are left alone.
- Libraries outside the allowlist (the optional bundled ROCm math
  libraries when present, …) are explicitly excluded — hip-python
  ``dlopen``-resolves these at runtime via the
  ``rocm.bindings.util.loader`` shim, so they MUST NOT be
  bundled into the wheel.
- The ``manylinux_2_17_x86_64`` (or matching) tag is added to the
  wheel filename, making it pip-installable on systems older than
  the build host's glibc.

The result is a wheel that runs on a wider range of distributions
than the unrepaired one, while still loading the system's actual
ROCm libraries at import time.

See also
--------

- ``share/design/BUILDING.md`` in the source tree — full build-system
  design including CMake helper APIs (``hip_python_initialize``,
  ``hip_python_add_cython_module``, …), the
  generator-emitted CMake include files, sdist build mechanics, and
  Cython version requirements.
- ``share/design/CODEGEN.md`` — how the generator-managed parts of
  the build (module lists, Cython sources, TOC entries) are
  produced.
- ``share/design/BINDINGS.md`` — the anatomy of the generated
  bindings (two-tier wrapper layout, GIL semantics, pointer-arg
  classification, handcoded helpers).
