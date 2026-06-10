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

The build produces **seven Python wheels**, one per package directory
under ``packages/``. All seven contribute to the same two PEP 420
implicit namespace packages at runtime — ``rocm.bindings.*`` and
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
       ``hipsparselt``, ``hiptensor``, ``hipdnn``. Module list is
       generator-managed.

   * - ``rocm-bindings-systems``
     - System-level libraries: ``rccl``, ``roctx``, ``hipfile``,
       ``amdsmi``, ``hsa``. Includes the high-level
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

The unified build orchestrator at ``packages/CMakeLists.txt`` builds
all seven via the ``all_wheels`` aggregate target. A development loop
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

The TOC under ``docs_src/sphinx/_toc.yml.in`` is hand-maintained
with one subtree per wheel; generator-emitted per-module pages slot
in without TOC edits because the codegen renders an updated copy on
every codegen run.

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
- Libraries outside the allowlist (``libamdhip64.so``,
  ``libhsa-runtime64.so.1``, the optional bundled ROCm math
  libraries when present, …) are explicitly excluded — hip-python
  ``dlopen``-resolves these at runtime via the
  :py:obj:`rocm.bindings.util.loader` shim, so they MUST NOT be
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
