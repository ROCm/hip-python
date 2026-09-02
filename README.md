<!-- MIT License
  --
  -- Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
  --
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  --
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  --
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->
# HIP Python Source Repository

This repository provides low-level Python and Cython Bindings
for HIP and an interoperability layer for CUDA&reg; Python programs
(Python and Cython).

## Requirements

* **Linux** is the primary supported platform (prebuilt packages and code).
  * Prebuilt packages distributed via PyPI are only provided for
    Linux systems that match the `manylinux_2_17_x86_64` tag.
* **Windows** is supported from source: the wheels build with MSVC and
  the test suites pass there
  ([`ci/internal/build-wheels.ps1`](ci/internal/build-wheels.ps1),
  [`ci/internal/test.ps1`](ci/internal/test.ps1)). No prebuilt Windows
  wheels are published yet. ROCm does not ship every component for
  Windows, so a Windows install offers fewer bindings — the user guide
  lists which ones and why. A wheel-installed ROCm SDK can also end up
  incomplete there; see [Known Limitations](#known-limitations).
* Requires that a compatible ROCm&trade; HIP SDK is installed on your system.
  * Source code is provided only for particular ROCm versions.
    * See the `git` branches tagged with `release/rocm-rel-X.Y[.Z]`
  * Prebuilt packages are built only for particular ROCm versions.
* Prebuilt packages may be provided only for **Python 3.10 and newer**,
  even though the packages themselves declare `requires-python = ">=3.9"`.
  The published wheel set is built for CPython 3.10 and 3.11 plus an abi3
  wheel whose 3.12 floor also serves every later CPython
  ([`ci/internal/build-wheels-manylinux.sh`](ci/internal/build-wheels-manylinux.sh)),
  so a 3.9 interpreter has to build from source.

> [!NOTE]
> You may find that packages for one ROCm&trade; release are compatible with
> the ROCm&trade; HIP SDK of another release as the HIP Python functions load
> HIP C functions in a lazy manner.

### Build requirements

* Linux or Windows
* A C compiler (GCC/Clang on Linux, MSVC on Windows)
* `python3` + `venv`, and a shell the CI scripts run in (`bash` on Linux,
  PowerShell on Windows)
* The ROCm&trade; HIP SDK
* Python 3.9+ with `pip>=24.0`.

## Install Prebuilt Packages

***

> [!IMPORTANT]
> Ensure that `pip` has at least version `24.0`, please upgrade it otherwise.

***

### Via PyPI

First identify the first three digits of the version number of your
ROCm&trade; installation (the `~=$rocm_version.0` pin matches every
build of that ROCm patch release). Then choose what to install based
on the ROCm libraries you need:

<!-- markdownlint-disable  MD013 -->

```shell
# Minimum: HIP runtime + HIPRTC + the hip.* alias namespace
# (pulls rocm-bindings-core + rocm-bindings-hip + hip-python).
python3 -m pip install hip-python~=$rocm_version.0

# Math libraries (hipBLAS, hipSOLVER, hipRAND, hipFFT, hipSPARSE).
python3 -m pip install hip-python[libraries]~=$rocm_version.0

# Communication + tracing (RCCL, ROCTX).
python3 -m pip install hip-python[systems]~=$rocm_version.0

# Compiler bindings (LLVM-C + AMD COMGR + bundled libLLVM.so).
python3 -m pip install hip-python[compiler]~=$rocm_version.0

# Several extras at once:
python3 -m pip install "hip-python[libraries,systems,compiler]~=$rocm_version.0"

# CUDA Python interoperability layer
# (cuda.bindings.{driver,runtime,nvrtc}).
python3 -m pip install hip-python-interop~=$rocm_version.0
```

You can also pull individual `rocm-bindings-*` wheels directly
(`rocm-bindings-core`, `-hip`, `-libraries`, `-systems`, `-compiler`)
if you don't want the `hip.*` alias namespace that the `hip-python`
metapackage provides.

<!-- markdownlint-enable  MD013 -->

> [!NOTE]
> See the HIP Python user guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Build from Source

The build uses CMake with `scikit-build-core` as the Python build backend.
It produces eight wheels:

- `rocm-bindings-core` — DLL loader, types, ROCm path resolution
- `rocm-bindings-hip` — `hip` and `hiprtc` Python bindings
- `rocm-bindings-libraries` — math libraries: `hipblas`, `hipblaslt`*,
  `hipsolver`, `hiprand`, `hipfft`, `hipsparse`, `hipsparselt`*
  (* = experimental, see [Known Limitations](#known-limitations))
- `rocm-bindings-systems` — system-level libraries: `rccl`
  (collective communication), `roctx` (profiling/tracing),
  `hipfile`*, `amdsmi`
- `rocm-bindings-compiler` — LLVM-C and AMD COMGR bindings (with optional
  bundled `libLLVM.so` / `LLVM.dll`)
- `hip-python-interop` — `cuda.bindings.{driver,runtime,nvrtc,cufile}` interop
  layer, plus `pynvml` (NVML, AMD SMI-backed), `nvtx` (NVTX, ROCTX-backed) and
  minimal `cuda.core.Device` (HIP-backed) compatibility shims
- `hip-python` — exposes the `hip.*` namespace as an alias of
  `rocm.bindings.*`, so that `from hip import hip, hiprtc, hipblas` (etc.)
  keeps working unchanged
- `numba-hip` — the ROCm HIP target for Numba (`numba.hip`), a pure-Python
  wheel with its own version; built by default and skippable with
  `-DHIP_PYTHON_BUILD_NUMBA_HIP=OFF`

> [!NOTE]
> The HIP-side `from hip import hip, hiprtc` (and friends) are aliases
> of the per-package `rocm.bindings.*` modules — both styles are
> supported. The CUDA interop side has **no such alias package**: only
> `from cuda.bindings import driver, runtime, nvrtc` works (`cuda.bindings`
> is the package, `driver` / `runtime` / `nvrtc` are the modules).
> The interop wheel additionally ships a top-level `pynvml` (NVML) shim
> backed by AMD SMI, an `nvtx` (NVTX) shim backed by ROCTX, and a minimal
> `cuda.core.Device` shim backed by HIP, so `import pynvml`, `import nvtx`
> and `from cuda.core import Device` keep working on AMD GPUs.
>
> **New code should prefer** `from rocm.bindings import hip, hiprtc` and
> `from cuda.bindings import driver, runtime, nvrtc` directly — these
> match the per-package layout used throughout the documentation and
> examples.

> [!NOTE]
> Most users do **not** need to build from source — prebuilt wheels are
> distributed via PyPI for every supported ROCm release. See the [Install
> Prebuilt Packages](#install-prebuilt-packages) section above.

### Quick Start (build all packages)

1. Install ROCm and Python development tools:

   ```shell
   # Ubuntu:
   sudo apt install python3-pip python3-venv python3-dev
   ```

2. Check out the feature branch `release/rocm-rel-X.Y[.Z]` for your particular
   ROCm&trade; installation.

3. Create a virtual environment and install build requirements:

   ```shell
   python3 -m venv .venv
   . .venv/bin/activate
   pip install --upgrade pip
   pip install build "scikit-build-core>=0.11.2" "cmake>=3.26" "ninja>=1.11" "cython>=3.1.0"
   ```

   > **Cython 3.1.0 is required.** Cython 3.0.x silently miscompiles
   > a `cdef T x = <T>expr` initializer when `T` contains the inner
   > `*const *` pattern (e.g. `const char *const *`), leaving the local
   > NULL — the resulting wrapper segfaults inside the backend (e.g.
   > `hiprtcCompileProgram`). Fixed upstream in Cython 3.1.0; see
   > `share/design/BUILDING.md` (under "Cython version requirement")
   > for details.

4. Configure and build all wheels:

   ```shell
   cd packages
   cmake -B build
   cmake --build build --target all_wheels -j$(nproc)
   ```

   Wheels for all eight packages land in `packages/build/dist/`.

5. Install the wheels:

   ```shell
   pip install build/dist/*.whl
   ```

### Build Individual Packages

Each package has its own `pyproject.toml` and can be built standalone — useful for
development loops on a single package. Run the unified configure once first to
populate the per-package `VERSION` and shared cmake helper (both gitignored):

```shell
# One-time: populate per-package VERSION + cmake helper from the repo-root files
cd packages && cmake -B build && cd ..

# Build just rocm-bindings-core:
cd packages/rocm-bindings-core
python3 -m build --wheel --no-isolation

# Build just rocm-bindings-hip:
cd packages/rocm-bindings-hip
python3 -m build --wheel --no-isolation
```

Standalone per-package builds skip `auditwheel repair` (the resulting wheel is tagged
`linux_x86_64` rather than `manylinux_*`) and only build that one package's targets.
Prefer the unified CMake build above when you want all wheels and/or manylinux compatibility.

### Build and Install from sdist

Each package ships a self-contained source distribution. The unified CMake
build provides per-package `<pkg>_sdist` targets and an aggregate
`all_sdists` that mirrors `all_wheels`:

```shell
cd packages
cmake -B build
cmake --build build --target all_sdists       # build sdists for every enabled package
# or one at a time:
cmake --build build --target core_sdist
cmake --build build --target compiler_sdist

# Install:
pip install --no-build-isolation build/dist/rocm_bindings_compiler-*.tar.gz
```

The sdist tarball bundles the per-package `VERSION`, the shared cmake helper, every
`.pxd`/`.pyx`/`.pyi`/`.py` source, the per-package `CMakeLists.txt`, and (for
`rocm-bindings-compiler`) the `src/` subtree that builds the bundled `libLLVM.so`.
Installing the sdist re-runs CMake against the package's own self-contained
`CMakeLists.txt` — no parent directory or repo checkout required.

To build a subset via the unified CMake build, disable the packages you don't want at
configure time:

```shell
# Build only core, hip, and libraries (skip compiler and interop):
cd packages
cmake -B build \
  -DHIP_PYTHON_BUILD_COMPILER=OFF \
  -DHIP_PYTHON_BUILD_INTEROP=OFF
cmake --build build --target all_wheels
```

You can also build a single package's wheel from the unified build:

```shell
cd packages
cmake -B build
cmake --build build --target core_wheel        # rocm-bindings-core only
cmake --build build --target hip_wheel         # rocm-bindings-hip only
cmake --build build --target libraries_wheel   # rocm-bindings-libraries only
cmake --build build --target systems_wheel     # rocm-bindings-systems only
cmake --build build --target compiler_wheel    # rocm-bindings-compiler only
cmake --build build --target interop_wheel     # hip-python-interop only
cmake --build build --target hip_python_wheel  # hip-python metapackage only
```

### Regenerating `.pyi` stubs for handcoded Cython modules (developer)

A handful of Cython modules are handcoded rather than generated by
interfacegen (`rocm.bindings.util.types` in `rocm-bindings-core`;
`_hip_helpers` and `_hiprtc_helpers` in `rocm-bindings-hip`). Their
`.pyi` type stubs are committed to git so sphinx-autoapi, mypy, and
IDEs see real signatures. The handcoded DLL loaders next to
`util.types` get none: they are `cimport`-only, with no Python-visible
API to describe.

When you edit one of those `.pyx` files and want to refresh the
matching `.pyi`, opt in to the developer-only stubgen targets:

```shell
pip install mypy                                    # one-time
cd packages
cmake -B build -DHIP_PYTHON_ENABLE_STUBGEN=ON
cmake --build build --target all_stubs              # all handcoded modules
# or one package at a time:
cmake --build build --target core_stubs
cmake --build build --target hip_stubs
```

The regenerated `.pyi` lands in the source tree next to the `.pyx`.
Inspect the diff with `git diff`, then commit `<module>.pyx` and
`<module>.pyi` together.

End-user `pip install` from sdist or wheel does **not** invoke
stubgen, and `mypy` is **not** a build-system dependency. The list
of modules that need stubbing lives in a single repo-spanning CMake
variable `HIP_PYTHON_STUBGEN_MODULES` in `packages/CMakeLists.txt` —
see [share/design/BUILDING.md](share/design/BUILDING.md)
"Regenerating stubs for handcoded Cython modules" for the full
developer workflow and how to add a new module to the list.

### Build Options

Pass options to CMake via `-D<NAME>=<VALUE>` at configure time.

```shell
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Custom ROCm path
cmake -B build -DROCM_PATH=/opt/rocm-7.13

# Production manylinux wheels (requires auditwheel)
cmake -B build -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON

# Custom wheel output directory
cmake -B build -DHIP_PYTHON_WHEEL_OUTPUT_DIR=/output

# Use sccache
cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache

# Combine multiple options
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=/output
cmake --build build --target all_wheels -j16
```

For a clean rebuild, just remove the build directory:

```shell
rm -rf packages/build && cmake -S packages -B packages/build && cmake --build packages/build --target all_wheels
```

For deeper documentation:

- [share/design/BUILDING.md](share/design/BUILDING.md) — Build system architecture, helper functions, package layout, full CMake target/option reference
- [share/design/CODEGEN.md](share/design/CODEGEN.md) — How the interfacegen code generator interacts with the hip-python source tree to produce a release
- [share/design/BINDINGS.md](share/design/BINDINGS.md) — The bindings emission contract: how generated `rocm.bindings.*` modules are structured

> [!NOTE]
> See the HIP Python developer guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

### Build Configuration Options

<!-- markdownlint-disable  MD013 -->

| Option | Default | Effect |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` (or `$ROCM_PATH`/`$ROCM_HOME`) | Path to the ROCm installation. |
| `HIP_PLATFORM` | `amd` | HIP backend selector. Only `amd` and `hcc` are supported. |
| `HIP_PYTHON_BUILD_<NAME>` | `ON` | Per-package opt-in: `CORE`, `HIP`, `LIBRARIES`, `SYSTEMS`, `COMPILER`, `INTEROP`, `HIP_PYTHON`, `NUMBA_HIP`. |
| `HIP_PYTHON_RUNTIME_LINKING` | `ON` | When `ON`, generated extensions resolve ROCm shared libraries lazily at runtime; when `OFF`, they link against them at build time. |
| `HIP_PYTHON_ENABLE_LIB_<NAME>` | `ON` | Per-library toggle inside `rocm-bindings-libraries` (e.g. `HIP_PYTHON_ENABLE_LIB_HIPRAND=OFF`). |
| `HIP_PYTHON_BUNDLE_LIBLLVM` | `ON`, `OFF` on Windows | Bundle a shared LLVM inside the `rocm-bindings-compiler` wheel. ROCm ships none for Windows, so there it is linked from the static archives and costs ~75 MB, hence the opt-in. |
| `HIP_PYTHON_FORCE_BUILD_LIBLLVM` | `OFF` | Link the bundled LLVM from the static archives even where ROCm ships a shared one (always the case on Windows). |
| `HIP_PYTHON_AUDITWHEEL_REPAIR` | `OFF` | Run `auditwheel repair` to produce manylinux wheels. |
| `HIP_PYTHON_WHEEL_OUTPUT_DIR` | `${CMAKE_BINARY_DIR}/dist` | Wheel output directory. |
| `HIP_PYTHON_BUILD_DOCS` | `OFF` | Build the Sphinx HTML documentation as a CMake target (`docs`). |
| `HIP_PYTHON_DOCS_OUTPUT_DIR` | `<repo>/docs` | Destination for the rendered HTML docs. |
| `HIP_PYTHON_DOCS_DOCTREE_DIR` | `<build>/docs/_doctrees` | Sphinx intermediate cache. |
| `CMAKE_BUILD_TYPE` | `Release` | Standard CMake build type. |

<!-- markdownlint-enable  MD013 -->

### Build the Documentation

The Sphinx documentation is a separate, optional CMake target that runs in
parallel to (and independently of) the wheel build. It uses
[`sphinx-autoapi`](https://sphinx-autoapi.readthedocs.io/) to parse Python
sources and `.pyi` stubs directly, so it does **not** require the wheels to
be built or installed first.

```shell
# Install Sphinx + dependencies (one-time):
pip install -r docs_src/sphinx/requirements.txt

# Configure and build the docs:
cd packages
cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target docs
# open ../docs/index.html
```

By default the rendered HTML lands in `<repo>/docs/`. Override
`HIP_PYTHON_DOCS_OUTPUT_DIR` to redirect anywhere — e.g. for per-version doc
hosting:

```shell
cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON \
               -DHIP_PYTHON_DOCS_OUTPUT_DIR=docs/rocm-rel-7.13.0
cmake --build build --target docs
```

The doc input language is **reStructuredText** (under `docs_src/`), distinct
from the Markdown READMEs and design documents at the repo root.

## Developer

Tooling and scripts for contributors.

**Start here.** [CONTRIBUTING.md](CONTRIBUTING.md) routes a change to the place
it belongs — the code generator or the handcoded components — and names the
test loop for each. The generated bindings must never be edited in place.

**Code generator.** The Cython/Python bindings are not handwritten — they are
generated from the ROCm C headers by an in-tree code generator:

- [`tools/interfacegen`](tools/interfacegen) — the clang-based binding generator
  (with its own README and design docs).
- [`tools/hip-python-generate`](tools/hip-python-generate) — the hip-python
  codegen CLI/recipe that drives interfacegen for this repo.

See [share/design/CODEGEN.md](share/design/CODEGEN.md) and
[share/design/BINDINGS.md](share/design/BINDINGS.md) for the generator contract.

**CI scripts** (under [`ci/`](ci)):

- `ci/internal/build-wheels.sh` / `ci/internal/build-wheels.ps1` — build the
  package wheels (full or light mode) on Linux / Windows.
- `ci/internal/test.sh` / `ci/internal/test.ps1` — run the example, interop,
  bindings and numba-hip test suites against the built wheels.
- `ci/internal/env-rocm.ps1` — locate ROCm and bootstrap the MSVC environment
  for the two Windows scripts above.
- `ci/internal/generate-bindings.sh` — the standalone codegen run that produces
  a release branch's generated tree; `ci/internal/commit-bindings.sh` commits it.
- `ci/internal/prepare-release.sh` — author the release-only `VERSION.in` template
  on a `release/rocm-rel-*` branch.
- `ci/internal/build-hipfile.sh` — build and install hipFILE, which ROCm does
  not ship (see [share/design/HIPFILE.md](share/design/HIPFILE.md)).
- `ci/docs/build.sh` — render the Sphinx documentation (thin wrapper over the
  `docs` CMake target).
- `ci/docs/regenerate-stubs.sh` — regenerate the handcoded-Cython `.pyi` stubs.

## Known Limitations

### Windows: an incompletely installed ROCm SDK

The pip ROCm SDK unpacks its development tree on first use, and part of
that tree consists of symlinks. Creating a symlink on Windows needs the
create-symlink privilege, which a normal process only holds with
Developer Mode enabled or when running elevated. Without it the
expansion can stop early and still report success, leaving a
development tree that silently misses files. The CMake packages under
`lib/llvm/lib/cmake/` are a typical casualty, in which case a source
build fails inside `find_package(LLVM CONFIG)` rather than with a
clear missing-file error.

If you installed the SDK from wheels without Developer Mode and a build
cannot find parts of it, enable Developer Mode and reinstall
`rocm-sdk-devel` with `pip install --force-reinstall --no-deps` — the
first run already removed the archive, so a reinstall is what triggers
a second expansion attempt. Installing ROCm from a
[tarball](https://stable.repo.amd.com/rocm/core/tarball/) sidesteps
this altogether, since it ships the tree already unpacked, and is the
more predictable choice for building from source on Windows.

### Experimental libraries

The newly added bindings — `hipfile`, `hipblaslt`, and `hipsparselt` —
are marked **experimental** for one release cycle. What this means in
practice:

- The Python-level API surface is generated automatically from the
  upstream C headers and is functional today, but parameter
  classification (especially **OUT vs INOUT pointer parameters**)
  is heuristic. Some parameters that are currently classified as
  output-only may be re-tuned to in/out (or vice-versa) once we
  collect user feedback. Programs depending on these libraries may
  need minor signature adjustments after a future tuning pass.
- File issues at the hip-python tracker if you find a parameter
  classification that doesn't match the underlying C semantics.
- All other interfaces (return values, opaque handles, scalar types)
  are stable.

> [!IMPORTANT]
> The shared libraries backing `hipfile`, `hipblaslt`, and `hipsparselt`
> may **not be part of a standard ROCm installation**. If `dlopen` of
> `libhipfile.so`, `libhipblaslt.so`, or `libhipsparselt.so` fails on
> your system, you have to build the corresponding library manually
> by following the build instructions in its source package:
>
> - `hipblaslt` — see
>   [rocm-libraries/projects/hipblaslt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt).
> - `hipsparselt` — see
>   [rocm-libraries/projects/hipsparselt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt).
> - `hipfile` — see
>   [rocm-systems/projects/hipfile](https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile).
>
> After building, install the resulting `.so` into a directory on
> `LD_LIBRARY_PATH` (or `${ROCM_PATH}/lib`) so hip-python's loader
> can find it at runtime.

### Not every ROCm library is bound

The code generator emits more bindings than the wheels compile. A
binding ships only when the ROCm installation provides a header the
build can compile against and a shared library the loader can open at
runtime; the build probes for both at configure time and silently drops
the modules that fail. `hsakmt` is a deliberate exception: `/opt/rocm/lib/`
ships only the static archive `libhsakmt.a`, which a `dlopen`-based
runtime cannot load, so the binding is deferred rather than shipped
non-functional. `libhsakmt` is developed alongside the ROCr runtime in
[rocm-systems/projects/rocr-runtime/libhsakmt](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt);
track it there for a shared-library variant.

### `hipblaslt`: Cython-level (`cimport`) usage may require C++ compilation

The Python-level API (`from rocm.bindings import hipblaslt`) works
as expected — at runtime hip-python `dlopen`s `libhipblaslt.so` and
calls C-ABI symbols, which is unaffected by header-source issues.

However, downstream Cython users who do
`cimport rocm.bindings.cyhipblaslt` will cause Cython to emit
`#include <hipblaslt/hipblaslt.h>` in the generated C, and the
upstream header (as of ROCm 10.0.0 / hipBLASLt 1.4.1)
unconditionally pulls in `<memory>`, `<regex>`, `<vector>` (C++
stdlib) even though it is otherwise structured as a pure C-API
header (the C++ extension API lives separately in
`hipblaslt-ext.hpp`). As a result, such extensions must currently
be compiled as C++ (or you must `#define`-shim around the
includes) until the upstream fix lands.

The hip-python codegen itself works around this with an in-memory
strip of the offending lines before parsing; that workaround is not
visible to downstream Cython consumers because it operates only at
generation time. Track this upstream in
[rocm-libraries/projects/hipblaslt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt).

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>
