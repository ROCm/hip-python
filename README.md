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

* Currently, only Linux is supported (prebuilt packages and code).
  * Prebuilt packages distributed via PyPI are only provided for
    Linux systems that agree with the `manylinux_2_17_x86_64` tag.
* Requires that a compatible ROCm&trade; HIP SDK is installed on your system.
  * Source code is provided only for particular ROCm versions.
    * See the `git` branches tagged with `release/rocm-rel-X.Y[.Z]`
  * Prebuilt packages are built only for particular ROCm versions.

> [!NOTE]
> You may find that packages for one ROCm&trade; release are compatible with
> the ROCm&trade; HIP SDK of another release as the HIP Python functions load
> HIP C functions in a lazy manner.

### Build requirements

* A Linux operating system
* A C compiler
* `bash`, `python3` + `venv`
* The ROCm&trade; HIP SDK
* Python 3.8+.

## Install Prebuilt Packages

***

> [!IMPORTANT]
> Ensure that `pip` has at least version `24.0`, please upgrade it otherwise.

***

### Via PyPI

First identify the first three digits of the version number of your
ROCm&trade; installation. Then install the HIP Python package(s) as follows:

<!-- markdownlint-disable  MD013 -->

```shell
# Install the rocm-bindings-* packages (HIP, libraries, compiler) plus
# the hip-python alias:
python3 -m pip install hip-python~=$rocm_version.0
# Install the CUDA Python interoperability package too:
python3 -m pip install hip-python-interop~=$rocm_version.0
```

<!-- markdownlint-enable  MD013 -->

### Via Wheel in Local Filesystem

If you have HIP Python package wheels on your filesystem, you can run:

```shell
python3 -m pip install $path_to_hip_python.whl
# if you want the CUDA Python interoperability package too, run:
python3 -m pip install $path_to_hip_python_interop.whl
```

> [!NOTE]
> See the HIP Python user guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Build from Source

The build uses CMake with `scikit-build-core` as the Python build backend.
It produces five wheels:

- `rocm-bindings-core` — DLL loader, types, ROCm path resolution
- `rocm-bindings-hip` — `hip` and `hiprtc` Python bindings
- `rocm-bindings-libraries` — `hipblas`, `hipsolver`, `rccl`, `hiprand`,
  `hipfft`, `hipsparse`, `roctx`
- `rocm-bindings-compiler` — LLVM-C and AMD COMGR bindings (with optional bundled `libLLVM.so`)
- `hip-python-interop` — `cuda.bindings.{driver,runtime,nvrtc}` interop layer

Plus a `hip-python` package that exposes the `hip.*` namespace as an
alias of `rocm.bindings.*`, so that `from hip import hip, hiprtc, hipblas`
(etc.) keeps working unchanged.

> [!NOTE]
> The HIP-side `from hip import hip, hiprtc` (and friends) are aliases
> of the per-package `rocm.bindings.*` modules — both styles are
> supported. The CUDA interop side has **no such alias package**: only
> `from cuda.bindings import driver, runtime, nvrtc` works (`cuda.bindings`
> is the package, `driver` / `runtime` / `nvrtc` are the modules).
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
   pip install build "scikit-build-core>=0.11.2" "cmake>=3.26" "ninja>=1.11" "cython>=3.0,<3.1"
   ```

4. Configure and build all wheels:

   ```shell
   cd python
   cmake -B build
   cmake --build build --target all_wheels -j$(nproc)
   ```

   Wheels for all five packages plus the `hip-python` metapackage land in
   `python/build/dist/`.

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
cd python && cmake -B build && cd ..

# Build just rocm-bindings-core:
cd python/rocm-bindings-core
python3 -m build --wheel --no-isolation

# Build just rocm-bindings-hip:
cd python/rocm-bindings-hip
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
cd python
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
cd python
cmake -B build \
  -DHIP_PYTHON_BUILD_COMPILER=OFF \
  -DHIP_PYTHON_BUILD_INTEROP=OFF
cmake --build build --target all_wheels
```

You can also build a single package's wheel from the unified build:

```shell
cd python
cmake -B build
cmake --build build --target core_wheel        # rocm-bindings-core only
cmake --build build --target hip_wheel         # rocm-bindings-hip only
cmake --build build --target libraries_wheel   # rocm-bindings-libraries only
cmake --build build --target compiler_wheel    # rocm-bindings-compiler only
cmake --build build --target interop_wheel     # hip-python-interop only
cmake --build build --target hip_python_wheel  # hip-python metapackage only
```

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
rm -rf python/build && cmake -S python -B python/build && cmake --build python/build --target all_wheels
```

For deeper documentation:

- [share/design/BUILDING.md](share/design/BUILDING.md) — Build system architecture, helper functions, package layout
- [share/design/CODEGEN.md](share/design/CODEGEN.md) — How the interfacegen code generator interacts with the hip-python source tree to produce a release
- [BUILD.md](BUILD.md) — CMake target reference and historical CMake-specific notes

> [!NOTE]
> See the HIP Python developer guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

### Build Configuration Options

<!-- markdownlint-disable  MD013 -->

| Option | Default | Effect |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` (or `$ROCM_PATH`/`$ROCM_HOME`) | Path to the ROCm installation. |
| `HIP_PLATFORM` | `amd` | HIP backend selector. Only `amd` and `hcc` are supported. |
| `HIP_PYTHON_BUILD_<NAME>` | `ON` | Per-package opt-in: `UTIL`, `HIP`, `LIBRARIES`, `COMPILER`, `INTEROP`, `HIP_PYTHON`. |
| `HIP_PYTHON_RUNTIME_LINKING` | `ON` | When `ON`, generated extensions resolve ROCm shared libraries lazily at runtime; when `OFF`, they link against them at build time. |
| `HIP_PYTHON_ENABLE_LIB_<NAME>` | `ON` | Per-library toggle inside `rocm-bindings-libraries` (e.g. `HIP_PYTHON_ENABLE_LIB_HIPRAND=OFF`). |
| `HIP_PYTHON_BUNDLE_LIBLLVM` | `ON` | Bundle `libLLVM.so` inside the `rocm-bindings-compiler` wheel. |
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
cd python
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

## Legacy Build from Source

These are the original script-based build instructions for older releases up to
and including ROCm 7.2.2.

1. Install ROCm.
1. Install `pip`, virtual environment and development headers for Python 3:

   ```shell
   # Ubuntu:
   sudo apt install python3-pip python3-venv python3-dev
   ```

1. Check out the feature branch `release/rocm-rel-X.Y[.Z]` for your particular
   ROCm installation.
1. Initialize the branch:

   ```shell
   ./init.sh
   ```

1. Build the packages:

   ```shell
   ./build_hip_python_pkgs.sh --hip --cuda --post-clean
   ```

The legacy build process produces Python binary wheels in `hip-python/dist/`
and `hip-python-as-cuda/dist/`.

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>
