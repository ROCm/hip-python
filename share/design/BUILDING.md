# Build system design

This document describes the hip-python build system: its layout, the
five packages it produces, the shared CMake helpers, and the supported
build invocations.

## Goals

The build system is designed around three properties:

1. **Five independent wheels, one source tree.** Each of the five Python
   packages can be built and installed on its own, but they share build
   wiring (CMake helpers, generated module lists, namespace markers).
2. **Generator-friendly.** Module counts and source filenames in three
   of the four generator-owned packages can grow between ROCm releases
   without requiring per-package CMake edits — the CMake build reads
   generator-emitted include files (see [CODEGEN.md](CODEGEN.md)).
3. **Cython-source-only generator.** The Python packaging metadata
   (`pyproject.toml`, `_version.py.in`, `__init__.py`, `setup.cfg`) is
   handcoded and never overwritten by the generator.

## The five packages

| Package | Source path | Provides |
|---|---|---|
| `rocm-bindings-util` | `python/rocm-bindings-util/` | DLL loader (`posixloader`/`win32loader` + platform-agnostic `loader`), shared Cython types (`Pointer`, `CStr`, `NDBuffer`, …), and the `paths` module that does lazy ROCm library lookup. **Handcoded; not generator output.** |
| `rocm-bindings-hip` | `python/rocm-bindings-hip/` | `hip` and `hiprtc` bindings (high-level + cy*-prefixed C-level pairs). Helpers (`_hip_helpers`, `_hiprtc_helpers`) are handcoded. |
| `rocm-bindings-libraries` | `python/rocm-bindings-libraries/` | The math/comm/profile libraries: hipblas, hipsolver, rccl, hiprand, hipfft, hipsparse, roctx. List is generator-managed. |
| `rocm-bindings-compiler` | `python/rocm-bindings-compiler/` | LLVM-C bindings, AMD COMGR bindings, optional bundled `libLLVM.so`. Module list is generator-managed. |
| `hip-python-interop` | `python/hip-python-interop/` | CUDA interop layer: `cuda.bindings.{driver,runtime,nvrtc}`. Implemented on top of HIP. |
| `hip-python` | `python/hip-python/` | Provides the `hip.*` namespace as an alias of `rocm.bindings.*` (`from hip import hip, hiprtc, hipblas, …` re-export). Pure Python. |

All five packages contribute to two PEP 420 implicit namespace packages
at runtime: `rocm.bindings.*` and `cuda.bindings.*`. Multiple packages
add modules to the same namespace; only `rocm-bindings-util` ships the
runtime `__init__.pxd` markers for `rocm/` and `rocm/bindings/`.

## Top-level layout

```
hip-python/
├── cmake/
│   ├── HipPythonBuild.cmake       Shared helpers (see below)
│   └── render_version.cmake
├── python/
│   ├── CMakeLists.txt             Unified top-level build; orchestrates all five packages + docs
│   ├── pyproject.toml             Metadata for the `hip-python` (root) wheel target
│   ├── rocm-bindings-util/        per-package source tree + CMakeLists.txt + pyproject.toml
│   ├── rocm-bindings-hip/         …
│   ├── rocm-bindings-libraries/   …
│   ├── rocm-bindings-compiler/    …
│   ├── hip-python-interop/        …
│   └── hip-python/                pure-Python `hip.*` alias of `rocm.bindings.*`
├── docs_src/                      Sphinx source (reStructuredText)
├── docs/                          Generator output: rendered Sphinx HTML (when HIP_PYTHON_BUILD_DOCS=ON)
└── share/design/                  this folder (BUILDING.md, CODEGEN.md)
```

## Two ways to build

### A. Unified build via CMake

```sh
cd python
cmake -B build
cmake --build build --target all_wheels -j$(nproc)
```

This is the canonical build flow. It:

- Configures every enabled package via the unified `python/CMakeLists.txt`.
- Compiles all Cython extensions across all packages.
- Invokes `python -m build --wheel --no-isolation` for each enabled
  package via per-package wheel targets.
- Optionally runs `auditwheel repair` when
  `-DHIP_PYTHON_AUDITWHEEL_REPAIR=ON`.

Wheel artifacts land in `${HIP_PYTHON_WHEEL_OUTPUT_DIR}` (default
`python/build/dist/`).

For a clean rebuild, remove the build directory:

```sh
rm -rf python/build
cmake -S python -B python/build && cmake --build python/build --target all_wheels
```

Common CMake options:

```sh
cd python
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PYTHON_BUILD_UTIL=ON \
  -DHIP_PYTHON_BUILD_HIP=ON \
  -DHIP_PYTHON_BUILD_LIBRARIES=ON \
  -DHIP_PYTHON_BUILD_COMPILER=ON \
  -DHIP_PYTHON_BUILD_INTEROP=ON \
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=dist
cmake --build build --target all_wheels -j$(nproc)
```

You can also build a single package's wheel from the unified build:

```sh
cd python
cmake -B build
cmake --build build --target util_wheel        # rocm-bindings-util only
cmake --build build --target hip_wheel
cmake --build build --target libraries_wheel
cmake --build build --target compiler_wheel
cmake --build build --target interop_wheel
cmake --build build --target hip_python_wheel
```

### B. Single-package build (development loop)

```sh
cd python/rocm-bindings-util
python3 -m build --wheel --no-isolation
```

Each package has its own `pyproject.toml` with `scikit-build-core` as
the build backend pointing at `cmake.source-dir = ".."` (the `python/`
directory). This is the right entry point when iterating on one package
in isolation; it skips `auditwheel repair` and only builds that one
package's targets.

## CMake build orchestration

### Top-level `python/CMakeLists.txt`

Responsibilities:

1. **Per-package opt-in.** `option(HIP_PYTHON_BUILD_<NAME> …)` for each
   of the five packages. Disabling one skips its `add_subdirectory()`
   and its wheel target.

2. **Cross-package include paths.** Builds the
   `HIP_PYTHON_GLOBAL_INCLUDE_DIRS` list from the enabled package source
   directories. Every Cython compilation in the unified build picks
   these up automatically (see `hip_python_add_cython_module` below) so
   `cimport rocm.bindings.cyhip` from a libraries module resolves
   without any per-package include-dir boilerplate.

3. **Wheel targets.** For each enabled package, calls
   `hip_python_add_wheel_target()` which invokes
   `python -m build --wheel --no-isolation` against the package
   directory. The aggregate `all_wheels` target depends on all enabled
   wheel targets.

4. **Optional auditwheel repair.** When `HIP_PYTHON_AUDITWHEEL_REPAIR=ON`,
   each wheel target additionally runs `auditwheel repair` to produce
   `manylinux_*` wheels.

### Shared helpers in `cmake/HipPythonBuild.cmake`

The shared helper module that every per-package `CMakeLists.txt`
includes. Key functions:

#### `hip_python_initialize()`

- Resolves `ROCM_PATH` (from CMake cache, `$ROCM_PATH`, `$ROCM_HOME`, or
  `/opt/rocm`).
- Validates that `${ROCM_PATH}/include` exists and locates the ROCm
  library directory (`lib` or `lib64`).
- Sets `HIP_PLATFORM` (default `amd`).
- Caches `HIP_PYTHON_COMMON_COMPILE_DEFINITIONS` (`__HIP_PLATFORM_AMD__`,
  `__half=uint16_t`) used by every Cython extension.

Each per-package `CMakeLists.txt` calls this once near the top.

#### `hip_python_add_cython_module(...)`

Compiles a single Cython source into a Python extension module and
sets up its install rules. Auto-prepends `${CMAKE_CURRENT_SOURCE_DIR}`
and (when defined) `${HIP_PYTHON_GLOBAL_INCLUDE_DIRS}` to the Cython
and C include paths — per-package `CMakeLists.txt` only needs to pass
package-specific extras like `LLVM_INCLUDE_DIRS`.

```cmake
hip_python_add_cython_module(
  TARGET rocm_bindings_cyhip
  MODULE_NAME rocm.bindings.cyhip
  SOURCE rocm/bindings/cyhip.pyx
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-hip
  LINK_LIBRARIES amdhip64                   # only when not runtime-linking
  DEPENDS ${HIP_CYTHON_DEPENDS}
)
```

#### `hip_python_collect_cython_depends(out_var <patterns>...)`

Globs for `.pxd`/`.pyx` files matching the given patterns and stores the
result in `out_var`. Used to set up `DEPENDS` for incremental rebuilds.

#### `hip_python_define_module_options(...)` / `hip_python_collect_enabled_modules(...)`

Used in `rocm-bindings-libraries` and `rocm-bindings-compiler` to expose
per-module CMake options (e.g. `HIP_PYTHON_ENABLE_LIB_HIPRAND=OFF`)
that disable individual modules at configure time.

#### `hip_python_add_wheel_target(...)`

Creates a CMake custom target that runs `python -m build` against a
single package source directory. Optionally performs `auditwheel repair`
when `HIP_PYTHON_AUDITWHEEL_REPAIR=ON`.

## Generator-managed CMake includes

Two of the four generator-owned packages have module lists that change
between ROCm releases. To keep `CMakeLists.txt` stable across releases,
these packages source the module list from a generator-emitted include
file:

### `python/rocm-bindings-libraries/cmake/generated_modules.cmake`

```cmake
# AUTO-GENERATED by the hip-python code generator. Do not edit by hand.
set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES
    hipblas hipsolver rccl hiprand hipfft hipsparse roctx)
```

The corresponding CMakeLists.txt does:

```cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/generated_modules.cmake")
set(HIP_PYTHON_ALL_LIBRARIES ${HIP_PYTHON_LIBRARIES_GENERATED_MODULES})
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_LIBRARIES)
    hip_python_add_cython_module(... cy${_lib}.pyx ...)
    hip_python_add_cython_module(... ${_lib}.pyx ...)
endforeach()
```

### `python/rocm-bindings-compiler/cmake/generated_modules.cmake`

```cmake
set(HIP_PYTHON_LLVM_C_MODULES disassembler lljit lto debuginfo executionengine ...)
set(HIP_PYTHON_LLVM_C_TRANSFORMS_MODULES passbuilder ...)
set(HIP_PYTHON_LLVM_CONFIG_MODULES llvm_config ...)
set(HIP_PYTHON_COMGR_MODULES amd_comgr cyamd_comgr ...)
```

A default version of each file is checked in so a fresh checkout (with
no generator run) builds. The generator overwrites these on each run.

`rocm-bindings-hip` and `hip-python-interop` keep hand-listed
CMakeLists.txt — their module sets (hip + hiprtc; driver + runtime +
nvrtc) are stable across releases.

## Per-package CMake options

### Common to all packages

| Option | Default | Effect |
|---|---|---|
| `HIP_PYTHON_BUILD_<NAME>` | `ON` | Enable/disable each of the five packages (UTIL, HIP, LIBRARIES, COMPILER, INTEROP). |
| `HIP_PYTHON_BUILD_HIP_PYTHON` | `ON` | Build the legacy `hip-python` metapackage shim. |
| `HIP_PYTHON_AUDITWHEEL_REPAIR` | `OFF` | Run `auditwheel repair` after each wheel build. |
| `HIP_PYTHON_WHEEL_OUTPUT_DIR` | `${CMAKE_BINARY_DIR}/dist` | Where `*.whl` files land. |
| `ROCM_PATH` | `/opt/rocm` (or `$ROCM_PATH`/`$ROCM_HOME`) | ROCm SDK location. |
| `HIP_PLATFORM` | `amd` | Only `amd` and `hcc` are supported. |
| `CMAKE_BUILD_TYPE` | `Release` | Standard CMake build type. |

### `rocm-bindings-hip` and `rocm-bindings-libraries`

| Option | Default | Effect |
|---|---|---|
| `HIP_PYTHON_RUNTIME_LINKING` | `ON` | When ON, generated extensions resolve ROCm shared libraries lazily via `rocm.bindings.util.paths.get_library_path()`. When OFF, they link against `amdhip64` / `hipblas` / etc. at build time. Runtime linking is the default because it keeps wheels portable across slightly different ROCm installs. |
| `HIP_PYTHON_ENABLE_LIB_<NAME>` | `ON` | Per-library opt-in for the libraries package (e.g. `HIP_PYTHON_ENABLE_LIB_HIPRAND=OFF`). |

### `rocm-bindings-compiler`

| Option | Default | Effect |
|---|---|---|
| `HIP_PYTHON_BUNDLE_LIBLLVM` | `ON` | Bundle a working `libLLVM.so` inside the wheel (uses the system one if available; otherwise builds from sources via `python/rocm-bindings-compiler/src/`). |
| `HIP_PYTHON_FORCE_BUILD_LIBLLVM` | `OFF` | Force-build `libLLVM.so` from sources even if a system one is present. Implies BUNDLE. |

## Cython namespace markers and install layout

`__init__.pxd` files live in two tiers:

- **Top-of-namespace** (`rocm/`, `rocm/bindings/`, `cuda/`,
  `cuda/bindings/`): handcoded, committed in every package's source
  tree. Required at build time so cross-package `cimport` resolves
  inside any of the four generator-owned packages.
- **Deeper in the hierarchy** (`rocm/bindings/llvm/`,
  `rocm/bindings/llvm/c/`, …): emitted by the consolidated generator
  alongside the `.pxd`/`.pyx` files. See [CODEGEN.md](CODEGEN.md).

**None** of these are installed except the two from `rocm-bindings-util`:

```cmake
# python/rocm-bindings-util/CMakeLists.txt
install(FILES rocm/__init__.pxd          DESTINATION rocm        COMPONENT rocm-bindings-util)
install(FILES rocm/bindings/__init__.pxd DESTINATION rocm/bindings COMPONENT rocm-bindings-util)
```

The other packages explicitly EXCLUDE `__init__.pxd` from their
recursive `install(DIRECTORY ... PATTERN "*.pxd")` rules:

```cmake
install(
  DIRECTORY rocm/bindings/llvm/c/
  DESTINATION rocm/bindings/llvm/c
  COMPONENT rocm-bindings-compiler
  FILES_MATCHING
    PATTERN "*.pxd"
    PATTERN "__init__.pxd" EXCLUDE
)
```

This keeps the runtime install tree free of duplicate namespace markers
that could conflict between the five wheels.

## Standalone (sdist) build outside the unified tree

When a single package is built from its own `pyproject.toml` (e.g. via
`python -m build` inside its directory), the unified top-level
`CMakeLists.txt` is **not** in scope and `HIP_PYTHON_GLOBAL_INCLUDE_DIRS`
is undefined. The `hip_python_add_cython_module` helper handles this
gracefully — it still auto-prepends `${CMAKE_CURRENT_SOURCE_DIR}`. The
package is responsible for ensuring it can build with only its own
sources and its installed dependencies.

The per-package CMakeLists also includes a "helper resolution" block at
the top that copies `HipPythonBuild.cmake` into the package's `cmake/`
subdirectory when building an sdist (`SKBUILD_STATE STREQUAL "sdist"`),
so the resulting source distribution is self-contained.

## Build requirements

- **Linux** (only platform tested; manylinux wheels target
  `manylinux_2_17_x86_64`).
- **C compiler** (GCC or Clang).
- **Python 3.9+** with `pip>=24.0`, `venv`, and development headers.
- **CMake ≥ 3.26** and **Ninja ≥ 1.11** recommended.
- Python packages: `scikit-build-core>=0.11.2`, `cython>=3.0,<3.1`,
  `build`. Optional for production wheels: `auditwheel`, `patchelf`.
- **ROCm SDK** at `${ROCM_PATH}` (defaults to `/opt/rocm`).
- For docs builds: `sphinx`, `sphinx-autoapi`, `rocm-docs-core`, plus the
  rest of `docs_src/sphinx/requirements.txt`.

## Documentation build

The Sphinx HTML documentation is a separate, optional CMake target that runs
**in parallel to** and **independently of** the wheel build. It is gated on
`HIP_PYTHON_BUILD_DOCS=ON`.

The doc input language is **reStructuredText** (under `docs_src/`), distinct
from the Markdown READMEs and design docs at the repo root. Sphinx parses
Python sources and `.pyi` stubs directly via
[`sphinx-autoapi`](https://sphinx-autoapi.readthedocs.io/), so the docs build
does **not** require the wheels to be built or installed first.

CMake options:

| Option | Default | Effect |
|---|---|---|
| `HIP_PYTHON_BUILD_DOCS` | `OFF` | Gates target creation. When ON, fails fast if `python -m sphinx` is unavailable. |
| `HIP_PYTHON_DOCS_OUTPUT_DIR` | `<repo>/docs` | Destination directory for the rendered HTML. The default keeps the served URL at `docs/index.html`. Override for per-version layouts (e.g. `docs/rocm-rel-7.13.0`) or direct hosting (`/var/www/...`). |
| `HIP_PYTHON_DOCS_DOCTREE_DIR` | `<build>/docs/_doctrees` | Sphinx intermediate cache. Default keeps it inside the build dir. |

Usage:

```sh
cd python
cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target docs
# open ../docs/index.html
```

The `docs` target is **not** part of `all_wheels` — neither depends on the
other. They can run concurrently:

```sh
cmake --build build --target all_wheels docs -j$(nproc)
```

The TOC structure (`docs_src/sphinx/_toc.yml.in`) is hand-maintained with one
subtree per package, so generator-emitted per-module pages slot in cleanly
without TOC edits. Generator-emitted pages live next to handcoded pages
under `docs_src/python_api/` and `docs_src/python_api_manual/`; the
`.gitignore` in the former ignores generator output by default while
force-tracking the handcoded `rocm.bindings.util.rst` and `hip.rst`.

See [CODEGEN.md](CODEGEN.md) for the full list of generator-owned
documentation files.

## Common build invocations

```sh
# Fastest dev iteration on a single package:
cd python/rocm-bindings-util && python3 -m build --wheel --no-isolation

# Full build, all five packages (run from python/ subdir):
cd python && cmake -B build && cmake --build build --target all_wheels -j$(nproc)

# Production manylinux wheels:
cd python && cmake -B build -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON
cmake --build build --target all_wheels -j$(nproc)

# Skip the compiler package (faster; doesn't need libLLVM):
cd python && cmake -B build -DHIP_PYTHON_BUILD_COMPILER=OFF
cmake --build build --target all_wheels

# Debug build:
cd python && cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target all_wheels

# Use sccache:
cd python && cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake --build build --target all_wheels

# Custom ROCm path:
cd python && cmake -B build -DROCM_PATH=/opt/rocm-7.13
cmake --build build --target all_wheels

# Single package wheel from the unified build:
cd python && cmake -B build && cmake --build build --target util_wheel

# Build the documentation (independent of all_wheels):
cd python && cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target docs

# Wheels and docs in parallel:
cd python && cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target all_wheels docs -j$(nproc)
```

## See also

- [CODEGEN.md](CODEGEN.md) — how interfacegen produces the generated Cython sources
- Project [README](../../README.md) — install + quick-start
