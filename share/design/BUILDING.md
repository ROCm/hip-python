# Build system design

This document describes the hip-python build system: its layout, the
eight packages it produces, the shared CMake helpers, and the supported
build invocations.

## Goals

The build system is designed around three properties:

1. **Eight independent wheels, one source tree.** Each of the eight
   Python packages can be built and installed on its own, but they share
   build wiring (CMake helpers, generated module lists, namespace
   markers). Six of the eight are compiled (Cython extensions); the
   other two, `hip-python` and `numba-hip`, are pure Python.
2. **Generator-friendly.** Module counts and source filenames in three
   of the five generator-owned packages (`rocm-bindings-libraries`,
   `rocm-bindings-systems`, `rocm-bindings-compiler`) can grow between
   ROCm releases without requiring per-package CMake edits — the CMake
   build reads generator-emitted include files (see
   [CODEGEN.md](CODEGEN.md)).
3. **Cython-source-only generator.** The Python packaging metadata
   (`pyproject.toml`, `__init__.py`, `setup.cfg`) is handcoded and never
   overwritten by the generator. (The one rendered exception is
   `rocm-bindings-core/src/rocm/version.py`, produced from the handcoded
   `version.py.in` template — see [CODEGEN.md](CODEGEN.md).)

## The eight packages

| Package | Source path | Provides |
|---|---|---|
| `rocm-bindings-core` | `packages/rocm-bindings-core/` | DLL loader (`posixloader`/`win32loader` + platform-agnostic `loader`), shared Cython types (`Pointer`, `CStr`, `NDBuffer`, …), and the `paths` module that does lazy ROCm library lookup. **Handcoded; not generator output.** |
| `rocm-bindings-hip` | `packages/rocm-bindings-hip/` | `hip` and `hiprtc` bindings (high-level + cy*-prefixed C-level pairs). Helpers (`_hip_helpers`, `_hiprtc_helpers`) are handcoded. |
| `rocm-bindings-libraries` | `packages/rocm-bindings-libraries/` | Math/FFT/random/sparse libraries: hipblas, hipsolver, hiprand, hipfft, hipsparse. List is generator-managed. |
| `rocm-bindings-systems` | `packages/rocm-bindings-systems/` | System-level libraries: rccl (collective communication), roctx (profiling/tracing), hipfile (accelerated file I/O), amdsmi (system management interface). Optional bundled `libhipfile.so`. List is generator-managed. |
| `rocm-bindings-compiler` | `packages/rocm-bindings-compiler/` | LLVM-C bindings, AMD COMGR bindings, optional bundled `libLLVM.so`. Module list is generator-managed. |
| `hip-python-interop` | `packages/hip-python-interop/` | CUDA interop layer: `cuda.bindings.{driver,runtime,nvrtc}`. Implemented on top of HIP. |
| `hip-python` | `packages/hip-python/` | Provides the `hip.*` namespace as an alias of `rocm.bindings.*` (`from hip import hip, hiprtc, hipblas, …` re-export). Pure Python. |
| `numba-hip` | `packages/numba-hip/` | The ROCm HIP backend for Numba (`numba.hip`). Pure Python, versioned independently of the binding wheels; built unless `HIP_PYTHON_BUILD_NUMBA_HIP=OFF`. |

The binding packages contribute to two PEP 420 implicit namespace packages
at runtime: `rocm.bindings.*` and `cuda.bindings.*`. Multiple packages
add modules to the same namespace; only `rocm-bindings-core` ships the
runtime `__init__.pxd` markers — for `rocm/`, `rocm/bindings/`, and
`rocm/bindings/util/` (the `util` marker lets other packages'
`cimport rocm.bindings.util.{types,loader}` resolve against installed core).

## Top-level layout

```
hip-python/
├── cmake/
│   ├── HipPythonBuild.cmake       Shared helpers (see below)
│   └── HipPythonCodegen.cmake     Configure-time HIP Python code generation
├── packages/
│   ├── CMakeLists.txt             Unified top-level build; orchestrates all eight packages + docs
│   ├── rocm-bindings-core/        per-package: pyproject.toml + CMakeLists.txt + cmake/ + src/rocm/
│   ├── rocm-bindings-hip/         …
│   ├── rocm-bindings-libraries/   …
│   ├── rocm-bindings-systems/     …  + bundled/libhipfile/ (optional libhipfile.so bundling)
│   ├── rocm-bindings-compiler/    …  + bundled/libllvm/ (libLLVM detection + optional bundling)
│   ├── hip-python-interop/        …  src/cuda/ instead of src/rocm/
│   ├── hip-python/                pure-Python `hip.*` alias of `rocm.bindings.*` (src/hip/)
│   └── numba-hip/                 pure-Python Numba HIP backend (src/numba/hip/)
├── docs_src/                      Sphinx source (reStructuredText)
├── docs/                          Generator output: rendered Sphinx HTML (when HIP_PYTHON_BUILD_DOCS=ON)
└── share/design/                  this folder (BUILDING.md, CODEGEN.md, BINDINGS.md)
```

## Per-package layout (PyPA src-layout)

Every wheel has the same three top-level subdirectories:

| Subdir | Purpose |
|---|---|
| `src/<top-pkg>/...` | Python sources (`.pxd`/`.pyx`/`.pyi`/`.py`). `<top-pkg>` is `rocm` for the binding wheels and `cuda` for `hip-python-interop`. |
| `cmake/HipPythonBuild.cmake` | Shared CMake helper, mirrored from the repo-root `cmake/` at configure time. |
| `bundled/<libname>/CMakeLists.txt` | Optional. Build/detect a vendored shared library and copy it into the wheel. See section below. |

Plus per-wheel `pyproject.toml`, `CMakeLists.txt`, `VERSION`, `LICENSE`, `README.md`.

The `src/` layout is a PyPA convention (not a PEP). It keeps the importable
package isolated from build artifacts and tooling so test runs against the
installed wheel can't accidentally pick up the in-tree source.

## Vendored libraries: `bundled/<libname>/`

Some wheels redistribute or wrap a system shared library that is built /
detected at wheel-build time and copied into the wheel for self-contained
installs. Two examples currently:

- `packages/rocm-bindings-compiler/bundled/libllvm/` — builds
  `librocmllvm.so` from LLVM static archives via `--whole-archive`, OR
  copies a system `libLLVM.so` from the ROCm install. The cython modules
  in this wheel get an `$ORIGIN/..` RPATH so they find the bundled lib.
  On Windows ROCm ships no shared LLVM at all, so the library is always
  linked from the archives — with an export list rather than
  `/WHOLEARCHIVE`, since PE images export only what they are told to (see
  the comments in that `CMakeLists.txt` and `gen_msvc_exports.py`).
- `packages/rocm-bindings-systems/bundled/libhipfile/` — when
  `HIP_PYTHON_BUNDLE_LIBHIPFILE=ON`, copies the resolved `libhipfile.so`
  into the wheel and sets `$ORIGIN` RPATH on the cython modules.

### Convention

Every vendored library lives in its own `bundled/<libname>/` subdirectory
of the wheel that ships it. The subdirectory contains at minimum a
`CMakeLists.txt` that:

1. Resolves or builds the shared library at configure time.
2. `install(...)` it into the wheel's runtime directory (typically
   `rocm/bindings/`).
3. Sets `INSTALL_RPATH` on the relevant cython module targets so they
   find the bundled `.so` via `$ORIGIN`-relative lookup.

### Why a separate subdirectory

- **Symmetric with `src/`.** Each wheel has at most three top-level subdirs
  (`src/`, `cmake/`, `bundled/`); each has one job. No directory is a
  grab-bag.
- **Discoverable.** A new contributor opening `packages/<wheel>/` can
  immediately tell what gets shipped and where it comes from.
- **Per-library isolation.** Bundling logic for one vendored lib doesn't
  pollute the parent CMakeLists.txt. Adding another vendored lib later is
  just another `bundled/<libname>/` directory + an `add_subdirectory()`
  call.

### Parent–child cmake contract

The parent `packages/<wheel>/CMakeLists.txt`:
- Calls `find_package(<lib> [QUIET])` so the variables/targets are in scope
- Declares the `option(HIP_PYTHON_BUNDLE_LIB<NAME> …)` flag
- Creates the cython module targets in its main foreach loop
- Calls `add_subdirectory(bundled/<libname>)` **after** the foreach loop
  (the subdirectory's CMakeLists.txt sets RPATH on the cython targets,
  so they must already exist)

The bundled `CMakeLists.txt`:
- Reads only the inputs documented at the top (find_package outputs +
  the `BUNDLE_LIB<NAME>` option)
- Does not call `find_package` itself (avoids redundant detection)
- Is a no-op when bundling is disabled

## Two ways to build

### A. Unified build via CMake

```sh
cd packages
cmake -B build
cmake --build build --target all_wheels -j$(nproc)
```

This is the canonical build flow. It:

- Configures every enabled package via the unified `packages/CMakeLists.txt`.
- Compiles all Cython extensions across all packages **once**, in the
  unified build tree.
- Assembles each of the six compiled wheels from that already-compiled
  output (see [Wheel assembly: compile once, then collect](#wheel-assembly-compile-once-then-collect)
  below) rather than recompiling. The pure-Python `hip-python`
  metapackage is still built with `python -m build`.
- Optionally runs `auditwheel repair` when
  `-DHIP_PYTHON_AUDITWHEEL_REPAIR=ON`.

Wheel artifacts land in `${HIP_PYTHON_WHEEL_OUTPUT_DIR}` (default
`packages/build/dist/`).

For a clean rebuild, remove the build directory:

```sh
rm -rf packages/build
cmake -S packages -B packages/build && cmake --build packages/build --target all_wheels
```

Common CMake options:

```sh
cd packages
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PYTHON_BUILD_CORE=ON \
  -DHIP_PYTHON_BUILD_HIP=ON \
  -DHIP_PYTHON_BUILD_LIBRARIES=ON \
  -DHIP_PYTHON_BUILD_SYSTEMS=ON \
  -DHIP_PYTHON_BUILD_COMPILER=ON \
  -DHIP_PYTHON_BUILD_INTEROP=ON \
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=dist
cmake --build build --target all_wheels -j$(nproc)
```

You can also build a single package's wheel from the unified build:

```sh
cd packages
cmake -B build
cmake --build build --target core_wheel        # rocm-bindings-core only
cmake --build build --target hip_wheel
cmake --build build --target libraries_wheel
cmake --build build --target systems_wheel
cmake --build build --target compiler_wheel
cmake --build build --target interop_wheel
cmake --build build --target hip_python_wheel
cmake --build build --target numba_hip_wheel   # pure Python, own version
```

### Optional configure-time code generation

The Cython sources and per-package `cmake/generated_modules.cmake` lists
are normally consumed pre-generated from the source tree. To regenerate
them as part of the build, enable codegen at configure time:

```sh
cmake -S packages -B build \
  -DHIP_PYTHON_RUN_CODEGEN=ON \
  -DHIP_PYTHON_ROCM_PATH=/opt/rocm \
  -DHIP_PYTHON_ROCM_VERSION=7.13.0
cmake --build build --target all_wheels
```

This runs the `hip-python-generate` tool (install it first:
`pip install tools/hip-python-generate`) **during the configure step**,
before any build target is created — so a single configure + single
build picks up the freshly generated sources even when the module set
changes. Because it runs at configure time and libclang parses every
ROCm header, it is **SLOW**: `cmake -S packages -B build` blocks for
several minutes up to ~30 min depending on core count. A SHA256 stamp
guard (over the ROCm version/path and the optional source-dir inputs)
skips regeneration on no-op reconfigures; pass
`-DHIP_PYTHON_FORCE_CODEGEN=ON` to force a re-run. Optional header-source
overrides: `HIP_PYTHON_ROCM_SYSTEMS_DIR`, `HIP_PYTHON_ROCM_LIBRARIES_DIR`,
`HIP_PYTHON_ROCM_LLVM_PROJECT_DIR`, `HIP_PYTHON_CLANG_RESOURCE_DIR`,
`HIP_PYTHON_CODEGEN_INCLUDE`. Against an older ROCm,
`-DHIP_PYTHON_CODEGEN_ALLOW_MISSING_HEADERS=ON` turns a library whose
header that release never shipped into a skip: the codegen leaves it out
and the package loops build only what was generated.
`-DHIP_PYTHON_CODEGEN_SKIP_LIBRARIES=hiptensor,hipdnn_backend` leaves
named libraries out the same way, whatever the reason. See
[CODEGEN.md](CODEGEN.md).

numba-hip (`numba_hip_wheel` / `numba_hip_sdist`) is a pure-Python
package with its own independent version (not mirrored from the repo-root
`VERSION`); it is part of `all_wheels`/`all_sdists` and gated by
`HIP_PYTHON_BUILD_NUMBA_HIP`. Its test suite lives in the repo-root
`tests/numba-hip/` (outside any importable package, so it exercises the
*installed* `numba.hip`), and is run by the unified `ci/internal/test.sh`
and `ci/internal/test.ps1` alongside the hip-python example suite —
numba-hip no longer carries its own `ci/` scripts.

The unified build is shell-free outside `ci/`: wheel/sdist artifacts are
copied with `cmake -E copy_directory` (no `cp`/`sh` glob), so the build
works on Windows. The only remaining `sh -c` is the optional
`auditwheel repair`, which is a Linux-only ELF retagger and is gated on
`NOT WIN32` (Windows wheels already carry the correct `win_amd64` tag and
are copied as-is).

#### Wheel assembly: compile once, then collect

The unified build compiles every Cython extension once. Naively
building each wheel with `python -m build` would compile every
extension a **second** time, because `python -m build` re-invokes
scikit-build-core, which runs its own CMake configure+build in a
*separate* build tree.

That second compile is unavoidable as long as scikit-build-core owns
the build, and the two trees cannot share artifacts:

- **A CMake build directory is pinned to one source root.** Its
  `CMakeCache.txt` records `CMAKE_HOME_DIRECTORY`. The unified build's
  root is `packages/`; each per-package scikit-build configure roots at
  the package directory (`cmake.source-dir = "."`). Two configures with
  different roots cannot share one build directory.
- **ninja/make incrementality is per-tree.** A target is skipped only
  when its output already exists *inside that build tree* and is newer
  than its inputs, tracked in that tree's own dependency log. A `.so`
  compiled in the unified tree is invisible to scikit-build-core's tree.

So the unified flow does not call `python -m build` for the compiled
packages. Instead, `hip_python_add_wheel_target` runs
[`cmake/hip_python_assemble_wheel.py`](../../cmake/hip_python_assemble_wheel.py),
which reproduces exactly what scikit-build-core would have packed:

1. **`cmake --install <unified-build> --component <pkg> --prefix
   <staging>`** — copies the package's compiled `.so` modules and its
   explicitly `install()`-ed files (`.pxd`, `.pyi`, pure `.py`)
   into a staging dir, applying each target's
   `INSTALL_RPATH` (so bundled-lib lookups like the compiler package's
   `$ORIGIN/..` libLLVM keep working). This is the `install.components`
   half of a scikit-build-core wheel.
2. **Source overlay** — copies the `wheel.packages` source subtree
   (`src/rocm` -> `rocm/`, or `src/cuda/` for interop) on top, adding
   the source-only files (`.pyx`, `.pyi` stubs, namespace markers,
   pure-Python modules) that are not produced by the CMake install.
   Installed files win on the (byte-identical) overlap.
3. **`.dist-info` metadata** — `METADATA` is generated with
   `pyproject-metadata` from the package's `pyproject.toml` (the same
   PEP 621 -> core-metadata path scikit-build-core uses), with the
   dynamic version filled in from the per-package `VERSION` file.
   `WHEEL` and `RECORD` are written directly.
4. **Pack** — the staging tree is zipped into a wheel carrying a generic
   `linux_<arch>` platform tag.

The auditwheel/copy/stamp tail of `hip_python_add_wheel_target` is
unchanged: when `HIP_PYTHON_AUDITWHEEL_REPAIR=ON`, `auditwheel repair`
retags the assembled wheel to its `manylinux_*` aliases exactly as it
did for the `python -m build` output.

This affects **only** the unified `all_wheels` flow. The single-package
(section B) and sdist (section C) builds still invoke scikit-build-core
directly and compile normally — they have no unified tree to collect
from. The assembler therefore needs `pyproject-metadata` in the build
environment (see [Build requirements](#build-requirements)).

### B. Single-package build (development loop)

```sh
cd packages
cmake -B build                                # one-time configure step
cd rocm-bindings-core
python3 -m build --wheel --no-isolation
```

Each per-package `pyproject.toml` points scikit-build-core at
`cmake.source-dir = "."` (the package's own `CMakeLists.txt`) and
`metadata.version.input = "VERSION"` (a per-package file populated by
the unified configure step). This makes every package self-sufficient:

- For wheel/sdist builds, the per-package `CMakeLists.txt` is the
  CMake root.
- The version file `packages/<pkg>/VERSION` is a copy of the canonical
  repo-root `VERSION`; the unified `cmake -B build` step does the
  copy at configure time. Each per-package `VERSION` is **gitignored**
  (single source of truth lives at the repo root).
- The shared cmake helper `cmake/HipPythonBuild.cmake` is similarly
  mirrored into each `packages/<pkg>/cmake/` at configure time, so
  every package finds it locally without a `../../cmake/...` lookup.
  Also gitignored.

If you skip the unified configure and try
`python -m build --wheel` from a per-package directory directly,
scikit-build-core's metadata provider will fail with
`FileNotFoundError: VERSION` — the fix is to run the unified
configure once to populate the per-package files.

### C. sdist build + install (offline, distribution-friendly)

The unified build exposes per-package sdist targets in addition to
the wheel targets:

```sh
cd packages
cmake -B build
cmake --build build --target all_sdists       # all enabled packages
# or one at a time:
cmake --build build --target core_sdist
cmake --build build --target compiler_sdist
```

Each `<pkg>_sdist` target runs `python -m build --sdist
--no-isolation` from the package directory and drops the resulting
tarball into `${HIP_PYTHON_WHEEL_OUTPUT_DIR}` (default
`packages/build/dist/`). The same per-package opt-in
(`HIP_PYTHON_BUILD_<NAME>`) controls both wheel and sdist targets.

To install a sdist downstream:

```sh
pip install --no-build-isolation \
    packages/build/dist/rocm_bindings_compiler-*.tar.gz
```

The sdist tarball bundles `VERSION`, the shared cmake helper
(`cmake/HipPythonBuild.cmake`), every `.pxd`/`.pyx`/`.pyi`/`.py`
source, the per-package `CMakeLists.txt`, and (for
`rocm-bindings-compiler`) the `src/` subtree that builds the bundled
`libLLVM.so`. Installing the sdist re-runs CMake against the
package's own self-contained `CMakeLists.txt`, compiles the Cython
extensions, and (for the compiler package) builds or copies
`libLLVM.so` into the resulting wheel — exactly what a
`python -m build --wheel` from the source tree would produce.

## CMake build orchestration

### Top-level `packages/CMakeLists.txt`

Responsibilities:

1. **Per-package opt-in.** `option(HIP_PYTHON_BUILD_<NAME> …)` for each
   of the six compiled packages, plus `HIP_PYTHON_BUILD_HIP_PYTHON` for
   the pure-Python metapackage. Disabling one skips its
   `add_subdirectory()` and its wheel target.

2. **Cross-package include paths.** Builds the
   `HIP_PYTHON_GLOBAL_INCLUDE_DIRS` list from the enabled package source
   directories. Every Cython compilation in the unified build picks
   these up automatically (see `hip_python_add_cython_module` below) so
   `cimport rocm.bindings.cyhip` from a libraries module resolves
   without any per-package include-dir boilerplate.

3. **Wheel targets.** For each enabled package, calls
   `hip_python_add_wheel_target()`. For the six compiled packages this
   assembles the wheel from the unified build's compiled output (see
   [Wheel assembly](#wheel-assembly-compile-once-then-collect)); for the
   pure-Python `hip-python` metapackage it invokes
   `python -m build --wheel --no-isolation`. The aggregate `all_wheels`
   target depends on all enabled wheel targets.

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

Creates a CMake custom target that produces a single package's wheel
into `${HIP_PYTHON_WHEEL_OUTPUT_DIR}`.

- When called with `COMPONENT <name>` (the six compiled packages), it
  runs [`cmake/hip_python_assemble_wheel.py`](../../cmake/hip_python_assemble_wheel.py)
  to assemble the wheel from the unified build's already-compiled
  output: `cmake --install` of that component into a staging dir, the
  `wheel.packages` source overlay, `pyproject-metadata`-generated
  `.dist-info`, then pack. No recompile. See
  [Wheel assembly](#wheel-assembly-compile-once-then-collect).
- Without `COMPONENT` (the pure-Python `hip-python` metapackage), it
  runs `python -m build --wheel --no-isolation` against the package
  directory.

In both cases, when `HIP_PYTHON_AUDITWHEEL_REPAIR=ON` the target then
runs `auditwheel repair` to retag/repair the wheel. The `DEPENDS`
argument gates the target on the package's compile aggregate target
(`package_rocm_bindings_<pkg>`) so the `.so` files exist before
assembly.

## Generator-managed CMake includes

Three of the five generator-owned packages have module lists that change
between ROCm releases. To keep `CMakeLists.txt` stable across releases,
these packages source the module list from a generator-emitted include
file:

### `packages/rocm-bindings-libraries/cmake/generated_modules.cmake`

```cmake
# AUTO-GENERATED by the hip-python code generator. Do not edit by hand.
set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES
    hipblas hipsolver hiprand hipfft hipsparse)
```

### `packages/rocm-bindings-systems/cmake/generated_modules.cmake`

```cmake
# AUTO-GENERATED by the hip-python code generator. Do not edit by hand.
set(HIP_PYTHON_SYSTEMS_GENERATED_MODULES
    rccl roctx hipfile amdsmi hsa)
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

### `packages/rocm-bindings-compiler/cmake/generated_modules.cmake`

```cmake
set(HIP_PYTHON_LLVM_C_MODULES disassembler lljit lto debuginfo executionengine ...)
set(HIP_PYTHON_LLVM_C_TRANSFORMS_MODULES passbuilder ...)
set(HIP_PYTHON_LLVM_CONFIG_MODULES llvm_config ...)
set(HIP_PYTHON_COMGR_MODULES amd_comgr cyamd_comgr ...)
```

These `generated_modules.cmake` files are generator outputs: like the
`.pxd`/`.pyx` sources they are **absent on the codegen base branch** and
written by the generator (committed on release branches). A configure on
the base branch with no generator run therefore only builds the packages
whose module lists are hand-listed (see below); the math/systems/compiler
module lists come from the generator. See [CODEGEN.md](CODEGEN.md).

`rocm-bindings-hip` and `hip-python-interop` keep hand-listed
CMakeLists.txt — their module sets (hip + hiprtc; driver + runtime +
nvrtc) are stable across releases.

## Per-package CMake options

### Common to all packages

| Option | Default | Effect |
|---|---|---|
| `HIP_PYTHON_BUILD_<NAME>` | `ON` | Enable/disable each of the six compiled packages (CORE, HIP, LIBRARIES, SYSTEMS, COMPILER, INTEROP). |
| `HIP_PYTHON_BUILD_HIP_PYTHON` | `ON` | Build the legacy `hip-python` metapackage shim. |
| `HIP_PYTHON_BUILD_NUMBA_HIP` | `ON` | Build the pure-Python `numba-hip` wheel/sdist (own independent version). |
| `HIP_PYTHON_AUDITWHEEL_REPAIR` | `OFF` | Run `auditwheel repair` after each wheel build (Linux only; ignored on Windows). |
| `HIP_PYTHON_RUN_CODEGEN` | `OFF` | Run `hip-python-generate` at configure time (SLOW; blocks configure). |
| `HIP_PYTHON_ROCM_VERSION` | _(empty)_ | ROCm version for codegen (`--rocm-version`); required when `HIP_PYTHON_RUN_CODEGEN=ON`. |
| `HIP_PYTHON_ROCM_PATH` | `/opt/rocm` (or `$ROCM_PATH`/`$ROCM_HOME`) | ROCm install passed to codegen (`--rocm-path`). |
| `HIP_PYTHON_FORCE_CODEGEN` | `OFF` | Bypass the codegen stamp guard and force regeneration. |
| `HIP_PYTHON_CODEGEN_ALLOW_MISSING_HEADERS` | `OFF` | Skip a library whose header the ROCm at hand does not carry (`--allow-missing-headers`) and drop it from the build, rather than failing. For building against an older ROCm. |
| `HIP_PYTHON_CODEGEN_SKIP_LIBRARIES` | _(empty)_ | Comma-separated libraries not to generate, by name (`--skip-libraries`), e.g. `hiptensor,hipdnn_backend`. Dropped from the build the same way a missing header is. An unknown name fails the codegen. |
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
| `HIP_PYTHON_BUNDLE_LIBLLVM` | `ON` | Bundle a working `libLLVM.so` inside the wheel (uses the system one if available; otherwise builds from sources via `packages/rocm-bindings-compiler/src/`). Windows has no system one to reuse, so bundling there means a ~75 MB `LLVM.dll` linked from the static archives. Turn it off to save that space, at the cost of the `rocm.bindings.llvm.*` bindings, which then import but raise on first use, and of `numba.hip`, which is built on them. |
| `HIP_PYTHON_FORCE_BUILD_LIBLLVM` | `OFF` | Force-build `libLLVM.so` from sources even if a system one is present. Implies BUNDLE. |

## Cython namespace markers and install layout

`__init__.pxd` files live in two tiers:

- **Top-of-namespace** (`rocm/`, `rocm/bindings/`, `cuda/`,
  `cuda/bindings/`): handcoded, committed in every package's source
  tree. Required at build time so cross-package `cimport` resolves
  inside any of the generator-owned packages.
- **Deeper in the hierarchy** (`rocm/bindings/llvm/`,
  `rocm/bindings/llvm/c/`, …): emitted by the consolidated generator
  alongside the `.pxd`/`.pyx` files. See [CODEGEN.md](CODEGEN.md).

**None** of these are installed except the three from `rocm-bindings-core`:

```cmake
# packages/rocm-bindings-core/CMakeLists.txt
install(FILES rocm/__init__.pxd               DESTINATION rocm              COMPONENT rocm-bindings-core)
install(FILES rocm/bindings/__init__.pxd      DESTINATION rocm/bindings      COMPONENT rocm-bindings-core)
install(FILES rocm/bindings/util/__init__.pxd DESTINATION rocm/bindings/util COMPONENT rocm-bindings-core)
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
that could conflict between the five `rocm-bindings-*` wheels that share
the `rocm/bindings` namespace.

## Standalone (sdist) build outside the unified tree

When a single package is built from its own `pyproject.toml` (e.g. via
`python -m build` inside its directory), the unified top-level
`CMakeLists.txt` is **not** in scope and `HIP_PYTHON_GLOBAL_INCLUDE_DIRS`
is undefined. The `hip_python_add_cython_module` helper handles this
gracefully — it still auto-prepends `${CMAKE_CURRENT_SOURCE_DIR}`. The
package is responsible for ensuring it can build with only its own
sources and its installed dependencies.

### How the version file reaches scikit-build's metadata provider

`metadata.version.input = "VERSION"` (per-package) is read by
scikit-build-core **before** CMake runs. So `VERSION` must already
exist locally.

The canonical version is **hardcoded**, not derived from git history:
the repo-root `HIP_PYTHON_VERSION` file holds the hip-python version
(starts at `0.0.1`). At configure time the unified `cmake -B build`
renders the repo-root `VERSION` from it:

- **Release branches** commit a `VERSION.in` template that embeds the
  ROCm version, e.g. `7.13.0.@HIP_PYTHON_VERSION@`; CMake configures it
  (`@ONLY`) with `HIP_PYTHON_VERSION` substituted in.
- **Development branches** have no `VERSION.in`, so `VERSION` is the
  `HIP_PYTHON_VERSION` value verbatim.

`ci/internal/build-wheels.sh` narrows that second case: when the branch
has no `VERSION.in` and `ROCM_VERSION` is set, it authors one into its
working copy through the same `ci/internal/prepare-release.sh` the
release path uses, so a wheel built against a ROCm names it. Only a build
that was never told which ROCm it is for keeps the bare version. That
also keeps the wheels installable as a set: `packages/numba-hip` pins its
siblings at `>=7.2.3`, a release version, which a bare `0.0.1` cannot
satisfy.

`HIP_PYTHON_VERSION` and (on release branches) `VERSION.in` are tracked;
the rendered `VERSION` and per-package copies are gitignored. Override
with `-DHIP_PYTHON_VERSION=...`.

- **Source-tree wheel/sdist build**: the unified
  `cmake -B build` from `packages/` does a `configure_file()` of
  repo-root `VERSION` → each `packages/<pkg>/VERSION`. Run the
  unified configure once before any per-package `python -m build`
  invocation.
- **sdist install path** (downstream consumer of the .tar.gz): the
  sdist already contains `VERSION` (added via `sdist.include`).
  pip extracts it and scikit-build reads it directly — no source
  tree, no unified configure needed.
- **`hip_python_resolve_version()`** (in
  `cmake/HipPythonBuild.cmake`) ensures the per-package `VERSION`
  file exists at CMake configure time (populating it from the
  repo-root `VERSION` on sdist builds). scikit-build-core reads the
  distribution version from that `VERSION` file directly
  (`metadata.version.input = "VERSION"`), so `importlib.metadata`
  is the runtime version source — there is no longer a generated
  `_version.py`.

The shared `cmake/HipPythonBuild.cmake` follows the same model:
mirrored into each `packages/<pkg>/cmake/HipPythonBuild.cmake` by the
unified configure step, listed in each per-package
`sdist.include`, gitignored.

### How inter-package dependency constraints stay in sync

Packages that depend on hip-python siblings pin them with a
compatible-release constraint (e.g.
`rocm-bindings-core~=7.13.0.0`). To keep that constraint from drifting
away from the canonical `VERSION`, those packages **do not** commit a
`pyproject.toml`. Instead they commit a `pyproject.toml.in` template
whose dependency lines reference `@HIP_PYTHON_DEP_VERSION@`, and the
unified `cmake -B build` renders `pyproject.toml` from it (`@ONLY`) in
the same per-package `configure_file()` loop that mirrors `VERSION`.

`HIP_PYTHON_DEP_VERSION` is derived in `packages/CMakeLists.txt` from
the first (up to) four dot-separated segments of the rendered
`VERSION` — so `7.13.0.0.0.1` yields the `7.13.0.0` prefix, while a
development-branch `0.0.1` degrades to that verbatim value.

The rendered `pyproject.toml` is gitignored and is what the
standalone single-package build, the unified wheel assembler
(`pyproject-metadata` reads it for `.dist-info`), and the sdist tarball
all consume — downstream `pip install` never sees the `.in` template.
`rocm-bindings-core` (no version-dependent siblings) and `numba-hip`
(independent version) have no template and keep a tracked
`pyproject.toml`. As with `VERSION`, run the unified configure once
before any per-package `python -m build`, or the rendered
`pyproject.toml` will be missing.

## Build requirements

- **Linux** (only platform tested; manylinux wheels target
  `manylinux_2_17_x86_64`).
- **C compiler** (GCC or Clang).
- **Python 3.9+** with `pip>=24.0`, `venv`, and development headers.
- **CMake ≥ 3.26** and **Ninja ≥ 1.11** recommended. 3.26 is also what
  the `Development.SABIModule` component needs — see
  [Stable-ABI (abi3) wheels](#stable-abi-abi3-wheels).
- Python packages: `scikit-build-core>=0.11.2`, `cython>=3.1.0`,
  `build`, `pyproject-metadata>=0.9` (used by the unified build's
  wheel assembler — see
  [Wheel assembly](#wheel-assembly-compile-once-then-collect)).
  Optional for production wheels: `auditwheel`, `patchelf`.
  See [Cython version requirement](#cython-version-requirement) for
  why the floor is 3.1.0.
- **ROCm SDK** at `${ROCM_PATH}` (defaults to `/opt/rocm`).
- For docs builds: `sphinx`, `sphinx-autoapi`, `rocm-docs-core`, plus the
  rest of `docs_src/sphinx/requirements.txt`.
- For developer-only stub regeneration (see next section): `mypy`.

## Stable-ABI (abi3) wheels

Ordinarily each compiled wheel is tagged for the exact CPython that
built it (`cp312-cp312-…`), so supporting four interpreters means four
builds. Building the extensions against the CPython stable ABI instead
collapses that to one wheel set per platform, tagged
`cp<floor>-abi3-<platform>`, which installs on every interpreter from
`<floor>` upwards.

One CMake variable turns it on, and it behaves identically on Linux and
Windows:

```sh
cd packages
cmake -B build -DHIP_PYTHON_ABI3_FLOOR=3.11
cmake --build build --target all_wheels
```

Both CI wrappers pass it for you:

```sh
USE_SABI=3.11 ci/internal/build-wheels.sh          # Linux
```

```powershell
ci\internal\build-wheels.ps1 -UseSabi 3.11         # Windows
```

`build-wheels.ps1` also reads `USE_SABI` from the environment, so a CI
job can set that one variable for both platforms. It rejects a
malformed or too-low floor before the toolchain is even probed; the
bash script only checks the shape and leaves the rest to CMake.

### Choosing the floor

**The floor must be at least 3.11.** `rocm-bindings-core`'s `types.pyx`
consumes the buffer protocol (`PyObject_GetBuffer`, `PyBuffer_Release`)
to accept `bytes`, `bytearray` and NumPy arrays wherever a `Pointer` is
expected, and those entered the limited API in CPython 3.11. A lower
floor compiles until it reaches those calls and then fails in the C
compiler.

It must also be **no newer than the interpreter running the build** —
you cannot target a stable ABI that the headers you compile against do
not describe yet. `hip_python_initialize()` in
`cmake/HipPythonBuild.cmake` checks both the `major.minor` shape and
this upper bound, since it is the code that knows which Python
`find_package` found.

The floor does not otherwise constrain the build interpreter: a floor of
3.11 built under Python 3.14 produces wheels that install and run on
3.11, 3.12, 3.13 and 3.14 alike.

### How it is wired

- `find_package(Python …)` adds the `Development.SABIModule` component
  when a floor is set, which is what defines the `Python::SABIModule`
  target (CMake ≥ 3.26; already the project floor).
- `Python_add_library(… USE_SABI <floor> …)` defines `Py_LIMITED_API`
  for the floor (`0x030b0000` for 3.11) and links the stable-ABI import
  library instead of the version-specific one.
- The wheel assembler receives `--abi3-floor` and tags the wheel
  `cp<floor>-abi3-<platform>` instead of `cp<ver>-cp<ver>-<platform>`.
- The pure-Python packages (`hip-python`, `numba-hip`) are unaffected:
  they are built by `python -m build` and stay `py3-none-any`.

Package metadata is deliberately not narrowed to the floor:
`requires-python` remains `>=3.9` because a non-abi3 build of the same
sources supports it. The wheel tag is what gates installation — pip
will not install a `cp311-abi3` wheel on 3.10.

### On Windows

Nothing extra is needed, and the two things that could plausibly break
do not:

- **Linking.** The stable-ABI import library `python3.lib` ships in
  `<prefix>\libs` with every CPython Windows installation, so
  `Development.SABIModule` resolves out of the box, and the resulting
  `.pyd` imports `python3.dll` only. That DLL resolves at import time
  through CPython's own extension-loading search, with no help from
  `PATH`.
- **Module filenames.** Windows has no `.abi3.pyd` import suffix —
  `importlib.machinery.EXTENSION_SUFFIXES` lists only
  `.cp3XX-win_amd64.pyd` and `.pyd` — so an abi3 infix would produce an
  unimportable module. CMake leaves `Python_SOSABI` empty there, so the
  modules keep their plain `types.pyd` names while the *wheel* still
  carries the `cp<floor>-abi3-win_amd64` tag.

Prefer `-UseSabi` over hand-writing the CMake flag on Windows. Windows
PowerShell 5.1 splits an unquoted `-DHIP_PYTHON_ABI3_FLOOR=3.11` on its
way to a native command into `-DHIP_PYTHON_ABI3_FLOOR=3` plus a stray
`.11`, and CMake then rejects the floor as `'3'` while warning about an
extra path. Quoting the whole argument avoids it, which is what the
script does.

This was validated with MSVC 14.51 and Cython 3.2.9: `rocm-bindings-core`
and `rocm-bindings-hip` built at floor 3.11 under Python 3.14, then
installed into 3.12 and 3.13 environments where a host↔device `hipMemcpy`
roundtrip ran on gfx1103.

## Cython version requirement

**Floor: `cython >= 3.1.0`.** Earlier 3.0.x releases (up to and
including the latest 3.0.12) silently miscompile a specific Cython
construct used by the interfacegen-generated bindings, producing a
wrapper that segfaults at runtime.

### The bug

For a `cdef`-with-initializer where the type contains the inner
`*const *` pattern (i.e. `const T *const *` or `T *const *` —
shapes that arise naturally for C signatures like
`hiprtcCompileProgram`'s
`const char *const * options` or `hipSignalExternalSemaphoresAsync`'s
`const hipExternalSemaphore_t * extSemArray`), Cython 3.0.x parses
the statement, emits a warning
`local variable 'x' referenced before assignment`, and then **drops
the initializer assignment from the generated C**. The local is
declared but never assigned. At runtime the call site passes
whatever the stack happened to contain (typically `NULL`) to the
backend, segfaulting on the first dereference.

The bug is fixed in Cython 3.1.0 and later; 3.1.x and 3.2.x emit
the assignment correctly with no warning. The interfacegen repo
carries unit-test coverage that pins both the bug shape (the
``*const *`` pattern) and the trailing-const handling — see
``test_typed_helpers.py::test_call_arg_hoist_double_const_pointer_uses_combined_form``
and the sibling tests around it.

### The floor is the only defense

The codegen emits the combined `cdef T x = <T>expr` form for every
hoisted call argument — precisely the shape 3.0.x miscompiles. It
used to split that into a bare `cdef` plus a separate assignment as
a second layer of defense, but that workaround was removed once the
floor was in place, and
``test_call_arg_hoist_double_const_pointer_uses_combined_form`` now
asserts the split form does not come back. So there is no
codegen-side fallback: the floor is what stands between a 3.0.x
toolchain and a segfaulting wrapper.

Not to be confused with the *two-local* split that
`interfacegen.cython.CallArgHoist.render_prehoist` still emits for
wrapper-bound arguments (`cdef <wrapper> x_obj = ...` followed by
`cdef T x = <T>x_obj.getPtr()`). That one keeps the adapter alive
across the C call and has nothing to do with this Cython bug; see
[BINDINGS.md](BINDINGS.md) under "GIL semantics — what runs where".

### What if I have to use Cython 3.0.x

Don't. With no codegen-side fallback left, a 3.0.x build produces
bindings whose `*const *` arguments are NULL locals, which segfault
in the backend on first dereference — and any *handcoded* Cython
using the same `cdef T x = <T>expr` shape fails the same way.

## Regenerating stubs for handcoded Cython modules

A handful of handcoded Cython modules ship in `rocm-bindings-core`
(`rocm.bindings.util.types`) and `rocm-bindings-hip`
(`rocm.bindings._hip_helpers`, `rocm.bindings._hiprtc_helpers`).
Their `.pyi` type-stub counterparts are **committed to git**
alongside the `.pyx` so:

- `sphinx-autoapi` can document them (astroid cannot parse `.pyx`).
- Static type checkers (mypy, pyright) and IDEs see real signatures.
- Consumers building from sdist or wheel get the stubs in the
  installed wheel without running any extra tool.

The interfacegen-owned packages (`rocm-bindings-hip`'s `hip` and
`hiprtc`, libraries, systems, compiler, hip-python-interop) get
their `.pyi` from interfacegen at codegen time — those stubs are
**not** maintained via the dev workflow described here.

The DLL loaders are handcoded as well but carry **no** stub. Every
function `rocm.bindings.util.{loader,posixloader,win32loader}`
declares is `cdef`, so they are reachable only through `cimport` and
their `.pyi` would contain nothing but `__pyx_capi__` and `__test__`
— the same rule that keeps the generated `cy*` wrappers stubless.
They ship as `.pxd`/`.pyx` only.

### The one hand-maintained exception: `cuda.bindings.cufile`

`cuda.bindings.cufile` is handcoded but is deliberately **absent**
from `HIP_PYTHON_STUBGEN_MODULES`; `all_stubs` will not refresh it.
Its public surface is almost entirely module-level `cpdef`
functions, and stubgen can only render those as opaque
`cython_function_or_method` attributes — running it would throw away
the module docstring, the enum members and every typed signature
that sphinx-autoapi renders into the API reference. Edit
`cufile.pyi` by hand in the same commit as `cufile.pyx`. The
`tests/stubs` suite checks that the stub still names every public
symbol the module exposes, which is the drift that reaches users.

### Stubs are generated artifacts — never edit them

Every stub written by this workflow opens with:

```
# AUTO-GENERATED by `mypy stubgen` from the compiled extension module.
# Type stubs for <module>. Edits will be overwritten.
# Regenerate with `ci/docs/regenerate-stubs.sh` after editing the .pyx.
```

Hand-editing one is always wrong: the next run of the target
silently reverts it, and in the meantime the published docs describe
an API that does not exist. `rocm.bindings.util.types` drifted this
way through four consecutive commits, ending with a `CStr`
docstring that contradicted the implementation. Nothing re-runs
stubgen in CI to catch that — the banner is the whole guard, and
`tests/stubs` covers only the hand-maintained `cufile` stub.

`cmake/hip_python_postprocess_stub.py` adds that banner and also
strips the Cython version out of the stub: stubgen renders
module-level `cpdef` objects as
`_cython_<major>_<minor>_<patch>.cython_function_or_method`, which
would otherwise make the committed `.pyi` depend on whichever Cython
the developer happened to have installed. It is rewritten to
`Callable[..., Any]`, so the same `.pyx` yields a byte-identical
`.pyi` on any Cython 3.x.

### Developer workflow

After editing one of the handcoded `.pyx` files (changing a class
or function signature, adding a method, etc.):

```sh
pip install mypy                                   # one-time
cd packages
cmake -B build -DHIP_PYTHON_ENABLE_STUBGEN=ON
cmake --build build --target all_stubs             # all handcoded modules
# or one package at a time:
cmake --build build --target core_stubs
cmake --build build --target hip_stubs
# or a single module:
cmake --build build --target rocm_bindings_core_types_stub
```

The regenerated `.pyi` lands in the source tree next to the `.pyx`.
Inspect the diff with `git diff packages/rocm-bindings-*/src/rocm/...`,
then commit `<module>.pyx` and `<module>.pyi` together.

`ci/docs/regenerate-stubs.sh` wraps the same flow.

Configure that tree **without** `HIP_PYTHON_ABI3_FLOOR`. An extension
built against the CPython limited API tells stubgen much less about
itself: Cython replaces the per-method docstrings — the ones
sphinx-autoapi publishes — with CPython's generic text. Such a stub
looks plausible in a diff, so `hip_python_postprocess_stub.py`
detects the limited-API build and fails rather than let one reach the
source tree.

### What is NOT triggered

- `pip install rocm-bindings-core` (from wheel or sdist) — uses the
  committed `.pyi` directly. `mypy` is **not** a build-system
  dependency; it's a developer-only tool.
- `cmake --build build --target all_wheels` / `all_sdists` — wheel
  and sdist targets do **not** depend on `all_stubs`.
- `HIP_PYTHON_ENABLE_STUBGEN=OFF` (the default) — the stubgen
  targets are not even created; CMake configure does not look for
  `mypy`.

### Adding a new handcoded Cython module to be stubbed

The list of modules is repo-spanning and lives in **one place**:
`HIP_PYTHON_STUBGEN_MODULES` in `packages/CMakeLists.txt`. Append one
line of the form:

```
"<package-shortname>|<source-pyi-relative-dir>|<cython-target>"
```

`<package-shortname>` matches the `HIP_PYTHON_BUILD_<NAME>` option
suffix (`core`, `hip`, `libraries`, `systems`, `compiler`,
`interop`); the entry is silently skipped if the owning package is
disabled in this configure. The dispatch loop below the list calls
`hip_python_add_stubgen_target()` (in
`cmake/HipPythonBuild.cmake`) for each entry and aggregates them
into per-package `<pkg>_stubs` targets and the repo-wide
`all_stubs` target.

The dotted module name is **not** part of the entry: it is read back
from the target's `HIP_PYTHON_MODULE_NAME` property, set by
`hip_python_add_cython_module()`. Spelling it out here would be
wrong for any target whose module name differs per platform — a
hardcoded Linux name for `rocm_bindings_core_platform_loader`, which
compiles `posixloader` on Linux and `win32loader` on Windows, used to
make `core_stubs` fail outright on Windows.

Per-package `CMakeLists.txt` files do not contain any stubgen
wiring — the single list is the authoritative source.

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
cd packages
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
# NOTE: requires a one-time unified configure first (`cd packages &&
# cmake -B build`) to populate the gitignored per-package VERSION and
# mirror the shared cmake helper. Without it, scikit-build-core fails
# with `FileNotFoundError: VERSION`. See section B.
cd packages && cmake -B build                 # one-time, populates VERSION + cmake helper
cd packages/rocm-bindings-core && python3 -m build --wheel --no-isolation

# Full build, all eight packages (run from packages/ subdir):
cd packages && cmake -B build && cmake --build build --target all_wheels -j$(nproc)

# Production manylinux wheels:
cd packages && cmake -B build -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON
cmake --build build --target all_wheels -j$(nproc)

# One stable-ABI wheel set for Python 3.11+ instead of one per version
# (see "Stable-ABI (abi3) wheels"):
cd packages && cmake -B build -DHIP_PYTHON_ABI3_FLOOR=3.11
cmake --build build --target all_wheels -j$(nproc)

# Skip the compiler package (faster; doesn't need libLLVM):
cd packages && cmake -B build -DHIP_PYTHON_BUILD_COMPILER=OFF
cmake --build build --target all_wheels

# Debug build:
cd packages && cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target all_wheels

# Use sccache:
cd packages && cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake --build build --target all_wheels

# Custom ROCm path:
cd packages && cmake -B build -DROCM_PATH=/opt/rocm-7.13
cmake --build build --target all_wheels

# Single package wheel from the unified build:
cd packages && cmake -B build && cmake --build build --target core_wheel

# Build the documentation (independent of all_wheels):
cd packages && cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target docs

# Wheels and docs in parallel:
cd packages && cmake -B build -DHIP_PYTHON_BUILD_DOCS=ON
cmake --build build --target all_wheels docs -j$(nproc)
```

## See also

- [CODEGEN.md](CODEGEN.md) — how interfacegen produces the generated Cython sources
- [BINDINGS.md](BINDINGS.md) — the generated-bindings emission contract (module layout, pointer/return conventions)
- Project [README](../../README.md) — install + quick-start
