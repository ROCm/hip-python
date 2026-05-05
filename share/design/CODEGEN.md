# Code generation for HIP Python releases

This document describes how the **interfacegen** code generator and the
**hip-python codegen base branch** combine to produce a HIP Python release.

## Overview

HIP Python is built from two repositories:

| Repository | Role |
|---|---|
| [`interfacegen`](https://github.com/AMD-AIOSS/interfacegen) | The code generator. Parses ROCm C/C++ headers via libclang and emits Cython sources. |
| [`hip-python`](https://github.com/AMD-AIOSS/hip-python) | The Python project. Owns the build system (CMake, scikit-build-core), packaging metadata, handcoded sources, and the generated Cython sources after each release. |

A release flow looks like this:

```
                 +--------------------+
                 |  ROCm SDK headers  |
                 |  (/opt/rocm/...)   |
                 +---------+----------+
                           |
                           v
+--------------------+  parses & renders  +-----------------------+
|   interfacegen     | -----------------> | hip-python codegen    |
|   recipes/         |                    | base branch (handcoded|
|   hip_python/      |                    | sources + build files)|
+--------------------+                    +-----------+-----------+
                                                       |
                                                  generated
                                          .pxd / .pyx + .pyi stubs
                                          + docs_src/python_api/*.rst
                                          + cmake/generated_*.cmake
                                                       v
                                          +-----------------------+
                                          | Release branch:       |
                                          | release/rocm-rel-X.Y  |
                                          | (base + generated)    |
                                          +-----------+-----------+
                                                       |
                                              built into wheels (PyPI)
                                              + Sphinx HTML (docs/)
                                                       v
                                                  Wheels + Docs
```

## The two branch tiers in `hip-python`

The `hip-python` repo distinguishes two kinds of branches:

### Codegen base branch — `dev/docharri/hip-python-codegen-base`

Contains **only handcoded** content:

- All build infrastructure: `cmake/HipPythonBuild.cmake`, every per-package `CMakeLists.txt`, `pyproject.toml` files, `_version.py.in` templates, `setup.cfg`, `MANIFEST.in`.
- The `rocm-bindings-core` package in full — its loader, types, and `paths` modules are handcoded and not produced by the generator.
- Per-package `__init__.py` files (Python runtime markers).
- Top-of-namespace `__init__.pxd` markers (`rocm/__init__.pxd`, `rocm/bindings/__init__.pxd`, `cuda/__init__.pxd`, `cuda/bindings/__init__.pxd`) committed in each package source tree.
- The handcoded helper Cython modules `_hip_helpers.{pxd,pyx}` and `_hiprtc_helpers.{pxd,pyx}` in `rocm-bindings-hip`.
- Documentation, examples, license.

A bare clone of the codegen base branch is **not buildable** — the generator must run into it first to populate the `.pxd`/`.pyx` files for `hip`, `hiprtc` (`rocm-bindings-hip`); `hipblas`, `hipsolver`, `hiprand`, `hipfft`, `hipsparse` (`rocm-bindings-libraries`); `rccl`, `roctx` (`rocm-bindings-systems`); `amd_comgr`, the LLVM-C suite (`rocm-bindings-compiler`); and the CUDA interop layer.

### Release branches — `release/rocm-rel-X.Y[.Z]`

The base branch **plus** the generator output for one specific ROCm version. This is what users build from and what the prebuilt PyPI wheels are produced from.

Release branches are immutable snapshots: they are checked in once per supported ROCm release and not regenerated in place. To support a new ROCm version, the generator runs against that version's headers into a fresh checkout of the codegen base branch, and the result is committed as a new `release/rocm-rel-X.Y` branch.

## What the generator owns vs. what the base branch owns

The generator's responsibility is **strictly Cython source generation** plus the small set of build inputs that depend on the parsed ROCm version:

### Generator-owned outputs

For each release run, `recipes/hip_python/generate_hip_python.sh` writes:

| Path | Content |
|---|---|
| `python/rocm-bindings-hip/rocm/bindings/{,cy}{hip,hiprtc}.{pxd,pyx}` | HIP runtime + RTC bindings |
| `python/rocm-bindings-libraries/rocm/bindings/{,cy}<lib>.{pxd,pyx}` | hipblas, hipsolver, hiprand, hipfft, hipsparse |
| `python/rocm-bindings-systems/rocm/bindings/{,cy}<lib>.{pxd,pyx}` | rccl, roctx |
| `python/rocm-bindings-compiler/rocm/bindings/{,cy}amd_comgr.{pxd,pyx}` | AMD COMGR |
| `python/rocm-bindings-compiler/rocm/bindings/llvm/c/**/*.{pxd,pyx}` | LLVM-C suite (~30 modules + transforms + config) |
| `python/hip-python-interop/cuda/bindings/{,cy}{driver,runtime,nvrtc}.{pxd,pyx}` | CUDA interop layer (HIP-as-CUDA) |
| `__init__.pxd` files **below** `rocm/bindings/` and `cuda/bindings/` | Cython namespace markers (build-time only, never installed) |
| `python/rocm-bindings-libraries/cmake/generated_modules.cmake` | Module list for the libraries package — drives the per-library CMake foreach loop. |
| `python/rocm-bindings-systems/cmake/generated_modules.cmake` | Module list for the systems package (rccl, roctx). |
| `python/rocm-bindings-compiler/cmake/generated_modules.cmake` | LLVM-C / transforms / config / COMGR module lists. |
| `python/rocm-bindings-{hip,libraries,compiler}/cmake/generated_versions.cmake`<br>`python/hip-python-interop/cmake/generated_versions.cmake` | Version metadata: ROCm version, HIP version, code-generator branch/rev, hip-python branch/rev. Consumed by the existing `configure_file("_version.py.in" "_version.py")` flow. |
| `<package>/<rocm-or-cuda>/<…>/<name>.pyi` (high-level modules only) | Type-stub files emitted alongside every high-level `<name>.pxd`/`.pyx` pair. Used by static type checkers (mypy, pyright) and IDEs to resolve symbol signatures without the compiled extensions on `sys.path`. The cy* C-level wrappers do **not** get `.pyi` — they are `cimport`-only and have no honest Python type-system equivalents (see plan §B.2). Installed alongside the corresponding `.so`. |

> **Note on handcoded Cython modules.** The handful of handcoded
> `.pyx` files in `rocm-bindings-core` (`rocm.bindings.util.{types,
> loader,posixloader}`) and `rocm-bindings-hip` (`_hip_helpers`,
> `_hiprtc_helpers`) are **not** touched by interfacegen. Their
> `.pyi` stubs are committed to git and refreshed by a developer-run
> CMake target backed by `mypy stubgen` — see
> [BUILDING.md](BUILDING.md) §"Regenerating stubs for handcoded
> Cython modules" for the workflow.
| `docs_src/python_api/<dotted-module-name>.rst` (high-level modules) | Sphinx wrapper page that points `sphinx-autoapi` at the high-level Python module. One file per generated module (`rocm.bindings.hipblas.rst`, `cuda.bindings.driver.rst`, etc.). |
| `docs_src/python_api/<dotted-module-name>.rst` (cy* wrappers) | Sphinx wrapper page for each generated `cy<name>.pxd`. Uses `literalinclude` to embed the .pxd source with Cython syntax highlighting — the `.pxd` itself is the readable, source-of-truth contract for downstream Cython users. No autoapi or `.pyi` involved. (See plan §B.3.) |

### Forbidden outputs (handcoded; generator must NEVER write)

- `__init__.py` files anywhere
- `_version.py.in` templates
- `setup.py`, `setup.cfg`, `MANIFEST.in`, `pyproject.toml`
- `requirements.txt` files inside generated package trees
- The two top-of-namespace `__init__.pxd` files in each package
- The `rocm-bindings-hip` helper modules `_hip_helpers.{pxd,pyx}` and `_hiprtc_helpers.{pxd,pyx}`

The generator can be re-run safely against an existing tree — it only overwrites files it owns.

## Filename and naming conventions

- **C-level wrapper modules use the `cy` prefix.** `cyhip.pxd`/`cyhip.pyx` is the Cython-level wrapper for `hip`, `cyhipblas` for `hipblas`, `cyamd_comgr` for `amd_comgr`, `cydriver` for `driver`, etc. The high-level Python wrapper has no prefix (`hip.pxd`, `hipblas.pyx`, …).
- **Module names mirror file paths** under the modern hip-python layout: `rocm.bindings.hip`, `rocm.bindings.cyhip`, `rocm.bindings.llvm.c.core`, `cuda.bindings.driver`, etc.
- **Loader template uses lazy DLL path resolution** via `rocm.bindings.util.paths.get_library_path('<libname>')` — the generated `.pyx` files do not hard-code library file paths.

## Producing a new release

The end-to-end release flow:

1. **Choose the codegen base.**
   ```sh
   git -C /path/to/hip-python switch codegen/base
   git -C /path/to/hip-python switch -c release/rocm-rel-X.Y
   ```

2. **Run the consolidated generator** against the chosen ROCm SDK:
   ```sh
   /path/to/interfacegen/recipes/hip_python/generate_hip_python.sh \
       /path/to/hip-python \
       --rocm-version X.Y.Z \
       --rocm-path /opt/rocm
   ```
   The script sets up its own venv, installs `cython` + `libclang`, and writes:
   - `.pxd`/`.pyx` files into the four generator-owned packages
   - `__init__.pxd` namespace markers below `rocm/bindings/` and `cuda/bindings/`
   - `cmake/generated_modules.cmake` for libraries and compiler
   - `cmake/generated_versions.cmake` for every package

3. **Verify the round-trip.** A clean `cmake -S python -B build && cmake --build build --target all_wheels` should produce manylinux-compatible wheels for all six packages with no Cython errors.

4. **Commit and push the release branch.**
   ```sh
   git add python/
   git commit -m "[chore] generate bindings for ROCm X.Y.Z"
   git push origin release/rocm-rel-X.Y
   ```

5. **Build wheels for distribution** (CI builds them with `auditwheel repair` for the manylinux tag) and upload to PyPI.

## Adding or removing modules

The module count for `rocm-bindings-libraries`, `rocm-bindings-systems`, and `rocm-bindings-compiler` can change between ROCm releases (e.g., a new LLVM-C header appears, or AMD adds a new high-level library). The build system absorbs this through the `cmake/generated_modules.cmake` files:

```cmake
# python/rocm-bindings-libraries/cmake/generated_modules.cmake
# AUTO-GENERATED — do not edit by hand.
set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES
    hipblas hipsolver hiprand hipfft hipsparse)

# python/rocm-bindings-systems/cmake/generated_modules.cmake
set(HIP_PYTHON_SYSTEMS_GENERATED_MODULES
    rccl roctx)
```

The corresponding `CMakeLists.txt` does:

```cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/generated_modules.cmake")
set(HIP_PYTHON_ALL_LIBRARIES ${HIP_PYTHON_LIBRARIES_GENERATED_MODULES})
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_LIBRARIES)
    # build _lib.{pxd,pyx} and cy_lib.{pxd,pyx}
endforeach()
```

When a new module appears, the generator simply adds it to the list and emits the corresponding `.pxd`/`.pyx`; no per-package `CMakeLists.txt` edit is needed.

`rocm-bindings-hip` and `hip-python-interop` keep hand-listed CMakeLists.txt because their module set is stable (hip + hiprtc; driver + runtime + nvrtc respectively).

## Cython namespace markers — when, where, and why

Cython's cross-package `cimport` resolution requires that every directory along a namespace path contains a package marker (`__init__.pxd` or `__init__.py`). In hip-python:

- The two top-of-namespace markers per package (`rocm/__init__.pxd` and `rocm/bindings/__init__.pxd`; or `cuda/`, `cuda/bindings/`) are **handcoded** because they exist regardless of generator output. They live on the codegen base branch.
- Markers below those (`rocm/bindings/llvm/__init__.pxd`, `rocm/bindings/llvm/c/__init__.pxd`, etc.) are **generator-emitted** because the directory tree they describe is generator-owned.
- All non-`util` markers are **build-time only** — they are NOT installed. `rocm-bindings-core` is the single component that installs the runtime namespace markers (`rocm/__init__.pxd`, `rocm/bindings/__init__.pxd`).

This keeps the runtime install tree clean (one marker per namespace level, contributed by `rocm-bindings-core`) while letting Cython resolve cross-package `cimport` at build time inside every package's source tree.

## File-prefix history

Generator output used to use a `c<name>` prefix (`chip.pxd`, `chiprtc.pxd`, `chipblas.pxd`, `corc.pxd`, …) for C-level wrappers. The repo migrated to a `cy<name>` prefix (`cyhip.pxd`, `cyhiprtc.pxd`, …) to disambiguate Cython-level wrappers from anything `c` might collide with. The migration is centralized in `interfacegen.cython.CythonModuleGenerator.write_module_files`:

```python
cmodule_name = f"cy{self.module_name}"
```

Every site that emits a C-level module name — filenames, package-relative `cimport` paths, qualified attribute references inside generated bodies — flows through this single line.

## Repository layout (interfacegen side)

```
interfacegen/recipes/hip_python/
├── codegen.py                 unified driver (orchestrator)
├── generate_hip_python.sh     shell wrapper (venv setup + invocation)
├── requirements.txt           cython, libclang, pyparsing
├── Makefile                   builds librocmllvm.so when bundling
├── hip/
│   ├── generate_hip.py        HIP + library subgenerators (generate(opts) entry)
│   ├── cuda_interop.py        CUDA interop (driver/runtime/nvrtc) subgenerator
│   └── hipify.py              hipify-perl substitution parser
├── llvm/
│   └── generate_llvm.py       LLVM-C subgenerator
└── comgr/
    └── generate_comgr.py      AMD COMGR subgenerator
```

The orchestrator (`codegen.py`) loads each subgenerator, calls `generate(opts)`, then performs the cross-cutting steps: namespace marker emission and CMake include-file emission.

## Reproducing a release locally

To run the generator end-to-end against a checked-out codegen base branch:

```sh
# Prereqs: ROCm SDK installed at /opt/rocm; cython + libclang available.
cd /path/to/hip-python
git switch codegen/base

cd /path/to/interfacegen
recipes/hip_python/generate_hip_python.sh \
    "/path/to/hip-python" \
    --rocm-version X.Y.Z \
    --rocm-path /opt/rocm

cd /path/to/hip-python
cmake -S python -B build && cmake --build build --target all_wheels
```

After this, `python/build/dist/` (or whatever `HIP_PYTHON_WHEEL_OUTPUT_DIR` points to) contains the wheels.

## See also

- [BUILDING.md](BUILDING.md) — the hip-python build system in detail
