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
|   + hip-python-    |                    | base branch (handcoded|
|   generate         |                    | sources + build files)|
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

### Codegen base branch

Contains **only handcoded** content:

- All build infrastructure: `cmake/HipPythonBuild.cmake`, every per-package `CMakeLists.txt`, `pyproject.toml` files, the `rocm-bindings-core/version.py.in` template, `setup.cfg`, `MANIFEST.in`.
- The `rocm-bindings-core` package — its loader, types, and `paths` modules are handcoded and not produced by the generator. The one exception is `rocm/version.py`, which the generator renders from the handcoded `rocm-bindings-core/version.py.in` template (like the other generator outputs it is absent on the base branch and committed on release branches).
- Per-package `__init__.py` files (Python runtime markers).
- Top-of-namespace `__init__.pxd` markers (`rocm/__init__.pxd`, `rocm/bindings/__init__.pxd`, `cuda/__init__.pxd`, `cuda/bindings/__init__.pxd`) committed in each package source tree.
- The handcoded helper Cython modules `_hip_helpers.{pxd,pyx}` and `_hiprtc_helpers.{pxd,pyx}` in `rocm-bindings-hip`.
- Documentation, examples, license.

A bare clone of the codegen base branch is **not buildable** — the generator must run into it first to populate the `.pxd`/`.pyx` files for `hip`, `hiprtc` (`rocm-bindings-hip`); `hipblas`, `hipsolver`, `hiprand`, `hipfft`, `hipsparse` (`rocm-bindings-libraries`); `rccl`, `roctx` (`rocm-bindings-systems`); `amd_comgr`, the LLVM-C suite (`rocm-bindings-compiler`); the CUDA interop layer; and `rocm/version.py` (`rocm-bindings-core`, rendered from `version.py.in`).

### Release branches — `release/rocm-rel-X.Y[.Z]`

The base branch **plus** the generator output for one specific ROCm version. This is what users build from and what the prebuilt PyPI wheels are produced from.

Release branches are immutable snapshots: they are checked in once per supported ROCm release and not regenerated in place. To support a new ROCm version, the generator runs against that version's headers into a fresh checkout of the codegen base branch, and the result is committed as a new `release/rocm-rel-X.Y` branch.

## What the generator owns vs. what the base branch owns

The generator's responsibility is **strictly Cython source generation** plus the small set of build inputs that depend on the parsed ROCm version:

### Generator-owned outputs

For each release run, the `hip-python-generate` CLI (after `pip install`
from `tools/hip-python-generate/`) writes:

| Path | Content |
|---|---|
| `packages/rocm-bindings-hip/src/rocm/bindings/{,cy}{hip,hiprtc}.{pxd,pyx}` | HIP runtime + RTC bindings |
| `packages/rocm-bindings-libraries/src/rocm/bindings/{,cy}<lib>.{pxd,pyx}` | hipblas, hipsolver, hiprand, hipfft, hipsparse |
| `packages/rocm-bindings-systems/src/rocm/bindings/{,cy}<lib>.{pxd,pyx}` | rccl, roctx, hipfile, amdsmi, hsa |
| `packages/rocm-bindings-compiler/src/rocm/bindings/{,cy}amd_comgr.{pxd,pyx}` | AMD COMGR |
| `packages/rocm-bindings-compiler/src/rocm/bindings/llvm/c/**/*.{pxd,pyx}` | LLVM-C suite (~30 modules + transforms + config) |
| `packages/hip-python-interop/src/cuda/bindings/{,cy}{driver,runtime,nvrtc}.{pxd,pyx}` | CUDA interop layer (HIP-as-CUDA) |
| `__init__.pxd` files **below** `rocm/bindings/` and `cuda/bindings/` | Cython namespace markers (build-time only, never installed) |
| `packages/rocm-bindings-libraries/cmake/generated_modules.cmake` | Module list for the libraries package — drives the per-library CMake foreach loop. |
| `packages/rocm-bindings-systems/cmake/generated_modules.cmake` | Module list for the systems package (rccl, roctx, hipfile, amdsmi, hsa). |
| `packages/rocm-bindings-compiler/cmake/generated_modules.cmake` | LLVM-C / transforms / config / COMGR module lists. |
| `packages/rocm-bindings-core/src/rocm/version.py` | Runtime version module rendered from the handcoded `rocm-bindings-core/version.py.in` template. Exposes the ROCm/HIP versions + commit and codegen provenance (base branch/rev/version, interfacegen version) via `rocm.version`. Like the other generator outputs it is absent on the base branch and committed on release branches by the release commit's `git add packages`. |
| `packages/rocm-bindings-{hip,libraries,systems,compiler}/cmake/generated_versions.cmake`<br>`packages/hip-python-interop/cmake/generated_versions.cmake` | Version metadata for the docs landing page: ROCm version, HIP version, hip-python base branch/rev/version, interfacegen version, codegen date, and upstream source-tree revs. |
| `<package>/<rocm-or-cuda>/<…>/<name>.pyi` (high-level modules only) | Type-stub files emitted alongside every high-level `<name>.pxd`/`.pyx` pair. Used by static type checkers (mypy, pyright) and IDEs to resolve symbol signatures without the compiled extensions on `sys.path`. The cy* C-level wrappers do **not** get `.pyi` — they are `cimport`-only and have no honest Python type-system equivalents. Installed alongside the corresponding `.so`. |

> **Note on handcoded Cython modules.** The handful of handcoded
> `.pyx` files in `rocm-bindings-core` (`rocm.bindings.util.{types,
> loader,posixloader}`) and `rocm-bindings-hip` (`_hip_helpers`,
> `_hiprtc_helpers`) are **not** touched by interfacegen. Their
> `.pyi` stubs are committed to git and refreshed by a developer-run
> CMake target backed by `mypy stubgen` — see
> [BUILDING.md](BUILDING.md) §"Regenerating stubs for handcoded
> Cython modules" for the workflow.
| `docs_src/python_api/<dotted-module-name>.rst` (high-level modules) | Sphinx wrapper page that points `sphinx-autoapi` at the high-level Python module. One file per generated module (`rocm.bindings.hipblas.rst`, `cuda.bindings.driver.rst`, etc.). |
| `docs_src/python_api/<dotted-module-name>.rst` (cy* wrappers) | Sphinx wrapper page for each generated `cy<name>.pxd`. Uses `literalinclude` to embed the .pxd source with Cython syntax highlighting — the `.pxd` itself is the readable, source-of-truth contract for downstream Cython users. No autoapi or `.pyi` involved. |

### Forbidden outputs (handcoded; generator must NEVER write)

- `__init__.py` files anywhere
- the `version.py.in` template (the generator *reads* it to render `rocm/version.py`, but never writes the template itself)
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
   hip-python-generate \
       /path/to/hip-python \
       --rocm-version X.Y.Z \
       --rocm-path /opt/rocm
   ```
   (Install via `pip install -r dev-requirements.txt && pip install .`
   inside `tools/hip-python-generate/`; see that directory's `README.md`.)

   After installation, running the CLI writes:
   - `.pxd`/`.pyx` files into the five generator-owned packages
     (hip, libraries, systems, compiler, and the hip-python-interop shim)
   - `__init__.pxd` namespace markers below `rocm/bindings/` and `cuda/bindings/`
   - `cmake/generated_modules.cmake` for libraries and compiler
   - `cmake/generated_versions.cmake` for every package
   - `docs_src/sphinx/_toc.yml.in` (rendered from `_toc.yml.in.in`) and the
     `docs_src/python_api/*.rst` cy* literalinclude pages

3. **Verify the round-trip.** A clean `cmake -S packages -B build && cmake --build build --target all_wheels` should produce manylinux-compatible wheels for all six packages with no Cython errors.

4. **Author the release-only `VERSION.in`** embedding the ROCm version
   (writes `VERSION.in = X.Y.Z.@HIP_PYTHON_VERSION@`; `@HIP_PYTHON_VERSION@`
   stays literal for CMake's `configure_file`):
   ```sh
   ci/internal/prepare-release.sh X.Y.Z
   ```

5. **Commit and push the release branch.** `git add` captures the rendered
   `version.py` and the other generator outputs (none are git-ignored; they
   are simply absent on the base branch). This includes the docs-side outputs
   under `docs_src/` (`_toc.yml.in` + the `python_api/*.rst` cy* pages) that
   `ci/docs/build.sh` consumes without re-running codegen.
   ```sh
   git add packages/ VERSION.in docs_src/sphinx/_toc.yml.in docs_src/python_api
   git commit -m "[chore] generate bindings for ROCm X.Y.Z"
   git push origin release/rocm-rel-X.Y
   ```

6. **Build wheels for distribution** (CI builds them with `auditwheel repair` for the manylinux tag) and upload to PyPI.

## Adding or removing modules

The module count for `rocm-bindings-libraries`, `rocm-bindings-systems`, and `rocm-bindings-compiler` can change between ROCm releases (e.g., a new LLVM-C header appears, or AMD adds a new high-level library). The build system absorbs this through the `cmake/generated_modules.cmake` files:

```cmake
# packages/rocm-bindings-libraries/cmake/generated_modules.cmake
# AUTO-GENERATED — do not edit by hand.
set(HIP_PYTHON_LIBRARIES_GENERATED_MODULES
    hipblas hipsolver hiprand hipfft hipsparse)

# packages/rocm-bindings-systems/cmake/generated_modules.cmake
set(HIP_PYTHON_SYSTEMS_GENERATED_MODULES
    rccl roctx hipfile amdsmi hsa)
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

- The two top-of-namespace markers per package (`rocm/__init__.pxd` and `rocm/bindings/__init__.pxd`; or `cuda/`, `cuda/bindings/`) are **handcoded** because they exist regardless of generator output. They live on the codegen base branch. `rocm-bindings-core` also handcodes `rocm/bindings/util/__init__.pxd` (the `util/` subtree is handcoded, not generator-owned).
- Markers below those (`rocm/bindings/llvm/__init__.pxd`, `rocm/bindings/llvm/c/__init__.pxd`, etc.) are **generator-emitted** because the directory tree they describe is generator-owned.
- Most markers are **build-time only** — they are NOT installed. `rocm-bindings-core` is the single component that installs the runtime namespace markers (`rocm/__init__.pxd`, `rocm/bindings/__init__.pxd`, and `rocm/bindings/util/__init__.pxd`); the `util` marker is installed so the other packages' cross-package `cimport rocm.bindings.util.{types,loader}` resolves against the installed core. At Python runtime `rocm.bindings.util` is a PEP 420 namespace package (no `__init__.py`; the import system ignores the `.pxd`).

This keeps the runtime install tree clean (one marker per namespace level, contributed by `rocm-bindings-core`) while letting Cython resolve cross-package `cimport` at build time inside every package's source tree.

## File-prefix history

Generator output used to use a `c<name>` prefix (`chip.pxd`, `chiprtc.pxd`, `chipblas.pxd`, `corc.pxd`, …) for C-level wrappers. The repo migrated to a `cy<name>` prefix (`cyhip.pxd`, `cyhiprtc.pxd`, …) to disambiguate Cython-level wrappers from anything `c` might collide with. The migration is centralized in `interfacegen.cython.CythonModuleGenerator.write_module_files`:

```python
cmodule_name = f"cy{self.module_name}"
```

Every site that emits a C-level module name — filenames, package-relative `cimport` paths, qualified attribute references inside generated bodies — flows through this single line.

## Repository layout (generator side)

```
tools/hip-python-generate/
├── pyproject.toml              CLI entry-point + runtime deps (hip-python-codegen)
├── README.md                   CLI usage
├── dev-requirements.txt        path-relative interfacegen install
└── src/hip_python_codegen/
    ├── generate.py             CLI: argparse + main() dispatcher
    ├── binding_generator.py    master orchestrator + cmake/marker writers
    ├── docs_generator.py       Sphinx page + TOC YAML emission
    ├── generators_hip.py       hip + hiprtc generators
    ├── generators_libraries.py hipblas/hipsolver/hiprand/hipfft/hipsparse generators
    ├── generators_systems.py   rccl/roctx/hipfile/amdsmi/hsa generators
    ├── generators_compiler.py  amd_comgr + llvm generators
    ├── cuda_interop.py         CUDA interop (driver/runtime/nvrtc) subgenerator
    └── hipify.py               hipify-perl substitution parser
```

The CLI entry is `generate.py` (installed as the `hip-python-generate`
console script). The master orchestrator `binding_generator.py` invokes
the per-wheel generators (`generators_*.py`) and emits cross-cutting
build inputs (Cython namespace markers, cmake module/version lists).
Sphinx page emission lives in `docs_generator.py`. The previously
separate `comgr` and `llvm` recipes have been merged into the unified
hip recipe as libraries inside `binding_generator.AVAILABLE_GENERATORS`.

## Reproducing a release locally

To run the generator end-to-end against a checked-out codegen base branch:

```sh
# Prereqs: ROCm SDK installed at /opt/rocm.
cd /path/to/hip-python
git switch codegen/base

# One-time install of the codegen tool (now in-tree under tools/):
cd /path/to/hip-python/tools/hip-python-generate
python3 -m venv .venv && source .venv/bin/activate
pip install -r dev-requirements.txt   # editable interfacegen (../interfacegen)
pip install .                         # hip-python-codegen + runtime deps

# Each release run:
hip-python-generate \
    /path/to/hip-python \
    --rocm-version X.Y.Z \
    --rocm-path /opt/rocm

cd /path/to/hip-python
cmake -S packages -B build && cmake --build build --target all_wheels
```

Alternatively, let CMake run the generator for you at configure time
(it invokes the same `hip-python-generate` tool, so it must be installed
as above). This keeps the whole flow to a single configure + build:

```sh
cmake -S packages -B build \
    -DHIP_PYTHON_RUN_CODEGEN=ON \
    -DHIP_PYTHON_ROCM_PATH=/opt/rocm \
    -DHIP_PYTHON_ROCM_VERSION=X.Y.Z
cmake --build build --target all_wheels
```

Codegen then runs DURING the `cmake -S packages -B build` configure
(before any build target exists), so it is SLOW and blocks configure for
several minutes up to ~30 min depending on core count. A stamp guard
skips it on no-op reconfigures; pass `-DHIP_PYTHON_FORCE_CODEGEN=ON` to
force a re-run. See BUILDING.md "Optional configure-time code generation".

After this, `packages/build/dist/` (or whatever `HIP_PYTHON_WHEEL_OUTPUT_DIR` points to) contains the wheels.

## See also

- [BINDINGS.md](BINDINGS.md) — anatomy of the generated bindings (two-tier wrapper layout, GIL semantics, generated-function regions, naming conventions, pointer-arg intent classification, handcoded helpers)
- [BUILDING.md](BUILDING.md) — the hip-python build system in detail
