# Changelog

## Unreleased

### Add `cuda.bindings.cufile` interop module

The `hip-python-interop` wheel gained a `cuda.bindings.cufile` module:
a compiled interop layer mirroring CUDA Python's cuFile API, backed by
AMD's hipFILE. It is a hand-written pair
(`cufile.pyx`/`cufile.pxd` cpdef layer over a hand-written
`cycufile.pxd` that aliases `rocm.bindings.cyhipfile`), exposing the
snake_case functions (`driver_open`, `handle_register`, `buf_register`,
`read`/`write`, the batch/async/stream APIs), the `Descr`/`IOParams`/
`IOEvents` array helpers, the cuFile `IntEnum`s, and a `cuFileError`
exception. Like the `rocm.bindings.hipfile` bindings it builds on, the
module is optional (built only when hipFILE and a loadable
`libhipfile.so` are present). Ships with a `.pyi` stub, an API-reference
page, and a `1_CUDA_Interop/cufile_copy_with_cuda_bindings.py` example.

### Caller-allocated `OUT` scalars stay pointer arguments

Callee-vs-caller allocation is now derived solely from the explicit
`ParmIntent.OUT_CALLEE_ALLOCATED` hint. The Cython layer's additive
rank-0 fallback (`is_out_ptr and ptr_rank == 0`) in
`is_out_callee_allocated_ptr` was removed, so caller-allocated `IN`,
`INOUT`, and `OUT` scalars are all handled the same way — they stay
pointer arguments (a rank-0 `PointerTo*`, a rank-1 `ListOf*`) — and only
an explicit `OUT_CALLEE_ALLOCATED` becomes a synthesized return.

To preserve prior return values, the per-library recipe rules now state
`OUT_CALLEE_ALLOCATED` where the callee genuinely produces the value
(HIP `hipDeviceGetUuid`/`hipIpcGetMemHandle`; hipRTC version/size/handle
OUTs; RCCL `ncclGetUniqueId` and basic-scalar OUTs; hipFFT `workSize`;
hipSPARSE `hipsparseCreate`; amdsmi `_MISTAGGED_OUT` and a shape-aware
`amdsmi_get_*` catch-all). Redundant `T**` handle-creator hardcodes were
removed in favor of the shared `double_indirection_out` chain rule. The
motivating fix: hipFILE's async `hipFileReadAsync`/`hipFileWriteAsync`
`bytes_read_p`/`bytes_written_p` (`ssize_t*`, `@param[out]`, written by
the stream after the call returns) now remain caller-allocated
`PointerToLong` arguments instead of being synthesized as returns. Also
fixed an RCCL `ncclGetUniqueId` intent guard that compared a tuple to a
string and never fired.

### Add `PointerTo*` adapters for rank-0 scalar pointers

Added `PointerToInt` / `PointerToLong` / `PointerToUnsigned` /
`PointerToUnsignedLong` to `rocm.bindings.util.types` — length-1
subclasses of the matching `ListOf*`. The Cython complicated-type handler
now maps a caller-allocated **rank-0** typed scalar pointer (a `T *` at a
single value) to the corresponding `PointerTo*` instead of an opaque
`Pointer`; callee-allocated rank-0 scalar OUTs are unaffected (still bare
scalar returns). Each wrapper adds an ergonomic scalar surface —
`allocate()` defaults to one slot and a `.value` property reads/writes
that slot — so caller-allocated scalar `IN`/`INOUT`/`OUT` arguments (e.g.
hipFILE's async `bytes_read_p`/`bytes_written_p`) can be allocated,
passed, and read back without ctypes plumbing.

### Add `ListOfLong` adapter for signed-`long` buffers

Added a `ListOfLong` wrapper to `rocm.bindings.util.types` (mirroring
`ListOfInt`/`ListOfUnsignedLong`) and a `TypeKind.LONG` branch to the
Cython complicated-type handler. Rank-1 signed-`long` pointer parameters
(`off_t`/`hoff_t`/`ssize_t`/`int64_t`, e.g. hipFILE's `hipFileReadAsync`
/`hipFileWriteAsync` offset and byte-count params, and the hipBLAS/
hipSOLVER `_64` index-result params) now expose a list-constructible
`ListOfLong` instead of a plain `Pointer`, consistent with how `size_t*`
already maps to `ListOfUnsignedLong`.

### Version metadata overhaul

Retired the commit-count-derived version slots. The runtime
source of truth for the ROCm/HIP version is now
`rocm-bindings-core/src/rocm/version.py`, rendered by the code
generator from the tracked `version.py.in` template; it carries
the ROCm/HIP version + commit plus codegen provenance (hip-python
base branch version + git hash and the interfacegen version). The
rendered `version.py` is git-ignored on the codegen base branch
and committed on release branches via the new
`ci/internal/prepare-release.sh`, which also generates `VERSION.in`
(`<rocm_version>.@HIP_PYTHON_VERSION@`). The `[tool.rocm-bindings]`
pyproject sections and the per-package `_version.py.in` templates
were removed; `VERSION`/`__version__` now come from
`importlib.metadata`, and `LONG_VERSION`/`__long_version__` were
dropped.

## \*.\*.\*.\*.70 (2026-05-26)

**Scope.** Summarizes everything on
`dev/docharri/hip-python-codegen-base` not yet on
`origin/amd-integration` (70 commits). The 5th version slot is
the commit-count vs `amd-integration`; the other slots are
placeholders until the broader version scheme lands.

### Package layout overhaul

Renamed `python/` → `packages/`, switched every wheel to the
PEP 621 src-layout, and adopted PEP 639 metadata. The
previously monolithic tree is now seven independently buildable
wheels: `rocm-bindings-core`, `rocm-bindings-hip`,
`rocm-bindings-libraries`, `rocm-bindings-systems` (new),
`rocm-bindings-compiler`, `hip-python-interop`, and the
`hip-python` metapackage (pure-Python alias namespace).

### Build infrastructure

CMake + scikit-build-core became the unified build system
(`packages/CMakeLists.txt`). Each wheel now has its own
`_version.py`, sdist target, and `cmake/HipPythonBuild.cmake`
helper copy so it can build standalone. The cross-package
include-path machinery was simplified — `HIP_PYTHON_GLOBAL_INCLUDE_DIRS`
gave way to a sibling-includes helper that resolves cimports
from each package's `src/` root. A `ci/internal/build-wheels.sh`
driver builds all wheels; ninja is provisioned for per-wheel
builds detached from any outer make jobserver. Auditwheel-repair
support is opt-in via `HIP_PYTHON_AUDITWHEEL_REPAIR=ON`.

### New + reorganized bindings

`amdsmi` lands as the first member of `rocm-bindings-systems`,
which also absorbs `rccl` and `roctx` from the old layout. The
`hipfile` high-level Python API was ported from upstream
`hipfile.python` and lives at `rocm.hipfile`. Library bindings
were sorted into stable / experimental / deferred tiers — the
experimental ones (`hipblaslt`, `hipsparselt`, `hiptensor`,
`hipdnn_backend`) ship behind explicit READMEs; `hsakmt` is deferred.
The `comgr` high-level package migrated to `CStr`, gained
cross-linked docstrings, accepts full enum names, and exposes
`valid_*()` introspection helpers. The `hip-python-interop` wheel
gained three compatibility shims alongside its
`cuda.bindings.{driver,runtime,nvrtc}` modules: a `pynvml` (NVML)
shim backed by `rocm.bindings.amdsmi`, an `nvtx` (NVTX) shim backed
by `rocm.bindings.roctx`, and a minimal `cuda.core.Device` shim
backed by `rocm.bindings.hip`. The `nvtx` shim faithfully backs
markers, ranges, `annotate` and `Profile` on ROCTX; NVTX features
ROCTX cannot express (domains, colors, categories, payloads,
counters) are accepted for source compatibility but degrade to
no-op/dropped. An opt-in compatibility mode
(`HIP_PYTHON_NVTX_COMPAT` env var or `nvtx.set_compat_mode`) can
warn (`NvtxCompatWarning`) or raise (`NvtxCompatError`) when such
unsupported features are used, to help audit portability.

### Cython runtime polish

Generated `cy*` call sites are now wrapped in `with nogil:` for
parallel callability. `util.types.CStr` and `ListOfBytes` carry
a program-lifetime intern dict that pins `str`/`bytes` inputs
for the lifetime of the wrapping object — closing a class of
use-after-free bugs around string ownership. A non-raising
`has_symbol` probe was added on both posix and win32 loader
backends so callers can detect missing entry points without
exception flow. Cython floor bumped to `>=3.1.0` (3.0.x
mis-emits some `*const *` parameters; documented in
`share/design/UPSTREAM_BUGS.md`).

### Docs pipeline

Sphinx input migrated from MyST-Markdown to reStructuredText,
and the extension switched from `sphinx.ext.autodoc` to
`sphinx-autoapi` — autoapi parses the source tree directly
(including generator-emitted `.pyi` stubs), so the docs build
no longer needs the compiled wheels on `sys.path`. The source
tree was renamed `docs/` → `docs_src/` so `docs/` is free for
generated HTML output. The landing page now substitutes
`HIP_PYTHON_GENERATED_*` metadata from
`generated_versions.cmake` (codegen date, ROCm version, source
tree commit hashes). A new `ci/docs/` directory and
`.readthedocs.yaml` provide a thin wrapper for Read the Docs
and local doc builds — they invoke the cmake docs target with
`HIP_PYTHON_BUILD_DOCS_ONLY=ON`, which skips
`hip_python_initialize()` so configure works without `/opt/rocm`.

### Handcoded-stub regeneration

Five handcoded Cython modules (`util.types`, `util.loader`,
`util.posixloader`, `_hip_helpers`, `_hiprtc_helpers`) keep
companion `.pyi` stubs for sphinx-autoapi. A new opt-in
`HIP_PYTHON_ENABLE_STUBGEN=ON` cmake flow drives `stubgen
--include-docstrings` to regenerate them; `ci/docs/regenerate-stubs.sh`
wraps the whole thing as a one-command developer tool.

### Examples + design docs

Examples were reorganized for the new package layout and
extended with single-GPU samples for the new bindings
(`amdsmi`, hipfile, comgr, the new libraries). New
`share/design/` documents — `BINDINGS.md`, `BUILDING.md`,
`CODEGEN.md`, `UPSTREAM_BUGS.md` — describe the generated
bindings shape, the wheel build, the codegen contract, and
tracked upstream bugs respectively. The README gained
experimental-marker conventions and per-wheel install tables.

---

**Cross-reference.** Everything under
`packages/*/src/rocm/bindings/` (the `.pxd`/`.pyx`/`.pyi`
trio per module) and `docs_src/sphinx/_toc.yml.in` is
emitted by **interfacegen** — see that repo's `CHANGELOG.md`
for the generator-side changes that produced the contents
of those files.
