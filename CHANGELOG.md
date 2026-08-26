# Changelog

## Unreleased

### Stable-ABI wheels on Windows

The abi3 build was reachable on Windows only by hand-passing
`-DHIP_PYTHON_ABI3_FLOOR` through `-ExtraCMakeArgs`, where Windows
PowerShell 5.1 splits the unquoted `3.11` into two arguments and CMake
then rejects the floor as `'3'`. `ci\internal\build-wheels.ps1` now takes
`-UseSabi 3.11` and, like the bash script, reads `USE_SABI` from the
environment, so one CI variable drives both platforms. It rejects a
malformed floor, and a floor below 3.11, before the toolchain is probed —
below 3.11 the stable ABI has no buffer protocol and the build would
otherwise fail deep inside the C compiler on `types.pyx`.

Verified end to end on Windows: `rocm-bindings-core` and
`rocm-bindings-hip` built at floor 3.11 under Python 3.14 with MSVC, each
module linking `python3.dll` alone, then installed into 3.12 and 3.13
environments where a host↔device `hipMemcpy` roundtrip ran on gfx1103.
`share/design/BUILDING.md` gained a "Stable-ABI (abi3) wheels" section
covering the floor rules, how the CMake side is wired, and the two
Windows specifics that make it work: `python3.lib` ships with every
CPython install, and CMake leaves the module names free of the `.abi3`
infix that Windows cannot import.

Package metadata now also advertises Python 3.13 and 3.14, which the
classifiers had stopped short of at 3.12.

Unrelated tidying in the same area: `requirements-build.txt` and
`requirements-test.txt` moved from `ci/` into `ci/internal/`, next to the
scripts that define what goes in them. Windows builds now start with
`pip install -r ci\internal\requirements-build.txt`.

### Document the LLVM bindings

The LLVM-C bindings were the largest undocumented part of the user guide:
27 modules and three runnable examples that no page mentioned. They now
form the second half of the JIT chapter, which is retitled
`JIT Compilation and LLVM IR` and opens by naming all three compiler
entry points and when to reach for each: HIPRTC and `rocm.comgr` when the
goal is a kernel running on the GPU, `rocm.bindings.llvm.c` when the goal
is the IR itself.

The LLVM half covers how to tell whether the bindings are usable —
importing a module touches no library, so a missing one surfaces as a
`RuntimeError` at the first call, and `has_symbol` answers the question
up front, including for the `LLVMInitializeAll*` entry points that not
every `libLLVM` exports. It then covers how header names map to module
names and the calling conventions that differ from the rest of HIP Python:
LLVM returns bare values rather than a status tuple, and every object it
hands out has a matching `LLVMDispose*`. Four worked examples follow —
listing targets, reading bitcode, building and running a module, and
running a pass pipeline — and a closing section on passing bitcode back to
HIPRTC and COMGR. Where a shared LLVM comes from stays with the install
and build-from-source chapters, which the section points at.

`examples/2_Advanced/llvm_optimize_module.py` is new: it builds a function
with a stack slot, runs `default<O2>` over it through `LLVMRunPasses`, and
prints the IR before and after. It passes `None` for the target machine, so
it needs no `LLVMInitializeAll*` and runs against a plain system libLLVM.
`list_targets.py`, `parse_llvm_bitcode.py` and `execution_engine_sum.py`
gained named `literalinclude` markers so the chapter embeds tested code.

### Fix use-after-free of array arguments in the LLVM bindings

An argument adapter constructed inline in the C call expression was freed
before the call ran: Cython drops the intermediate as soon as
`getPtr()` returns, which is the line *before* the call in the generated
C. A `ListOfPointer` holding a two-element `void *` array therefore
reached `LLVMFunctionType` after glibc had overwritten both slots with
tcache bookkeeping, and the first `LLVMGetParam` on the resulting
function type segfaulted. Four call sites took a malloc'd array this
way: `LLVMFunctionType`, `LLVMGetParamTypes` and `LLVMGetParams` in
`rocm.bindings.llvm.c.core`, and `LLVMRunFunction` in
`rocm.bindings.llvm.c.executionengine` (the two `LLVMGetParam*` ones
wrote their output into the freed buffer). All four are LLVM bindings
because those are the only ones produced by the with-GIL emitter, which
was the emitter that inlined the chain on the theory that holding the
GIL kept the temporary alive. It does not.

Pre-call hoisting is now unconditional and identical in both emitters:
every Python-derived argument is bound to a named local, so the adapter
is still referenced when the C call executes. The with-GIL emitter now
differs from the with-nogil one only in having no `with nogil:` block
and keeping the return-value wrap inline. Regression coverage lives in
`test_codegen_wrapper_arg_lifetime.py`; the lifetime contract for
argument buffers is written up in `share/design/POINTER_ARGUMENTS.md`
section 10 and in the `ListOf*` docstrings.

### Examples pass `str` instead of byte strings

`CStr` and `ListOfBytes` have accepted `str` (and interned it) for a
while, but the examples still carried `b".."` literals and
`.encode("utf-8")` calls from before that. Roughly 70 sites across 16
examples now use plain `str`: LLVM module/function/block/value names,
target triples and CPU/feature strings, file paths, hipRTC program and
kernel names, compiler flag lists, and HIP source strings. `gcnArchName`
is decoded once at the assignment, which also fixes the
`Compiling kernel for b'gfx...'` output. Binary payloads stay `bytes`:
disassembler inputs, and every IR/bitcode/HSA image whose `len()` is
passed as a byte size to `hiprtcLinkAddData`, `hipModuleLoadData` or
`LLVMCreateMemoryBufferWithMemoryRange`. The user guide gained a `CStr`
section covering the accepted inputs and the pinning contract.

### Windows support: build, load, and run under MSVC

HIP Python builds, installs, and runs on Windows. The build system gained
an MSVC path: a Release default (CMake's MSVC module defaults to Debug,
MSVC then defines `_DEBUG`, and `pyconfig.h` reads that as a request for
the debug CPython ABI a release interpreter cannot link against), and
compile probes in place of platform guesswork. Each package now asks the
ROCm installation what it can build — `hip_python_filter_available_modules`
compiles a probe per module with that module's own defines — instead of
deciding from the platform and version. `rocm-bindings-systems` can come
up empty, which the packaging targets tolerate rather than ship a wheel
advertising bindings it does not carry, and `hip-python-interop` no
longer depends on it unconditionally.

The loader learned Windows' rules. Loading by absolute path failed with
error 126 whenever ROCm was not on `PATH`, because `LoadLibraryA` does
not search the directory of the library it is loading: `hipfft.dll` could
not see `rocfft.dll` next to it. The loader now passes
`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` together with `DEFAULT_DIRS` (naming
any search flag replaces the standard search order rather than adding to
it), and registers every ROCm tree with `os.add_dll_directory`, holding
the handles — the `rocm_sdk` wheels spread ROCm over several trees, with
the math libraries importing the HIP runtime from `rocm-sdk-core`. Path
resolution also finds libraries `rocm_sdk` ships without registering, and
resolves clang's resource directory next to the libclang it loaded.
Pointer arithmetic moved to `uintptr_t`, and the compiler bindings'
`ctypes` widths were corrected for LLP64, where `long` is 4 bytes.

The code generator itself runs on Windows, and `ci/internal/build-wheels.ps1`,
`ci/internal/test.ps1`, and `ci/internal/env-rocm.ps1` build and test
there (the last bootstraps MSVC when `VSCMD_VER` is inherited without the
matching `PATH`). Interop shims whose backing library ROCm does not ship
for Windows skip with a reason. The README and the user guide describe
what a Windows install offers and how to build it.

### Bundle a shared LLVM on Windows

ROCm ships no shared LLVM for Windows, only static archives, so
`rocm.bindings.llvm.*` imported and then raised on first call. Bundling
one the way Linux does cannot work directly: a PE image exports only what
its objects marked for export, and these archives were compiled with
LLVM's export annotations off (`llvm/Config/llvm-config.h` leaves
`LLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS` undefined), so `/WHOLEARCHIVE`
produces a DLL with no export table at all.

`gen_msvc_exports.py` names the symbols instead: it reads the archives
with `llvm-nm` and writes a module-definition file listing every
unmangled `LLVM*` symbol they define, plus the `LLVMInitializeAll*`
wrappers that ship `static inline` in `<llvm-c/Target.h>`. The C API is
the subset worth naming — it is what the bindings call, it is stable
across toolsets, and unlike the full symbol table it fits a PE export
table. Listing a symbol pulls in the object that defines it and `/OPT:REF`
drops the rest, so 0.55 GB of archives link in seconds into a 75 MB DLL.
Three Windows-only `llvm-config` traps are handled along the way: space-
separated `--libfiles` paths (addressed as `--libnames` joined onto
`--libdir`), `separate_arguments(... UNIX_COMMAND)` eating the
backslashes in `-LIBPATH:`, and `--system-libs` omitting the zstd archive
`LLVMSupport` references as unconditionally as the zlib one it does
report.

`HIP_PYTHON_BUNDLE_LIBLLVM` stays off by default on Windows because it
adds 75 MB to the `rocm-bindings-compiler` wheel — a size decision now
rather than a platform one. With it, the five examples that need a
loadable LLVM pass, `execution_engine_sum.py` included, which JITs and
runs code through MCJIT.

### `numba.hip` compiles and runs kernels on Windows

`numba.hip` refused Windows outright, in a branch of
`locate_runtime_and_loader` that raised `NotImplementedError`. Nothing
behind it was fundamentally Unix-bound. The runtime and the clang
resource directory now resolve through `rocm.bindings.util.paths`, which
knows every installation the bindings support — that is what makes the
version-suffixed `amdhip64_7.dll` resolve, and it lifts a Unix
limitation too, since the previous path went through
`hipconfig.get_rocm_path`, which raises when there is no ROCm tree, so a
wheel-only install could not load the runtime at all. The libclang
directory override looked for `libclang.so*`, a pattern that cannot match
on Windows. The cache directory was named after `os.getuid()`; the
account name serves the same purpose where there is no numeric ID. The
device-library wrappers were named after `cursor.mangled_name`, which
follows the parse target's ABI, so on Windows every wrapper came back
MSVC-mangled and the generated HIP source failed to parse — device code
is Itanium-mangled whatever the host is, so the parse now names a target
that agrees. Finally, hipRTC took the full path of a linked HIP source as
its program name, which a drive letter and backslashes make it reject
outright with an empty log.

`amdgcn.optimize_module` no longer isolates the pass run in a forked
child (Windows has no fork). The child existed because `LLVMRunPasses`
was said to abort rather than return an error, but what aborts is the
verifier, and `LLVMVerifyModule` reports the same defect as a return
value with the same message in this process. The module is verified
before the passes run, so a caller who hands over a broken module learns
what is wrong with it instead of reading an exit code. Removing the child
also removes an interpreter start and a bitcode round trip per call, and
lets an `LLVMOpaqueModule` argument be optimized in place again, as its
documentation always said. `numba-hip` also initializes its typing
context lazily in the target descriptor.

Building a working `numba.hip` on Windows needs
`HIP_PYTHON_BUNDLE_LIBLLVM=ON` alongside `HIP_PYTHON_BUILD_NUMBA_HIP=ON`,
since it compiles its kernels through the LLVM bindings.

### Pick the pointer adapter by the width a declaration pins

The complicated-type handler chose its wrapper from the innermost
canonical clang `TypeKind` while the C type in the same signature was
rendered from the preserved typedef name. The two disagree exactly where
the typedef exists to hide a platform difference: `size_t` canonicalizes
to `ULONG` on LP64 and `ULONGLONG` on LLP64, so a Linux run picked
`PointerToUnsignedLong` and a Windows run matched no branch at all — and
the tree that ships is the Linux one, whose wrapper allocates 4 bytes on
Windows for a slot the callee writes 8 into. The width a typedef pins is
now the answer everyone reads: `FIXED_WIDTH_INT_SPECS` gives each typedef
its signedness and bit width, the renderer takes the spelling and the
handler takes the spec, and a module can alias its width-carrying
typedefs to the `stdint` names. Both per-module maps are validated when
the backend is built.

`rocm.bindings.util.types` gained `ListOfInt64` and `ListOfUInt64` with
their `PointerTo*` specializations, elements typed `int64_t`/`uint64_t`
so the width is identical wherever the extension is compiled.
`ListOfLong` and `ListOfUnsignedLong` stay for declarations that
genuinely say `long`. hipFILE's offsets now have one spelling on every
platform, and the `cuda.bindings.cufile` interop names the hipFILE
integer types it passes through.

### Disable the `hsa` binding

The `hsa` binding is no longer generated: its registration in
`AVAILABLE_GENERATORS` is commented out, which removes it from codegen,
from the build, and from the documentation in one step. Binding and
module lists across the README, the user guide, and the systems package
were updated to match.

### Re-enable the hipBLASLt and hipSPARSELt bindings

Both public C API headers pull in C++-only includes. The codegen now
emits patched shim headers and the `shim_includes` wiring puts them on
the compile include path, so the bindings compile as C again and both
libraries are back in `HIP_PYTHON_ALL_LIBRARIES`. Supporting fixes: the
hipBLASLt shim rewrites C++ member initializers, non-integer hipBLASLt
macros no longer leak into the binding export, a rocRAND C-mode `uint4`
fallback is shimmed, hipBLASLt and hipSOLVER only import the hipBLAS
declarations they need, the `__has_symbol` DLL helper is module-tagged,
`timespec` is admitted for amdsmi's by-value telemetry fields, duplicate
top-level record and enum declarations are deduplicated, self-referential
struct fields resolve their typeref, and ROCm 7.14.0's hipify-perl
mapping expressions parse. Scalar `OUT` pointers in hipBLASLt,
hipSPARSELt, and hipSOLVER's `bufferSize` queries come back as values.

### Examples and tests report why they cannot run

An example that cannot run in the current environment now says so and
skips, rather than failing at import: missing libraries, an absent
shared LLVM, hipSPARSELt without configured kernel libraries. The
examples suite gained a hipSOLVER LU factorization example, wired in the
hipBLASLt and hipSPARSELt examples with `has_symbol` probes, fixed the
hipBLASLt GEMM heuristic allocation and layout, accepts either spelling
of the JIT input constants, declares its PyYAML dependency, and imports
`ctypes` in `hip_jacobi`. CI runs pytest with `-rs` so every skip
reason appears in the log. `rocm.comgr`, `rocm.hipfile`, the `pynvml`,
`nvtx`, and `cuda.core` shims, and `cuda.bindings.cufile` render their
docstrings as definition lists with normalized cross-references.

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
`share/design/BUILDING.md` under "Cython version requirement").

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
