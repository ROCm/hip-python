# Changelog

Entries are grouped by audience. **Bindings** is what someone importing
`rocm.*` or `cuda.*` sees, **Codegen** is the generator and the artifacts
it emits, and **Building** is the build, the packaging and the docs
pipeline. The generator itself keeps its own log in
[tools/interfacegen/CHANGELOG.md](tools/interfacegen/CHANGELOG.md), and
`numba-hip`, which carries its own version, in
[packages/numba-hip/CHANGELOG.md](packages/numba-hip/CHANGELOG.md).

## 0.1.0

The first version under the real version scheme, so it carries
everything that came before it: the entries previously filed under
`*.*.*.*.70` are folded in here, that slot having been a commit count
standing in until this scheme landed. On a release branch the wheel
version takes the ROCm version as a prefix and ships as
`<rocm_version>.0.1.0`. The `interfacegen` code generator moves to `0.3`
in step, since bindings produced before and after it differ in the
integer widths they pin.

Windows is the headline: HIP Python builds, installs and runs under
MSVC, and `numba.hip` compiles kernels there. The tree is also seven
separately installable wheels rather than one, and the LLVM bindings are
documented for the first time.

### Added

#### Bindings

- **Windows support.** Wheels build under MSVC and load ROCm libraries
  that are not on `PATH`, including across the several trees the
  `rocm_sdk` wheels spread ROCm over. Each package asks your ROCm
  installation what it can build, so a Windows install carries fewer
  bindings than a Linux one; the README and the install chapter say
  which. Interop shims whose backing library ROCm does not ship for
  Windows skip with a reason.
- **`numba.hip` on Windows.** Needs a build with both
  `HIP_PYTHON_BUILD_NUMBA_HIP=ON` and `HIP_PYTHON_BUNDLE_LIBLLVM=ON`,
  because kernels compile through the LLVM bindings. Linux gains from
  the same work: `numba.hip` now loads the runtime on a wheel-only
  install with no ROCm tree present.
- **Seven separately installable wheels** — `rocm-bindings-core`,
  `-hip`, `-libraries`, `-systems`, `-compiler`, `hip-python-interop`
  and the `hip-python` metapackage that aliases the namespace — so you
  can install only the surface you use.
- **`amdsmi` bindings and `rocm.hipfile`.** `amdsmi`, including the ESMI
  CPU-monitoring block, leads the new `rocm-bindings-systems` wheel,
  which also absorbs `rccl` and `roctx`. The `hipfile` high-level API
  ported from upstream lives at `rocm.hipfile`.
- **Three CUDA compatibility shims** in `hip-python-interop`, beside its
  `cuda.bindings.{driver,runtime,nvrtc}` modules: `pynvml` backed by
  `rocm.bindings.amdsmi`, `nvtx` backed by `rocm.bindings.roctx`, and a
  minimal `cuda.core.Device` backed by `rocm.bindings.hip`. NVTX
  features ROCTX cannot express — domains, colors, categories, payloads,
  counters — are accepted for source compatibility and degrade to
  no-ops; set `HIP_PYTHON_NVTX_COMPAT` or call `nvtx.set_compat_mode` to
  have them warn or raise instead.
- **`cuda.bindings.cufile`**, a cuFile-compatible interop module backed
  by AMD's hipFILE, with the snake_case functions, the array helpers,
  the cuFile enums and a `cuFileError`. Built where hipFILE is present.
- **hipBLASLt and hipSPARSELt bindings** are generated again and back in
  `HIP_PYTHON_ALL_LIBRARIES`, their C++-only includes worked around.
  `hiptensor` and `hipdnn_backend` ship as experimental behind their own
  READMEs; `hsakmt` is deferred.
- **An optional bundled LLVM on Windows**, where ROCm ships only static
  archives, so `rocm.bindings.llvm.*` can be used there at all.
  `HIP_PYTHON_BUNDLE_LIBLLVM` stays off by default because it adds 75 MB
  to the `rocm-bindings-compiler` wheel.
- **Scalar and 64-bit buffer adapters** in `rocm.bindings.util.types`:
  `PointerToInt`, `PointerToLong`, `PointerToUnsigned`,
  `PointerToUnsignedLong`, `ListOfLong`, `ListOfInt64` and
  `ListOfUInt64`. The `PointerTo*` ones allocate a single slot by
  default and expose a `.value` property, so a caller-allocated scalar
  argument can be passed and read back without ctypes plumbing.
- **A non-raising `has_symbol` probe** on both loader backends, so a
  missing entry point can be tested for without exception flow.
- **A `JIT Compilation and LLVM IR` chapter** in the user guide, covering
  the `rocm.bindings.llvm.c` modules that no page mentioned before, the
  conventions that differ from the rest of HIP Python (bare return
  values rather than a status tuple, a matching `LLVMDispose*` for every
  object handed out), and four worked examples. New
  `examples/2_Advanced/llvm_optimize_module.py` runs `default<O2>` over
  a module it builds and needs nothing but a system libLLVM.
- **Examples for each new binding**, including a hipSOLVER LU
  factorization and single-GPU samples for `amdsmi`, hipFILE, `comgr`
  and the new libraries.

#### Codegen

- **A stub regeneration flow** for the handcoded Cython modules, opt-in
  through `HIP_PYTHON_ENABLE_STUBGEN=ON` and wrapped as one command by
  `ci/docs/regenerate-stubs.sh`.
- **`share/design/` documents** — `BINDINGS.md`, `BUILDING.md`,
  `CODEGEN.md`, `HIPFILE.md`, `POINTER_ARGUMENTS.md` — on the shape of a
  generated binding, the wheel build, the codegen contract, hipFILE, and
  pointer intent and rank.

#### Building

- **A unified CMake and scikit-build-core build** in
  `packages/CMakeLists.txt`, with an sdist target and a helper copy per
  package so every wheel also builds standalone.
  `ci/internal/build-wheels.sh` builds them all, and auditwheel repair is
  opt-in through `HIP_PYTHON_AUDITWHEEL_REPAIR=ON`.
- **Stable-ABI (abi3) wheels on Windows**, through `-UseSabi 3.11` on
  `ci\internal\build-wheels.ps1` or the `USE_SABI` environment variable
  that already drove the Linux build. Package metadata now advertises
  Python 3.13 and 3.14.
- **A release-preparation step**, `ci/internal/prepare-release.sh`, which
  writes the `VERSION.in` template that prefixes the wheel version with
  the ROCm version on a release branch.

### Changed

#### Bindings

- **Long-running LLVM calls release the GIL**: verification, bitcode and
  IR parsing and writing, linking, pass pipelines, target setup and
  emission, MCJIT, ORC, and all of LTO. A pass pipeline or a JIT
  compilation in one thread no longer blocks the rest of the process.
  Two things to know: on a function returning `void` or a struct by
  value there is no return value left to carry the signal, so a missing
  libLLVM surfaces as an unraisable exception rather than a raised
  `RuntimeError` — probe with `has_symbol` first, as the examples do —
  and sharing an LLVM context, module or builder across threads is still
  yours to serialize, the GIL simply no longer hides it.
- **Caller-allocated `OUT` scalars stay pointer arguments** *(signature
  change)*. Only an explicit callee-allocated hint becomes a synthesized
  return value now, so caller-allocated `IN`, `INOUT` and `OUT` scalars
  are all passed the same way, as a `PointerTo*`. The per-library
  recipes state that hint wherever the callee genuinely produces the
  value, so the return values you already had are unchanged. What it
  fixes: hipFILE's async `bytes_read_p` and `bytes_written_p`, written
  by the stream after the call returns, are caller-allocated arguments
  again rather than returns. Scalar `OUT` pointers in the `bufferSize`
  queries of hipBLASLt, hipSPARSELt and hipSOLVER now come back as
  values.
- **Width-carrying typedefs have one spelling on every platform.** The
  adapter for a pointer argument follows the width its declaration pins
  rather than the width the generating host happens to canonicalize to,
  so hipFILE's offsets look the same under Windows LLP64 and Linux LP64.
  The shipped Linux-generated tree previously allocated 4 bytes on
  Windows for a slot the callee writes 8 into.
- **The generated `cy*` call sites release the GIL**, so they can be
  called in parallel.
- **`CStr` and `ListOfBytes` accept `str` and pin their input** for the
  lifetime of the wrapping object, closing a class of use-after-free
  bugs around string ownership.
- **Examples pass `str`** where a C string is wanted, at roughly 70
  sites that used to carry `b".."` literals and `.encode("utf-8")`.
  Binary payloads stay `bytes`. The user guide gained a `CStr` section
  on the accepted inputs and the pinning contract.
- **An example or test that cannot run here says why and skips**, rather
  than failing at import: missing libraries, no shared LLVM, hipSPARSELt
  without configured kernel libraries.
- **`rocm.comgr` accepts full enum names**, exposes `valid_*()`
  introspection helpers, and carries cross-linked docstrings.
- **Version metadata** comes from `rocm.version` — the ROCm/HIP version
  and commit plus codegen provenance — and `VERSION`/`__version__` come
  from `importlib.metadata`. The commit-count-derived version slots and
  `LONG_VERSION`/`__long_version__` are gone.

#### Codegen

- **`rocm/version.py` is generator-rendered.** The generator fills the
  tracked `version.py.in` with the ROCm/HIP version and commit plus the
  codegen provenance — the hip-python base version and hash, and the
  interfacegen version that produced the tree. It is absent on the
  codegen base branch and committed on release branches.

#### Building

- **The docs build no longer needs the compiled wheels.** Sphinx input
  moved from MyST-Markdown to reStructuredText and from
  `sphinx.ext.autodoc` to `sphinx-autoapi`, which parses the source tree
  and the generated stubs directly. The source tree is `docs_src/` so
  `docs/` is free for HTML output, and `ci/docs/` with `.readthedocs.yaml`
  builds it without `/opt/rocm`.
- **Every wheel uses the PEP 621 src-layout** with PEP 639 licence
  metadata, and cross-package cimports resolve from each package's `src/`
  root instead of a global include path.
- **The Cython floor is `>=3.1.0`**, because 3.0.x mis-emits some
  `*const *` parameters.
- **Build requirement files** moved from `ci/` to `ci/internal/`, next to
  the scripts that define what goes in them.

### Removed

#### Bindings

- **The `hsa` binding**, from codegen, the build and the documentation.
  The binding and module lists in the README, the user guide and the
  systems package were updated to match.

### Fixed

#### Bindings

- **Segfaults in four LLVM bindings** that take an array argument
  (`LLVMFunctionType`, `LLVMGetParamTypes`, `LLVMGetParams`,
  `LLVMRunFunction`): the argument buffer was freed before the call ran,
  so the callee read or wrote memory the allocator had already reused.
- **`NDBuffer` instances no longer carry a Python instance dict**, so
  `vars(buf)` and assigning an arbitrary attribute now raise. Reading
  `buf.__cuda_array_interface__` is unchanged, including through
  `hasattr` and `getattr`, which is how `numba` and CuPy consume it.
- **The dict `NDBuffer.__cuda_array_interface__` returns is a copy.**
  Adding or overwriting an entry — as `numba` does with `'strides'` —
  no longer alters the buffer it came from. Use
  `NDBuffer.configure` to change the buffer itself.
- **Building an `NDBuffer` from another object's CUDA array interface
  works**, where it previously raised `TypeError` for every such input.
- **An interface that spells out contiguous strides is accepted.**
  Previously any `'strides'` other than `None` was refused, including
  the explicit C-contiguous strides `numba` hands out. A genuinely
  non-contiguous layout still raises `RuntimeError`, and a mask or a
  non-zero offset — neither of which an `NDBuffer` can represent — now
  raises `NotImplementedError` instead of being ignored.
- **An invalid `stream` passed to `NDBuffer.configure` raises.** It used
  to return the `ValueError` as the call's result, so the caller
  received an exception object in place of the buffer.

#### Codegen

- **The stubs the documentation is built from cannot drift quietly.**
  Every generated stub opens with an `AUTO-GENERATED` banner, the
  developer's Cython version no longer leaks into the output, a stub
  taken from a limited-API build is refused, and a stub stubgen skips is
  an error rather than a silent success.
