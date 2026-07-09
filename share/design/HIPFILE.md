# hipFILE build, integration, and the errno side-channel

This document describes how [hipFILE](https://github.com/rocm/rocm-systems)
(AMD Infinity Storage — GPU-direct file I/O, cuFile-compatible) is built and
wired into the hip-python **systems** wheel, the dedicated CI stage that
performs that build, the codegen fixes needed to make the `rocm.bindings.hipfile`
bindings compile, and the answer to the `errno` / `hipPeekAtLastError`
"clobbering" question for `hipFileRead` / `hipFileWrite`.

## 1. Why a separate build step

hipFILE is **not** part of a stock ROCm install. The hip-python systems wheel
only builds its hipFILE bindings when hipFILE is discoverable:

```cmake
# packages/rocm-bindings-systems/CMakeLists.txt
find_package(hipfile QUIET)   # gates the cyhipfile / hipfile Cython modules
```

So hipFILE must be **built and installed into `${ROCM_PATH}` before** the wheel
build runs `find_package(hipfile)`. When it is present the systems configure log
prints:

```
-- Found hipfile package — hipfile bindings will be built
```

## 2. Manual build & install

Environment used for validation: AlmaLinux 8.10, ROCm 7.13.0, `gfx90a`,
`amdclang++` + CMake, `/dev/kfd` present.

```bash
# 2.1 Build dependencies
dnf install -y libmount-devel boost-devel
#   libmount-devel  REQUIRED: src/amd_detail/mountinfo.cpp -> <libmount/libmount.h>
#   boost-devel     only for tests/examples (program_options)
#   (Debian/Ubuntu: libmount-dev, libboost-program-options-dev)

# 2.2 Toolchain quirks on the manylinux_2_28 / AlmaLinux 8 family
#   amdclang++ can't find libstdc++/pthread without the gcc-toolset:
export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
#   glibc 2.28 headers predate two symbols hipFILE uses (kernel is new enough):
COMPAT="-DSYS_pidfd_open=434 -DF_SEAL_FUTURE_WRITE=0x0010 -pthread"

# 2.3 Configure + build + install (install prefix defaults to ROCM_PATH)
cmake -S /src/rocm_systems/projects/hipfile -B /src/rocm_systems/projects/hipfile/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_PLATFORM=amd -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DROCM_PATH=/opt/rocm \
  -DCMAKE_CXX_FLAGS="$COMPAT" -DCMAKE_HIP_FLAGS="$COMPAT" -DCMAKE_EXE_LINKER_FLAGS="-pthread" \
  -DBUILD_TESTING=OFF -DAIS_INSTALL_EXAMPLES=OFF
cmake --build /src/rocm_systems/projects/hipfile/build -j"$(nproc)"
cmake --install /src/rocm_systems/projects/hipfile/build
```

Artifacts installed: `${ROCM_PATH}/lib/libhipfile.so*`,
`${ROCM_PATH}/include/hipfile.h`, and the CMake package under
`${ROCM_PATH}/lib/cmake/hipfile/` (this is what `find_package(hipfile)` matches).
`${ROCM_PATH}/bin/ais-check` reports P2PDMA/HIP/amdgpu support.

### The two EL8 compile blockers (why the `COMPAT` flags exist)

| Symbol | Used in | glibc-2.28 header status | Fix |
|---|---|---|---|
| `SYS_pidfd_open` | `src/amd_detail/sys.cpp` | undefined (x86_64 syscall 434) | `-DSYS_pidfd_open=434` |
| `F_SEAL_FUTURE_WRITE` | `src/amd_detail/stats.cpp` | undefined (fcntl seal `0x0010`) | `-DF_SEAL_FUTURE_WRITE=0x0010` |

`std::thread` users (the MT stress tests + the library) additionally need
`-pthread` at compile+link. hipFILE has no blanket `-Werror` (only
`-Werror=switch-enum`), so on newer distros that already define these macros the
redefinition is harmless.

## 3. Rebuilding the systems wheel

Targeted build (core + hip + systems):

```bash
cmake -G "Unix Makefiles" -S packages -B packages/build \
  -DCMAKE_BUILD_TYPE=Release -DROCM_PATH=/opt/rocm -DHIP_PLATFORM=amd \
  -DHIP_PYTHON_BUILD_CORE=ON -DHIP_PYTHON_BUILD_HIP=ON -DHIP_PYTHON_BUILD_SYSTEMS=ON \
  -DHIP_PYTHON_BUILD_LIBRARIES=OFF -DHIP_PYTHON_BUILD_COMPILER=OFF -DHIP_PYTHON_BUILD_INTEROP=OFF \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=packages/build/dist
cmake --build packages/build --target core_wheel hip_wheel systems_wheel -j"$(nproc)"
```

Smoke test:

```python
from rocm.bindings import hipfile, cyhipfile
import rocm.hipfile
from rocm.hipfile import OpError, FileHandleType
from rocm.hipfile.file import FileHandle
```

### 3.1 Codegen fixes required for the hipFILE bindings to compile

The committed hipFILE bindings had **never been compiled** (hipFILE was never
installed in CI), so building them for the first time surfaced several latent
bugs. All fixes are durable — they live in the generator / handcoded sources,
not in the (auto-generated) `.pyx`/`.pxd`:

1. **Cross-module + system type imports** — `hipfile.h` uses `hipStream_t` and
   `hipError_t` (from the HIP runtime), none defined in `hipfile.h`.
   `generate_hipfile` (`tools/hip-python-generate/.../generators_systems.py`)
   injects the cimports, mirroring `generate_rccl`'s `hipStream_t` handling:
   - `c_interface_decl_prolog` (→ `cyhipfile.pxd`): `cimport hipStream_t, hipError_t`
     from `rocm.bindings.cyhip`.
   - `python_interface_impl_prolog` (→ `hipfile.pyx`): a **runtime**
     `from rocm.bindings.hip import hipError_t` (the high-level module constructs
     the Python enum to wrap `hipFileError.hip_drv_err`; a `cimport` of the C
     typedef would shadow the enum and make `hipError_t(...)` a non-callable
     type).
   - The plain POSIX structs `struct sockaddr` (userspace-RDMA fs-op callbacks)
     and `struct timespec` (batch-I/O poll timeout) are referenced by hipFILE
     **by pointer only** — never dereferenced, never used as a by-value field.
     They are therefore hand-declared as **opaque** structs
     (`cdef struct sockaddr: pass` / `cdef struct timespec: pass`) in
     `generate_hipfile`'s `c_interface_decl_prolog`
     (`_HIPFILE_OPAQUE_PTR_TYPES_DECL`), the standard Cython idiom for a
     pointer-only type (same shape as the `ihipStream_t` / `hipArray` opaque
     handles in `cyhip.pxd`). The real definitions come from `hipfile.h`'s
     transitive system includes at C-compile time. This is preferred over
     admitting them through the recipe `node_filter` (which would emit their
     full field layout from the AST): an opaque decl bakes **no** Linux ABI
     (`sa_family` widths, `time_t` size) into the bindings, and the recipe
     `node_filter` stays purely about the `hipFile*` library surface. Keeping
     the decl in the systems-wheel **generator** (not the shared
     `support/recipes/rocm.py` recipe) also localizes this platform-dependent
     provisioning to the one place a future Windows hipFILE build (which
     supplies these via different headers/types) would edit. It further avoids
     Cython's Linux-only `posix.time` cimport (and there is no `posix.socket`
     module at all — Cython ships no `sockaddr` declaration).

2. **Nested anonymous records** — `hipFileDriverProps.nvfs`,
   `hipFileDescr.handle`, and `hipFileIOParams.u/.batch` are anonymous nested
   struct/unions. interfacegen synthesizes `<parent>_struct_N`/`<parent>_union_N`
   field types for them, but a strict-prefix recipe `node_filter` (hipFILE admits
   only `hipFile*`) rejected the nested nodes — their synthesized names
   (`struct_0`/`union_0`) carry no prefix — so the matching `cdef struct`/`union`
   was never emitted while the parent field still referenced it, and Cython
   errored with "not a type identifier". This was **fixed at the root in the
   codegen tool**: `CythonModuleGenerator.walk_filtered_nodes`
   (`interfacegen/.../cython/_backend.py`) now admits a nested record/enum — and
   an inline anonymous function pointer — transitively whenever its **top-most
   enclosing declaration** is admitted (a shared `_topmost_ancestor` walk that
   also generalizes the former immediate-parent `AnonymousFunctionPointer` rule
   to arbitrary nesting depth, e.g. the 2-level `hipFileIOParams.u → batch`). The
   bindings therefore emit the real anonymous names (`hipFileDescr_union_0`,
   `hipFileIOParams_union_0_struct_0`, …), and no header preprocessing or
   per-recipe `_topmost_name` filter is needed. Regression coverage lives in
   `test_nested_record_shapes.py` (strict-prefix nested-anon + funptr-in-nested
   cases) and `test_codegen_gap_named_nested_struct_hoist.py` (the former
   `xfail` gap now passes).

   Note: the deeper "C11 anonymous member with **no field name**" case (e.g.
   HSA's `hsa_amd_memory_copy_op_s`) is a *separate* still-open gap — such
   members are dropped from the parent entirely — and remains blocklisted; see
   the `_CODEGEN_BLOCKLIST` note in `support/recipes/rocm.py`.

3. **Handcoded enum port** — `rocm/hipfile/enums.py` had collapsed the friendly
   `OpError` / `FileHandleType` enums (which `file.py` depends on, e.g.
   `FileHandleType.OPAQUE_FD`) into bare aliases of the generated enums (whose
   members carry raw C names like `hipFileHandleTypeOpaqueFD`). Restored the
   proper `IntEnum` classes, sourcing values from the generated enums.

## 4. Test results (validation host)

`ctest` from the hipFILE build dir (`LD_LIBRARY_PATH=/opt/rocm/lib`):

| Label | Result | Notes |
|---|---|---|
| `unit` (incl. `internal`) | **501 / 501 pass** | no GPU / AIS fs required |
| `stress` (`state_mt`, `batch_mt`) | **2 / 2 pass** | needs `-pthread` |
| `system` | **215 / 219 pass** | 4 failures are all P2PDMA-dependent |

The 4 `system` failures (`spawnedThread{Read,Write}RunsWithoutSegfault`,
`ReadToUnregisteredBufferAtOffset[ReturnsErrorIfOverflow]/Fastpath`) are the
**Fastpath / real-DMA** variants; the Fallback/Compat variants pass. `ais-check`
reports `Kernel P2PDMA support: False` on the validation host, so genuine
GPU→storage DMA cannot run — an environment limitation, not a build defect.

## 5. The CI stage

### 5.1 `ci/internal/build-hipfile.sh`

A standalone, env-driven script (mirrors `build-wheels.sh` conventions) that
performs §2 (deps → configure/build/install → verify + non-fatal `ais-check`).
Key env vars:

| Var | Default | Meaning |
|---|---|---|
| `SRC_DIR` | — | parent of `hip_python/` + the fetched rocm-systems tree |
| `ROCM_PATH` | `/opt/rocm` | install prefix (also the wheel's `find_package` search root) |
| `ROCM_SYSTEMS_DIR` | `${SRC_DIR}/rocm_systems` (falls back to `rocm-systems`) | hipFILE source root |
| `HIPFILE_GPU_TARGETS` | `gfx90a;gfx942;gfx1100` | `CMAKE_HIP_ARCHITECTURES` |
| `MAX_JOBS` | `16` | build parallelism |

It builds with `BUILD_TESTING=OFF` and `AIS_INSTALL_EXAMPLES=OFF` (keeps Boost
out of the CI critical path) and applies the gcc-toolset + EL8 compat flags from
§2 automatically.

### 5.2 Pipeline wiring (`rocm_python.jdp.yaml`)

A new boolean param `ROCM_PYTHON_BUILD_HIPFILE` (default `true`) gates a new
`build-hipfile` stage that runs **before** `build-hip-python`:

```yaml
stages:
- name: build-hipfile
  when:
  - environment: [ ROCM_PYTHON_BUILD_HIPFILE, true ]
  containerCommand: |
    set -xeu
    export ROCM_VERSION=${ROCM_VERSION}
    export ROCM_SYSTEMS_DIR=${SRC_DIR}/rocm_systems
    bash ${SRC_DIR}/hip_python/ci/internal/build-hipfile.sh
- name: build-hip-python
  ...
```

The `rocm_systems` repo is already fetched by this node (`extraGitRepos`), so no
new fetch is needed. Stage ordering guarantees hipFILE is installed into
`${ROCM_PATH}` before `build-wheels.sh` runs `find_package(hipfile)`.

## 6. The `errno` / `hipPeekAtLastError` side-channel

### 6.1 The question

`hipFileRead` / `hipFileWrite` return an `ssize_t`:

```
>= 0 : bytes transferred
-1   : system error — the real cause is in POSIX `errno`
else : negated hipFileOpError_t; if -hipFileHipDriverError, the real cause is
       the HIP driver error from hipPeekAtLastError()
```

Both `errno` and the HIP last-error are **transient thread-local side-channels**.
Can Python code still read them *after* the binding returns, or are they
clobbered? And does the "clobbering" the upstream authors worried about also
affect the hip-python bindings?

### 6.2 Old vs. new bindings

**Upstream `_hipfile.pyx`** captures the side-channel **inside** the `with nogil`
block, immediately after the C call, and returns a 2-tuple:

```cython
with nogil:
    ret = _c.hipFileRead(...)
    if ret == -1:
        extra = errno                      # captured at the C-return boundary
    elif ret == -<int>_c.hipFileHipDriverError:
        extra = <int>_c.hipPeekAtLastError()
return (ret, extra)
```

**The mechanical hip-python emitter** would return only the status (this is
what every *other* generated function does):

```cython
with nogil:
    _cy_hipFileRead__retval = cyhipfile.hipFileRead(...)
return (_cy_hipFileRead__retval,)          # errno / hip_drv_err are dropped
```

The docstring would still say "if -1: check `errno`", but such a wrapper
exposes no way to do so.

**The robust hip-python `hipfile.pyx`** (what we now generate — see §6.4)
captures both side-channels in the *same* `with nogil` block and returns a
3-tuple, mirroring upstream:

```cython
with nogil:
    _cy_hipFileRead__retval = cyhipfile.hipFileRead(...)
    _cy_hipFileRead__err = errno                        # from libc.errno cimport errno
    _cy_hipFileRead__hip_drv_err = <int>hipPeekAtLastError()
return (_cy_hipFileRead__retval, _cy_hipFileRead__err, _cy_hipFileRead__hip_drv_err)
```

### 6.3 Empirical result

`errno` is `*__errno_location()` — thread-local and overwritten by the next libc
call on that thread. A controlled experiment (`ctypes`, `use_errno=True`):

```
close(-1) -> errno captured-at-source = 9 (EBADF)          # reliable
after an intervening libc call, live C errno = 0           # original value gone
```

The value read **at the C-return boundary** is correct (9/EBADF); the value read
**afterwards** through the normal Python path is already gone. The GIL
re-acquire + `PyTuple_New`/allocation that any Cython wrapper performs on the way
out is exactly such an intervening window. **So yes — the clobbering the upstream
authors guarded against applies identically to the hip-python bindings.** A
mechanical 1-tuple wrapper doesn't *clobber-then-mislead* so much as **never
capture** the side-channel: once control returns to Python, `errno` /
`hipPeekAtLastError()` are unrecoverable.

In the Cython bindings the snapshot is taken with `from libc.errno cimport
errno` — the ISO C `<errno.h>` lvalue, declared `nogil`-safe, so it can be read
inside the `with nogil` block with no intervening call. (It is also
cross-platform, unlike Cython's Linux-only `posix.time` — relevant to the
Windows-port theme in this document; Cython ships no `posix.socket`/`sockaddr`
at all.)

### 6.4 The implemented fix

The capture **must** happen at the C-return boundary, in-`nogil`, before the GIL
is re-acquired and before any intervening C call. The mechanical emitter's
`with nogil` block is hardcoded to a single statement, so there is no
in-`nogil` hook to bolt a snapshot onto. Instead, the durable fix lives in the
generator and **overrides the whole rendered body** for just these two
functions:

- **Generic render-override hooks (interfacegen).** `Function` gained two
  optional attributes — `python_interface_impl_override` (short-circuits
  `render_python_interface_impl`) and `python_docstring_override`
  (short-circuits `_render_python_docstring`, so the `.pyi` stub stays in
  lockstep with the `.pyx`) — in
  `tools/interfacegen/.../cython/_function.py`. They are general "render this
  function with a hand-written body/docstring" hooks, not an errno-specific
  emitter change.
- **hipFILE wiring (generator).** `generate_hipfile`
  (`tools/hip-python-generate/.../generators_systems.py`) installs a
  `node_init` that, for the `hipFileRead` / `hipFileWrite` `Function` nodes,
  sets those overrides to a hardcoded body that runs the C call and **both**
  side-channel reads in the same `with nogil` block, returning
  `(retval, errno, hip_drv_err)` (see §6.2). The impl prolog cimports
  `errno` (`libc.errno`) and `hipPeekAtLastError` (`rocm.bindings.cyhip`).
  Scope is the two **sync** `ssize_t` functions only — the async variants
  already return `hipFileError` by value.

**Return shape: tuple, not exception (in the generated layer).** We keep the
generated layer a thin, mechanical value-returner — consistent with the rest of
hip-python (`share/design/BINDINGS.md`), where generated code never translates
error codes into semantic exceptions. The **Pythonic** surface is delivered by
the hand-written consumer: `rocm/hipfile/file.py`'s `_check_io_result` unpacks
`(n, err, drv)` and raises `OSError(err, os.strerror(err))` for `-1` or
`HipFileException(OpError(-n), drv)` for `< -1` (carrying the real HIP driver
error). This avoids a layering inversion (the generated `rocm.bindings.hipfile`
module would otherwise have to construct `HipFileException`, which lives in the
hand-written `rocm.hipfile` package) and sidesteps the `noexcept nogil`
constraint on the cy-level wrapper.

Empirically the loop is closed: `hipfile.hipFileRead(0, 0, 0, 0, 0)` now returns
`(-1, 22, 0)` (EINVAL captured in-`nogil`), and `FileHandle.read`/`write` raise
`OSError(errno=22, "Invalid argument")` instead of a detail-less error.
