# numba-hip 0.2.1 (31 Jul 2026)

* **Windows support.** `numba.hip` compiles and runs kernels on Windows.
  The HIP runtime and clang's resource directory are resolved through
  `rocm.bindings.util.paths` instead of hand-built Unix paths, which also
  lets a wheel-only install (no ROCm tree) find the runtime at all; the
  libclang directory override looks for the DLL names; the cache
  directory falls back to the account name where `os.getuid()` does not
  exist; the device-library wrappers are parsed against an
  Itanium-mangling target, since device code is Itanium-mangled whatever
  the host is; and hipRTC program names are derived from the dependency
  rather than passed a full path with a drive letter. Requires a
  `rocm-bindings-compiler` built with `HIP_PYTHON_BUNDLE_LIBLLVM=ON`,
  since kernels are compiled through the LLVM bindings.
* `amdgcn.optimize_module` no longer runs the LLVM passes in a forked
  child. It verifies the module with `LLVMVerifyModule` first — the same
  defect the verifier would abort on is reported as a return value — so
  the abort has no occasion to happen. An `LLVMOpaqueModule` argument is
  optimized in place again, and the per-call interpreter start plus
  bitcode round trip are gone (about 0.3 ms for a small module).
* Adopt the PEP 621 src-layout (`src/numba/hip`), relocate the test
  suite out of the importable package to the repo-root `tests/numba-hip`,
  and fold the build/test scripts into the top-level `ci/internal`. The
  unified `all_wheels` CMake target builds the `numba-hip` wheel, so
  there is no separate numba-hip build stage.
* Allow numba 0.63 (Python 3.14 support) and lower the binding floor to
  ROCm 7.2.3.
* Delegate libclang lookup to the shared `rocm.bindings` resolver.
* Initialize the typing context lazily in the target descriptor.
* `hipdrv/driver.py` replaces the deprecated context APIs with the
  device-based model; `hipdevicelib` demotes allow-listed device-library
  symbols to `linkonce_odr` and makes `_setup_libclang()` idempotent.
* Documentation: README converted to Markdown, project URLs point at the
  hip-python monorepo.

# numba-hip 0.2.0 (30 May 2026)

* **Hard cut to the new HIP Python bindings (ROCm 7.2.3+).** numba-hip now
  requires `rocm-bindings-hip`, `rocm-bindings-compiler`, and
  `hip-python-interop` `>=7.2.3` as core dependencies; the legacy
  per-ROCm `[project.optional-dependencies]` extras (which pinned the
  old `hip-python` / `hip-python-as-cuda` / `rocm-llvm-python`
  packages) have been removed.
* Migrate to the new binding namespaces: `rocm.bindings.*` (e.g.
  `rocm.bindings.hip`, `rocm.bindings.hiprtc`), `rocm.comgr`, and
  `rocm.bindings.util.types`.
* Port `from hip import hiprtc` to `from rocm.bindings import hiprtc`,
  and route the link-options helper through `rocm.bindings.hiprtc_pyext`
  (the old `hiprtc.ext` shim no longer exists).
* Remove legacy-binding compatibility duck typing in `hipdrv/driver.py`
  and fix the undefined module-global `_hip` reference (now bound to
  `rocm.bindings.hip`).

# numba-hip 0.1.6 (09 Feb 2026)

* Add fallback option to use AMD-SMI for UUID detection.
* Extend ROCm support metadata with keys for ROCm 7.1.1 and 7.2.0.
* Documentation: update tested GPUs and supported Numba versions.

# numba-hip 0.1.5 (29 Jan 2026)

## Bug Fixes

* Fix `__array__()` method for NumPy 2.0 compatibility.
* Add ROCm 7.2 data layout to `test_amdgcn`.

# numba-hip 0.1.4 (09 Dec 2025)

* Add Python 3.13 compatibility.
* Vendor HIP sigutils and switch HIP modules to use it.
* Move `devicefunc` and `dummyarray` into numba-hip.
* hipdrv: remove `modulerepl*`, use local module files.
* Documentation: extend support note with tested RDNA3/4 cards.
* Developer tooling: add `license_verificator` to pre-commit hooks.

## Bug Fixes

* Fix HIPDispatcher specialization typing to use `typeof_pyval`.
* tests(hipdrv): fix empty-slice assertions to validate array size.

# numba-hip 0.1.3 (07 Oct 2025)

* Compatiblity with ROCm 7.0.*.
* Cleaner and shorter LLVM IR output, smaller code object sizes
* HSA assembly inspection is now supported for specialized Numba HIP kernels

## Bug Fixes

* Remove non-kernel (device) function symbols from generated code object via 
  visibility attribute.
* Specify correct function attributes (only: target-cpu, target-features) for Numba HIP device functions
  and kernels. Fix situation where no attributes were set for device function.
  Fix that wrong attributes were assigned to kernel and device functions.

