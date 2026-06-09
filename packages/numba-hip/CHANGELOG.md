# numba-hip 0.2.0 (30 May 2026)

* **Hard cut to the ROCm 7.13.0+ HIP Python bindings.** numba-hip now
  requires `rocm-bindings-hip`, `rocm-bindings-compiler`, and
  `hip-python-interop` `>=7.13.0` as core dependencies; the legacy
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

