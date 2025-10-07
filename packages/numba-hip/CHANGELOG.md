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

