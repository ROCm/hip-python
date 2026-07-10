# hip-python-interop

CUDA interop layer for HIP: `cuda.bindings.{driver,runtime,nvrtc,cufile}`
implemented on top of ROCm/HIP, so CUDA Python code can run on AMD GPUs.

The `cuda.bindings.cufile` module mirrors NVIDIA's GPUDirect Storage (GDS)
cuFile Python API on top of AMD's hipFILE. It is built only when the `hipfile`
package is available; the accompanying `cuda.bindings.cycufile` provides the
C-level (`cimport`-only) declarations.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install hip-python-interop
```

## Usage

```python
from cuda.bindings import driver, runtime, nvrtc

driver.cuInit(0)
```

GPUDirect Storage (cuFile) example:

```python
from cuda.bindings import cufile

cufile.driver_open()
descr = cufile.Descr()
descr.type = cufile.FileHandleType.OPAQUE_FD
descr.handle.fd = fd  # an O_DIRECT-opened file descriptor
fh = cufile.handle_register(descr.ptr)
# ... cufile.buf_register / cufile.write / cufile.read ...
cufile.handle_deregister(fh)
cufile.driver_close()
```

For native HIP bindings use `rocm-bindings-hip` instead (or `rocm.hipfile` for
the Pythonic hipFILE wrapper).

## Dependencies

- `rocm-bindings-hip`
- `rocm-bindings-systems` (backs the bundled `pynvml` shim via
  `rocm.bindings.amdsmi`, the `nvtx` shim via `rocm.bindings.roctx`, and the
  `cuda.bindings.cufile` shim via `rocm.bindings.cyhipfile`)
