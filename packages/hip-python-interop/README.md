# hip-python-interop

CUDA interop layer for HIP: `cuda.bindings.{driver,runtime,nvrtc}` implemented on
top of ROCm/HIP, so CUDA Python code can run on AMD GPUs.

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

For native HIP bindings use `rocm-bindings-hip` instead.

## Dependencies

- `rocm-bindings-hip`
