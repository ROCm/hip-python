# rocm-bindings-hip

Python bindings for the HIP runtime API and HIPRTC (runtime compilation of
HIP/CUDA kernels).

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install rocm-bindings-hip
```

## Usage

```python
from rocm.bindings import hip, hiprtc

print(hip.hipRuntimeGetVersion())
```

## Dependencies

- `rocm-bindings-core`
