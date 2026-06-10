# rocm-bindings-libraries

Python bindings for the ROCm math libraries: hipBLAS, hipBLASLt, hipSOLVER,
hipRAND, hipFFT, hipSPARSE (and other experimental math libraries).

The system-level libraries (RCCL, ROCTX, amdsmi, HSA) live in the sibling
`rocm-bindings-systems` package.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install rocm-bindings-libraries   # or: pip install hip-python[libraries]
```

## Usage

```python
from rocm.bindings import hipblas, hipsolver, hiprand, hipfft, hipsparse

handle = hipblas.hipblasCreate()
```

## Dependencies

- `rocm-bindings-core`
- `rocm-bindings-hip`
