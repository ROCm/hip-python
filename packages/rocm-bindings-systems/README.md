# rocm-bindings-systems

Python bindings for ROCm system-level libraries: RCCL (collective
communication), ROCTX (profiling/tracing), amdsmi (system management), and
hipFILE (accelerated file I/O, also available as the higher-level
`rocm.hipfile`).

The math libraries (hipBLAS, hipSOLVER, hipRAND, hipFFT, hipSPARSE, ...) live in
the sibling `rocm-bindings-libraries` package.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install rocm-bindings-systems    # or: pip install hip-python[systems]
```

## Usage

```python
from rocm.bindings import rccl, roctx

roctx.roctxRangePush("my-region")
# ... compute work ...
roctx.roctxRangePop()
```

## Dependencies

- `rocm-bindings-core`
- `rocm-bindings-hip`
