# rocm-bindings-systems

Python bindings for ROCm system-level libraries.

This package provides bindings for:

- **RCCL** - ROCm Communication Collectives Library (multi-GPU
  communication primitives modeled on NCCL).
- **ROCTX** - ROCm profiling and tracing instrumentation API.

The math/FFT/random/sparse libraries (hipBLAS, hipSOLVER, hipRAND,
hipFFT, hipSPARSE) live in the sibling `rocm-bindings-libraries`
wheel.

## Installation

```bash
pip install rocm-bindings-systems
```

Or via the `hip-python` metapackage's optional extra:

```bash
pip install hip-python[systems]
```

## Usage

```python
from rocm.bindings import rccl, roctx

# RCCL: collective communication
nccl_id = rccl.ncclGetUniqueId()

# ROCTX: tracing markers
roctx.roctxRangePush("my-region")
# ... compute work ...
roctx.roctxRangePop()
```

## Dependencies

- `rocm-bindings-core` - Common utility types
- `rocm-bindings-hip` - HIP runtime and HIPRTC

## Related packages

- `rocm-bindings-libraries` - Math libraries (hipBLAS, hipFFT, ...)
- `hip-python` - Top-level package that re-exports `rocm.bindings.*`
  under the legacy `hip.*` namespace.
