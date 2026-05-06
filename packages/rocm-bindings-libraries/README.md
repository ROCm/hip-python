# rocm-bindings-libraries

Python bindings for ROCm libraries.

This package provides bindings for:
- **hipBLAS** - GPU-accelerated basic linear algebra
- **hipSOLVER** - GPU-accelerated linear algebra solvers
- **RCCL** - ROCm Communication Collectives Library
- **hipRAND** - GPU random number generation
- **hipFFT** - Fast Fourier Transform library
- **hipSPARSE** - Sparse linear algebra routines
- **ROCTX** - ROCm profiling and tracing API

## Installation

```bash
pip install rocm-bindings-libraries
```

## Usage

```python
from rocm.bindings import hipblas, hipsolver, rccl, hiprand, hipfft, hipsparse, roctx

# Use hipBLAS
handle = hipblas.hipblasCreate()
```

## Dependencies

- `rocm-bindings-core` - Common utility types
- `rocm-bindings-hip` - HIP runtime and HIPRTC

## Related Packages

- `hip-python` - Backward-compatible metadata package
