# rocm-bindings-hip

HIP and HIPRTC Python bindings for ROCm.

This package provides Python bindings for:
- **HIP Runtime API** - GPU kernel execution, memory management, device control
- **HIPRTC** - Runtime compilation of HIP/CUDA kernels

## Installation

```bash
pip install rocm-bindings-hip
```

## Usage

```python
from rocm.bindings import hip, hiprtc

# Check HIP version
print(hip.hipRuntimeGetVersion())

# Allocate device memory
ptr = hip.hipMalloc(1024)
```

## Dependencies

- `rocm-bindings-core` - Common utility types and loaders

## Related Packages

- `rocm-bindings-libraries` - Additional ROCm libraries (hipBLAS, hipSOLVER, etc.)
- `hip-python` - Backward-compatible metadata package
