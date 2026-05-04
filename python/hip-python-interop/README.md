# hip-python-interop

CUDA API bindings for HIP interoperability.

This package provides Python bindings for CUDA APIs when running on ROCm/HIP:
- **cuda.bindings.driver** - CUDA Driver API
- **cuda.bindings.runtime** - CUDA Runtime API
- **cuda.bindings.nvrtc** - NVIDIA Runtime Compilation (NVRTC)

These bindings allow CUDA code to run on AMD GPUs through HIP's CUDA compatibility layer.

## Installation

```bash
pip install hip-python-interop
```

## Usage

```python
from cuda.bindings import driver, runtime, nvrtc

# Use CUDA driver API on AMD GPUs via HIP
result = driver.cuInit(0)
```

## Note

This package is for CUDA-to-HIP interoperability. For native HIP bindings, use:
- `rocm-bindings-hip` - Native HIP bindings
- `hip-python` - Backward compatibility package

## License

MIT License - Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
