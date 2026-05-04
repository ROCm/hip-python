# hip-python

Backward compatibility package for HIP Python bindings.

This is a metadata package that re-exports modules from the new `rocm.bindings` namespace packages. It maintains compatibility with code written for older versions of hip-python.

## Installation

```bash
# Install with core HIP bindings
pip install hip-python

# Install with all ROCm libraries (hipBLAS, hipSOLVER, etc.)
pip install hip-python[libraries]
```

## Usage

### Old import style (still works)
```python
from hip import hip, hiprtc
from hip import hipblas  # Requires [libraries] extra
```

### New import style (recommended)
```python
from rocm.bindings import hip, hiprtc
from rocm.bindings import hipblas  # Requires rocm-bindings-libraries
```

## Package Structure

The hip-python ecosystem now consists of:
- **rocm-bindings-util** - Utility types and loaders
- **rocm-bindings-hip** - HIP and HIPRTC bindings
- **rocm-bindings-libraries** - hipBLAS, hipSOLVER, RCCL, etc.
- **hip-python** - This backward compatibility package (meta-package)

## Migration Guide

For new projects, prefer the new namespace structure:
- `from rocm.bindings import hip` instead of `from hip import hip`
- Install specific packages (`rocm-bindings-hip`) instead of the meta-package

Existing code continues to work without changes.

## License

MIT License - Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
