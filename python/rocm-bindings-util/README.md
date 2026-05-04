# rocm-bindings-util

Utility types and loaders for ROCm Python bindings.

This package provides:
- Common type definitions used across ROCm bindings
- POSIX library loading utilities for runtime library resolution

## Installation

```bash
pip install rocm-bindings-util
```

## Usage

```python
from rocm.bindings.util import types, posixloader
```

This is a low-level package typically used as a dependency by other ROCm binding packages.
