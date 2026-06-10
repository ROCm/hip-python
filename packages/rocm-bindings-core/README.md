# rocm-bindings-core

Shared foundation for the ROCm Python bindings: common type definitions, the
POSIX shared-library loader, and ROCm path resolution.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install rocm-bindings-core
```

## Usage

```python
from rocm.bindings.util import types, posixloader
```

This is a low-level package, normally pulled in automatically as a dependency of
the other `rocm-bindings-*` packages.

## Dependencies

- None — this is the base package the other bindings depend on.
