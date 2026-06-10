# rocm-bindings-compiler

Python bindings for the ROCm compiler stack: the LLVM-C API, the AMD Code Object
Manager (COMGR), and the Clang indexing library.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Install

```bash
pip install rocm-bindings-compiler    # or: pip install hip-python[compiler]
```

## Usage

```python
from rocm.bindings.llvm import c          # LLVM-C API
from rocm.bindings import amd_comgr        # COMGR C bindings
import rocm.comgr                          # high-level COMGR wrapper
from rocm.bindings.clang import cindex     # Clang indexing
```

> This package was previously distributed as `rocm-llvm-python`; its modules
> moved under the `rocm.bindings.*` namespace (e.g. `rocm.llvm` ->
> `rocm.bindings.llvm`, `rocm.amd_comgr.amd_comgr` -> `rocm.bindings.amd_comgr`).

## Dependencies

- `rocm-bindings-core`
