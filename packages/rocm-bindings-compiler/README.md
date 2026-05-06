# rocm-bindings-compiler

Python bindings for ROCm compiler infrastructure:
- **LLVM C API bindings**: Comprehensive access to LLVM compiler infrastructure
- **AMD Code Object Manager (Comgr)**: Manipulate code objects for AMD GPUs
- **Clang indexing library**: Parse and analyze C/C++ code

## Installation

```bash
pip install rocm-bindings-compiler
```

## Usage

### LLVM C API

```python
from rocm.bindings.llvm import c

# Get default target triple
triple = c.target.get_default_triple()
print(f"Default target: {triple}")

# Create a module
module = c.core.LLVMModuleCreateWithName(b"my_module")
```

### AMD Comgr

```python
from rocm.bindings import comgr

# Create a data set
data_set = comgr.create_data_set()
```

### Clang

```python
from rocm.bindings.clang import cindex

# Parse a C file
index = cindex.Index.create()
tu = index.parse('hello.c')
```

## Migrated from rocm-llvm-python

This package was previously distributed as `rocm-llvm-python`. The following namespace changes were made:

- `rocm.llvm.*` → `rocm.bindings.llvm.*`
- `rocm.amd_comgr.amd_comgr` → `rocm.bindings.comgr` (simplified)
- `rocm.clang.*` → `rocm.bindings.clang.*`

### Migration Guide

Update your imports:

```python
# Old
from rocm.llvm import c
import rocm.amd_comgr.amd_comgr as comgr
from rocm.clang import cindex

# New
from rocm.bindings.llvm import c
import rocm.bindings.comgr as comgr
from rocm.bindings.clang import cindex
```

## License

MIT License. See LICENSE file for details.
