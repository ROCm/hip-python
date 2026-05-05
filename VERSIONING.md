# Version Management

## Overview

All hip-python packages use a centralized version scheme managed from the repository root.

## Version Format

```
7.13.0.563.<short-version>
```

Where:
- `7.13.0` - ROCm version
- `563` - Build number
- `<short-version>` - Git commit info or custom identifier

## Files

### VERSION.in (Repository Root)
Template file containing:
```
7.13.0.563.@HIP_PYTHON_VERSION_SHORT@
```

### VERSION (Generated)
CMake processes `VERSION.in` to generate `VERSION` by replacing `@HIP_PYTHON_VERSION_SHORT@` with:
1. Value of `HIP_PYTHON_VERSION_SHORT` environment variable (if set), OR
2. Output of `git describe --tags --always --dirty`, OR
3. `"dev"` as fallback

Example generated VERSION:
```
7.13.0.563.abcd1234-dirty
```

## Build Process

### Top-level Build (python/CMakeLists.txt)

1. Determines `HIP_PYTHON_VERSION_SHORT`
2. Configures `VERSION.in` → `VERSION`
3. Reads `VERSION` into `HIP_PYTHON_VERSION_FULL`
4. Sets variables for subprojects:
   - `HIP_PYTHON_VERSION_NAME`
   - `HIP_PYTHON_LONG_VERSION_NAME`
5. Passes to all subprojects

### Subproject Builds

Each package's CMakeLists.txt:
1. Receives version variables from parent (unified build), OR
2. Should handle standalone builds (TODO: add version discovery to each package)

### Python Package Metadata

All `pyproject.toml` files use dynamic versioning from the `VERSION` file:

**For scikit-build-core packages** (rocm-bindings-*, cuda/):
```toml
[project]
dynamic = ["version"]

[tool.scikit-build]
metadata.version.provider = "scikit_build_core.metadata.regex"
metadata.version.input = "../../../VERSION"  # or "../VERSION" for cuda
```

**For setuptools packages** (hip-python):
```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {file = "../VERSION"}
```

## Setting Custom Version

### Via Environment Variable
```bash
export HIP_PYTHON_VERSION_SHORT="1.2.3-custom"
cd python
pip install -e .
```

### Via CMake
```bash
cd python
cmake -DHIP_PYTHON_VERSION_SHORT="1.2.3-custom" -B build
```

## Version Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Repository Root                                             │
├─────────────────────────────────────────────────────────────┤
│ VERSION.in: 7.13.0.563.@HIP_PYTHON_VERSION_SHORT@          │
│      ↓ (CMake configure)                                    │
│ VERSION: 7.13.0.563.abcd1234                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ CMake        │  │ CMake        │  │ Python       │
│ (build)      │  │ (build)      │  │ (packaging)  │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ _version.py  │  │ _version.py  │  │ pyproject.   │
│   .in        │  │   .in        │  │   toml       │
│      ↓       │  │      ↓       │  │      ↓       │
│ VERSION_NAME │  │ VERSION_NAME │  │ Reads        │
│      ↓       │  │      ↓       │  │ VERSION      │
│ _version.py  │  │ _version.py  │  │ file         │
└──────────────┘  └──────────────┘  └──────────────┘
  (installed)       (installed)       (wheel)
```

## Package Versions

All packages share the same version from the root `VERSION` file:

- `rocm-bindings-core` - 7.13.0.563.xxx
- `rocm-bindings-hip` - 7.13.0.563.xxx
- `rocm-bindings-libraries` - 7.13.0.563.xxx
- `hip-python-interop` - 7.13.0.563.xxx
- `hip-python` - 7.13.0.563.xxx

This ensures all packages in a build are version-synchronized.

## Runtime Version Access

### From Python
```python
# Via package metadata
import rocm.bindings.util
print(rocm.bindings.util.__version__)  # From _version.py

# Via module attributes (for packages that define them)
from rocm.bindings.hip import _version
print(_version.VERSION)
```

### From C/Cython
```cython
# _version.py is imported as a Python module
from . import _version
cdef str version = _version.VERSION
```

## Release Process

1. Update `VERSION.in` with new ROCm version/build number if needed
2. Set `HIP_PYTHON_VERSION_SHORT` or let it auto-generate from git
3. Build packages - all will use the same version
4. Tag release in git: `git tag v7.13.0.563`

## Notes

- VERSION file is **generated** and should NOT be committed to git
- VERSION.in is the **source of truth** and SHOULD be committed
- All packages automatically stay in sync
- For development builds, version includes git hash for traceability
- For release builds, set `HIP_PYTHON_VERSION_SHORT` to a clean version

## Future Enhancements

- [ ] Add standalone build support to each package (read VERSION file if parent doesn't provide)
- [ ] Add version validation to ensure compatibility
- [ ] Support pre-release versions (alpha, beta, rc)
- [ ] Add version bumping script
