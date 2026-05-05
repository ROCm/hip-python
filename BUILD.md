# Building hip-python

There are two ways to build hip-python packages:

## Option 1: Build individual package (Development)

For local development, build individual packages directly:

```bash
cd python/rocm-bindings-core
python3 -m build --wheel --no-isolation

# Or with custom output directory
python3 -m build --wheel --no-isolation --outdir=/custom/path
```

This builds just that package with its C extensions. No auditwheel repair is applied.

## Option 2: Build all packages with CMake (Production/CI)

For production builds with manylinux wheel repair:

```bash
cd python

# Configure
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_PYTHON_BUILD_CORE=ON \
  -DHIP_PYTHON_BUILD_HIP=ON \
  -DHIP_PYTHON_BUILD_LIBRARIES=ON \
  -DHIP_PYTHON_BUILD_INTEROP=ON \
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=dist

# Build all wheels
cmake --build build --target all_wheels -j$(nproc)

# Or build specific package wheel
cmake --build build --target core_wheel
```

### CMake Options

**Package Selection:**
- `HIP_PYTHON_BUILD_CORE` - Build rocm-bindings-core (default: ON)
- `HIP_PYTHON_BUILD_HIP` - Build rocm-bindings-hip (default: ON)
- `HIP_PYTHON_BUILD_LIBRARIES` - Build rocm-bindings-libraries (default: ON)
- `HIP_PYTHON_BUILD_INTEROP` - Build hip-python-interop (default: ON)
- `HIP_PYTHON_BUILD_HIP_PYTHON` - Build hip-python metapackage (default: ON)

**Wheel Options:**
- `HIP_PYTHON_WHEEL_OUTPUT_DIR` - Output directory for wheels (default: `${CMAKE_BINARY_DIR}/dist`)
- `HIP_PYTHON_AUDITWHEEL_REPAIR` - Run auditwheel repair to create manylinux wheels (default: OFF)
  - **Note**: Requires `auditwheel` to be installed (`pip install auditwheel`)
  - Uses `--allow-pure-python-wheel` flag to handle pure Python packages
  - Applies `--exclude "*"` to prevent bundling ROCm libraries

**ROCm Options:**
- `ROCM_PATH` - Path to ROCm installation (default: from env or `/opt/rocm`)
- `HIP_PLATFORM` - HIP platform (default: `amd`)

**Build Options:**
- `CMAKE_BUILD_TYPE` - Build type: Release, Debug, RelWithDebInfo (default: Release)
- `CMAKE_C_COMPILER_LAUNCHER` - Compiler launcher like `sccache` or `ccache`
- `CMAKE_CXX_COMPILER_LAUNCHER` - Compiler launcher for C++

### CMake Targets

**Wheel targets:**
- `all_wheels` - Build all enabled package wheels
- `core_wheel` - Build rocm-bindings-core wheel
- `hip_wheel` - Build rocm-bindings-hip wheel
- `libraries_wheel` - Build rocm-bindings-libraries wheel
- `interop_wheel` - Build hip-python-interop wheel
- `hip_python_wheel` - Build hip-python metapackage wheel

**C extension targets:**
- `package_rocm_bindings_core` - Build util C extensions
- `package_rocm_bindings_hip` - Build hip C extensions
- `package_rocm_bindings_libraries` - Build libraries C extensions
- `package_hip_python_interop` - Build interop C extensions

## Environment Variables

- `DIST_DIR` - Sets `HIP_PYTHON_WHEEL_OUTPUT_DIR` if not explicitly configured
- `ROCM_PATH` - ROCm installation path (fallback if not set via CMake)

## Examples

### Basic build with all packages
```bash
cd python
cmake -B build
cmake --build build --target all_wheels
ls build/dist/*.whl
```

### Production build with manylinux wheels
```bash
cd python
cmake -B build \
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON \
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=/output
cmake --build build --target all_wheels -j16
```

### Build only util and hip packages
```bash
cd python
cmake -B build \
  -DHIP_PYTHON_BUILD_LIBRARIES=OFF \
  -DHIP_PYTHON_BUILD_INTEROP=OFF
cmake --build build --target all_wheels
```

### Use with sccache
```bash
cd python
cmake -B build \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake --build build --target all_wheels
```

## Differences Between Build Methods

| Feature | Individual `python3 -m build` | CMake `all_wheels` |
|---------|------------------------------|-------------------|
| Use case | Local development | Production/CI |
| Auditwheel | No | Optional (ON/OFF) |
| Wheel type | `linux_x86_64` | `manylinux_*` (if enabled) |
| Build all packages | No | Yes |
| Parallel builds | No | Yes (`-j` flag) |
| Integrated | No | Yes |

## Requirements

- Python 3.9+
- CMake 3.26+
- GCC or Clang
- ROCm 7.13+
- Python packages: `scikit-build-core>=0.11.2`, `cython>=3.0`, `build`
- Optional: `auditwheel`, `patchelf` (for manylinux wheels)
