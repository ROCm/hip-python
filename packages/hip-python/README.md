# hip-python

Backward-compatibility metapackage that exposes the `hip.*` namespace as an
alias of `rocm.bindings.*`, so existing `from hip import hip, hiprtc` code keeps
working. Optional extras pull in the rest of the bindings.

Part of [HIP Python](https://github.com/rocm/hip-python). Full docs:
<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

The HIP Python ecosystem consists of:

- `rocm-bindings-core` — shared types, loaders, ROCm path resolution
- `rocm-bindings-hip` — HIP and HIPRTC bindings
- `rocm-bindings-libraries` — math libraries (hipBLAS, hipSOLVER, hipFFT, ...)
- `rocm-bindings-systems` — system libraries (RCCL, ROCTX, amdsmi, HSA)
- `rocm-bindings-compiler` — LLVM-C, COMGR, Clang bindings
- `hip-python-interop` — CUDA interop layer (`cuda.bindings.*`)
- `hip-python` — this metapackage

## Install

```bash
pip install hip-python                  # core + HIP only
pip install hip-python[libraries]       # + math libraries
pip install hip-python[systems]         # + RCCL/ROCTX/amdsmi/HSA
pip install hip-python[compiler]        # + LLVM/COMGR/Clang
```

## Usage

```python
from hip import hip, hiprtc                 # legacy alias namespace
from rocm.bindings import hip, hiprtc       # recommended for new code
```

## Dependencies

- `rocm-bindings-core`
- `rocm-bindings-hip`
- Optional extras: `[libraries]`, `[systems]`, `[compiler]`
