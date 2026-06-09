<!-- MIT License
  --
  -- Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
  --
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  --
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  --
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->
# InterfaceGen: Clang-based binding generator

This repository provides a Python package `interfacegen` that allows to generate language bindings from C header files to allow calling into the corresponding libraries from those other languages.
To this end, `interfacegen` utilizes `libclang`, i.e., the Python bindings to the LLVM `clang` runtime.

Among other recipes, this repository contains a recipe for generating the low-level Python and Cython bindings for HIP that are part of the HIP Python package [^hip-python].

## Install

```bash
pip install .
```

> **Cython 3.1.0 floor.** The codegen targets Cython and the
> generated bindings depend on a working `cdef T x = <T>expr`
> initializer for types that contain the inner `*const *` pattern
> (e.g. `const char *const *`). Cython 3.0.x **silently
> miscompiles** that statement — it parses the cdef but drops the
> initializer, leaving the local NULL at runtime. The codegen
> defends against this by emitting the prehoist as two separate
> statements (bare cdef + assignment), and we additionally pin
> `cython >= 3.1.0` so the underlying upstream bug isn't in the
> toolchain. The fix landed in Cython 3.1.0; 3.1.x and 3.2.x emit
> the assignment correctly. See
> `python/interfacegen/test/test_typed_helpers.py` for the
> regression tests that pin both the codegen split-form behaviour
> and the trailing-const handling.

## Develop in editable mode

Install in *editable* mode:

```bash
pip install -e .[dev]
```

## Recipes

Recipes for a couple of derived projects such as HIP Python can be found in subfolder `recipes`.

## Goals

General:

* Add support for different frameworks aside from HIP.
  * In particular, `HSA`, `OpenCL`, `OpenMP` and `ROCm LLVM` to broaden our support for Python developers and
    frameworks such as Numba.
  * Experimental recipes for ROCm LLVM and HSA have been created already but
    the code generation is incomplete / fails at a certain stage.
* Add support for other other languages aside from Python. In particular, we want to generate JAVA interfaces.
  We further might rewrite the HIPFORT code generator with this framework.
* Add support for different kinds of Python interfaces (Cython, CTypes, pybind11?)

HIP Python:

* Gradually add support more and more ROCm math libraries (hipsolver, roctx, rocblas, rocsparse, ...)

## TODOs

* [x] Add logging to all stages to more easily identify parsing and code generation errors.

Python / Cython:

* [x] Make runtime-linked library's path configurable via Python (and Cython)
* [ ] Adopt a CMake-based code generation process (cross platform, standardized).

## Discussions

### Namespaces

Namespaces should be sorted out before releasing this project to the public.
The following questions arised:

1. Move `hip` into `rocm.hip` or keep current structure mirrored from CUDA Python?
1. Provide `hsa` as  `rocm.hsa` (because of the AMD specific extensions)  or `hsa`?
1. Provide `opencl` as  `rocm.opencl` / `rocm.ocl` (because of the AMD specific extensions)  or `opencl` / omp?
1. Provide `openmp` as  `rocm.openmp` / `rocm.omp` (because of the AMD specific extensions)  or `openmp` / ocl?
1. Provide LLVM-C as  `rocm.llvmc` or `llvmc`?
   * Contribute interfaces back to LLVM project?

We currently lean towards using the prefix `rocm.` for all projects except `hip` as the latter should
be used similarly to the CUDA Python interfaces.

<!-- References -->

[^hip-python]: <https://github.com/rocm/hip-python>
