<!-- MIT License
  -- 
  -- Copyright (c) 2023 Advanced Micro Devices, Inc.
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
# InterfaceGen: Codegenerator for HIP Python and other projects

This repository provides a Python package `interfacegen` that
allows to generate interfaces for other languages from C APIs.
It is based on the Python interfaces for the LLVM `clang` runtime.

Contains recipe for generating low-level Python and Cython Bindings for HIP.

## Goals

General:

* Add support for different frameworks aside from HIP. 
  * In particular, `HSA` and `ROCm LLVM` to broaden our support for Python developers and 
    frameworks such as Numba.
* Add support for other other languages aside from Python. In particular, we want to generate JAVA interfaces.
  We further might rewrite the HIPFORT code generator with this framework.
* Add support for different kinds of Python interfaces (Cython, CTypes, pybind11?)

HIP Python:

* Gradually add support more and more ROCm math libraries (hipsolver, roctx, ROCm ...)

## Discussions

### Namespaces

* Move `hip` into `rocm.hip` or keep current structure mirrored from CUDA Python?
* 

#### Requirements

Requires that ROCm&trade; is installed on your system.

All Python requirements are taking care of by the `generate_hip_python_pkgs.sh` script. 
If you decide not to use it, take a look into the `requirements.txt` file 
in the top-level folder of the this repository.

## Recipes

### HIP Python

#### Run Codegenerator

1. Install ROCm&trade;.
2. Clone a branch of `https://github.com/ROCmSoftwarePlatform/hip-python` into `<path/to/rocm_software_platform_hip_python>`
   that you want to base your work on.
3. Generate Cython code from ROCm&trade; header files, specify `<path/to/rocm_software_platform_hip_python>` as output directory:
   ```bash
   ./generate_hip_python_pkgs.sh <path/to/rocm_software_platform_hip_python> --post-clean
   ```

### More Options

```
Usage: ./generate_hip_python_pkgs.sh output_dir [OPTIONS]

Required:
   output_dir       The output directory to which the files should be written to. Must contain 'hip-python' and 'hip-python-as-cuda' subfolders.

Options:
  --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs            HIP Python libraries to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                    Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
```
