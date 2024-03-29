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
# Building and Packaging

This chapter describes the HIP Python packaging process.

## Building

### Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` files in
the subfolders `packages/hip-python` and `packages/hip-python-as-cuda`.
:::

:::{admonition} Requirement analysis details

* Commit used for evaluation: 9a5505be427b95ead61a45c0f6e89c1ba2edc0ef
* Date: 04/21/2023
:::

* Python >= 3.7
* The following `pip` packages are required for running `setup.py`:
    * setuptools>=42
    * cython
    * wheel
    * build

<!--
  -- Python >= 3.7 is required plus Python development files (e.g. via ``apt install python3-dev`` on Ubuntu).
  -- 
  -- To build ``pip`` packages (``.whl``) you need to install the ``pip`` package ``build``.
  -- You further need to have `venv` installed (e.g. via ``apt install python3-venv`` on Ubuntu).
  -->

### Known issues

* With older versions, fix C compiler error in hiprand_hcc.h
* With older versions when using older GCC compiler, fix C compiler error with `[[deprecated(...)]]`
  in hipsparse header.

## Quick Start

```bash
Usage: ./build_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path        Path to a ROCm&reg; installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --no-hip           Do not build package 'hip-python'.
  --no-cuda          Do not build package 'hip-python-as-cuda'.
  --no-docs          Do not build the docs of package 'hip-python'.
  --no-api-docs      Temporarily move the 'hip-python/docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs    Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  -j,--num-jobs      Number of build jobs to use (currently only applied for building docs). Defaults to 1.
  --pre-clean        Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv      Do not create and use a virtual Python environment.
  -h, --help         Show this help message.
```

## Packaging

### Requirements

* pypa build
