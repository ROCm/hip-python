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
# HIP Python

Low-level Python and Cython Bindings for HIP.

## Requirements

Requires that ROCm&reg; is installed on your system.

All Python requirements are taking care of by installation scripts. 
If you decide not to use these scripts, take a look into the `requirements.txt` file 
in the top-level folder of the this repository and those 
in the repository's subfolders `packages/hip-python` and `packages/hip-python-as-cuda`.

## Install Prebuilt Package(s)

### Via PyPI

First identify the first three digits of the version number of the HIP runtime that is part 
of your ROCm&reg; installation, e.g. via:

```shell
# extract it visually
hipconfig  | head -n1 # example output: 'HIP version  : 5.4.22804-474e8620'
# or programmatically
hip_version=$(hipconfig  | head -n1 | grep -o "\([0-9]\+\.\)\{2\}[0-9]\+" 
echo $hip_version # example output: '5.4.22804')
```

Then install the HIP Python package(s):

```shell
python3 -m pip install hip-python-$hip_version
# if you want to install the CUDA Python interoperability packages too, run:
python3 -m pip install hip-python-as-cuda-$hip_version
```

### Via Wheel in Local Filesystem

If you have HIP Python package wheels somewhere in your filesystem, you can also run:

```shell
python3 -m pip install <path/to/hip_python>.whl
# if you want to install the CUDA Python interoperability packages too, run:
python3 -m pip install <path/to/hip_python_as_cuda>.whl
```

> **NOTE**: See the HIP Python user guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

## Build From Source

1. Install ROCM

```bash
cd hip-python
# 2. Generate Cython code from ROCm header files
./generate_hip_python_pkgs.sh --post-clean
# 3. Build the packages
cd packages
./build_hip_python_pkgs.sh --post-clean
```

> **NOTE**: See the HIP Python devloper guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

### Options for the Build Scripts

* `generate_hip_python_pkgs.sh`:

  ```
  Usage: ./generate_hip_python_pkgs.sh [OPTIONS]

  Options:
    --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
    --libs        HIP Python libraries to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                      Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.
    --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
    --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
    -n, --no-venv     Do not create and use a virtual Python environment.
    -h, --help        Show this help message.
  ```

* `build_hip_python_pkgs.sh`:

  ```
  Usage: ./build_hip_python_pkgs.sh [OPTIONS]

  Options:
    --rocm-path        Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
    --libs             HIP Python libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                      Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
    --cuda-libs        HIP Python CUDA interop libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_CUDA_LIBS' if set or '*'.
                      Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
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

> **NOTE**: See the HIP Python devloper guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

## Known Issues

### The `hipsparse` module won't compile with older GCC version

With all ROCm&trade; versions before version 5.6.0 (exclusive) and older GCC versions, 
compiling HIP Python's `hipsparse` module results in a compiler error caused by lines such as:

```c
HIPSPARSE_ORDER_COLUMN [[deprecated("Please use HIPSPARSE_ORDER_COL instead")]] = 1,
```

#### Workaround 1: Disable Build of Hiprand Module

Disabling the build of the `hipsparse` HIP python module can, e.g., 
be achieved by supplying `--libs "^hipsparse"` to `build_hip_python_pkgs.sh`.

#### Workaround 2 (Requires Access to Header File): Edit Header File

For this fix, you need write access to the ROCm&trade; header files.
Then, e.g., modify file `<path_to_rocm>/hiprand/hiprand_hcc.h` such that:

```c
HIPSPARSE_ORDER_COLUMN [[deprecated("Please use HIPSPARSE_ORDER_COL instead")]] = 1,
```

becomes 

```c
HIPSPARSE_ORDER_COLUMN = 1, // [[deprecated("Please use HIPSPARSE_ORDER_COL instead")]] = 1,
```

### The `hiprand` module won't compile

With all ROCm&trade; versions before and including version 5.6.0, compiling HIP Python's `hiprand` 
module results in a compiler error.

The error is caused by the following line in the C compilation
path of `<path_to_rocm>/hiprand/hiprand_hcc.h`, which is not legal in C
for aliasing a `struct` type:

```c
typedef rocrand_generator_base_type hiprandGenerator_st;
```

#### Workaround 1: Disable Build of Hiprand Module

Disabling the build of the `hiprand` HIP python module can, e.g., 
be achieved by supplying `--libs "^hiprand"` to `build_hip_python_pkgs.sh`.

#### Workaround 2 (Requires Access to Header File): Edit Header File

For this fix, you need write access to the ROCm&trade; header files.
Then, modify file `<path_to_rocm>/hiprand/hiprand_hcc.h` such that

```c
typedef rocrand_generator_base_type hiprandGenerator_st;
```

becomes 

```c
typedef struct rocrand_generator_base_type hiprandGenerator_st;
```

Note that Cython users will experience the same issue if they use one
of the Cython modules in their code and use `c` as compilation language.

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html