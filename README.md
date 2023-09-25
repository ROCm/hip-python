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
# HIP Python Source Repository

This repository provides low-level Python and Cython Bindings 
for HIP and an interoperability layer for CUDA&reg; Python programs
(Python and Cython).

## Requirements

* Currently, only Linux is supported (prebuilt packages and code).
  * Prebuilt packages distributed via PyPI (or Test PyPI) are only provided for Linux systems that agree with the `manylinux_2_17_x86_64` tag.
* Requires that a compatible ROCm&trade; HIP SDK is installed on your system.
  * Source code is provided only for particular ROCm versions.
    * See the git branches tagged with `release/rocm-rel-X.Y[.Z]`
  * Prebuilt packages are built only for particular ROCm versions. 

> **NOTE**: You may find that packages for one ROCm&trade; release might be compatible with the ROCm&trade; HIP SDK of another release as the HIP Python functions load HIP C functions in a lazy manner.

### Build requirements

* All Python requirements are taking care of by installation scripts. 
If you decide not to use these scripts, take a look into the `requirements.txt` file 
in the top-level folder of the this repository and those 
in the repository's subfolders `hip-python` and `hip-python-as-cuda`.

## Install Prebuilt Package(s)

<!--
> **NOTE**: The prebuilt packages might not be available on PyPI directly after a ROCm release as this project is not an official part of the ROCm HIP SDK yet and thus is not fully integrated into the global ROCm HIP SDK build process. Check the `simple` lists to see if your operating system and Python version is supported: [hip-python](https://test.pypi.org/simple/hip-python/), [hip-python-as-cuda](https://test.pypi.org/simple/hip-python-as-cuda/).
-->

> **NOTE**: Prebuilt packages for some ROCm releases are published to Test PyPI first. Check the `simple` lists to see if your operating system and Python version is supported: [hip-python](https://test.pypi.org/simple/hip-python/), [hip-python-as-cuda](https://test.pypi.org/simple/hip-python-as-cuda/).

> **Warning**: Currently, we have not uploaded any HIP Python packages to PyPI yet. So far we have only uploaded packages to TestPyPI, mainly intended for internal testing purposes. If you find similar named packages on PyPI they may been provided by others, possibly with malicious intent.

### Via TestPyPI

First identify the first three digits of the version number of your ROCm&trade; installation.
Then install the HIP Python package(s) as follows:

```shell
python3 -m pip install -i https://test.pypi.org/simple hip-python>=$rocm_version
# if you want to install the CUDA Python interoperability package too, run:
python3 -m pip install -i https://test.pypi.org/simple hip-python-as-cuda>=$rocm_version
```

<!--
-- #### Via TestPyPI
-- 
-- Packages can be installed via the TestPyPI index by prefixing the
-- the PIP install commands as follows:
-- 
-- ```shell
-- python3 -m pip install -i https://test.pypi.org/simple ...
-- ```
-->

### Via Wheel in Local Filesystem

If you have HIP Python package wheels somewhere in your filesystem, you can also run:

```shell
python3 -m pip install <path/to/hip_python>.whl
# if you want to install the CUDA Python interoperability package too, run:
python3 -m pip install <path/to/hip_python_as_cuda>.whl
```

> **NOTE**: See the HIP Python user guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

## Build From Source

> **NOTE**: Main is typically based on the latest `release/rocm-rel-X.Y[.Z]` release branch. However, it is recommended to use the release branches directly when building from source.

1. Install ROCM
1. Install `pip`, virtual environment and development headers for Python 3:
   ```bash
   # Ubuntu:
   sudo apt install python3-pip python3-venv python3-dev
   ```
1. Check out the feature branch `release/rocm-rel-X.Y[.Z]` for your particular ROCm&trade; installation:
1. Then run:
   ```bash
   ./build_hip_python_pkgs.sh --post-clean
   ```

> **NOTE**: See the HIP Python developer guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

### Build Options

```
Usage: ./build_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path            Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs                 HIP Python libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                         Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --cuda-libs            HIP Python CUDA interop libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_CUDA_LIBS' if set or '*'.
                         Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --no-hip               Do not build package 'hip-python'.
  --no-cuda              Do not build package 'hip-python-as-cuda'.
  --no-docs              Do not build the docs of package 'hip-python'.
  --no-api-docs          Temporarily move the 'hip-python/docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs        Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  --docs-use-testpypi    Get the HIP Python packages for building the docs from Test PyPI.
  --docs-use-pypi        Get the HIP Python packages for building the docs from PyPI.
  --run-tests            Run the tests.
  -j,--num-jobs          Number of build jobs to use (currently only applied for building docs). Defaults to 1.
  --pre-clean            Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean           Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv          Do not create and use a virtual Python environment.
  -h, --help             Show this help message.
```

> **NOTE**: See the HIP Python developer guide for more details:
> https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

## Publish to PyPI

After building, the created Python wheels have a generic `linux_x86_64` tag that PyPI (and Test PyPI) will not accept. 
Hence, before publishing to PyPI one needs to first run `auditwheel` and check the 
suggested `manylinux_<GLIBC_VERSION_MAJOR>_<GLIBC_VERSION_MINOR>_x86_64` tag.

```bash
python3 -m auditwheel show hip-python/dist/<pkg>-linux_x86_64.whl
```

The `linux_x86_64` in the filename must then be replaced by 
the `manylinux_<GLIBC_VERSION_MAJOR>_<GLIBC_VERSION_MINOR>_x86_64`,
and the renamed packaged can be published to PyPI (and Test PyPI).

The `manylinux` should be at maximum `manylinux_2_17_x86_64`.

```{note}
The `GLIBC` should be as low as possible to support as many other Linux operating
systems as possible. Of course, it makes sense to focus on Linux operating
systems that are officially supported by ROCm.
```

```{note}
For more details on `auditwheel` and how to install it, see: https://pypi.org/project/auditwheel/
```

### Auditwheel: Example Output

On Ubuntu 20.04.6 LTS, we obtained for `hip-python` using Python 3.8:

```
$ python3 -m auditwheel show hip-python/dist/hip_python-5.6.31061.216-cp38-cp38-linux_x86_64.whl

hip_python-5.6.31061.216-cp38-cp38-linux_x86_64.whl is consistent with
the following platform tag: "manylinux_2_17_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libc.so.6 with versions
{'GLIBC_2.2.5', 'GLIBC_2.14', 'GLIBC_2.4'}

This constrains the platform tag to "manylinux_2_17_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
```

And we obtained for `hip-python-as-cuda`:

```
$ python3 -m auditwheel show hip-python-as-cuda/dist/hip_python_as_cuda-5.6.31061.216-cp38-cp38-linux_x86_64.whl

hip_python_as_cuda-5.6.31061.216-cp38-cp38-linux_x86_64.whl is
consistent with the following platform tag: "manylinux_2_17_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libc.so.6 with versions
{'GLIBC_2.14', 'GLIBC_2.2.5', 'GLIBC_2.4'}

This constrains the platform tag to "manylinux_2_17_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
```

Therefore, we can use the `manylinux_2_17_x86_64` tag for both packages.

## Known Compilation Issues

### The `hipsparse` Module won't Compile with Older GCC Release

With all ROCm&trade; versions before version 5.6.0 (exclusive) and older GCC versions, 
compiling HIP Python's `hipsparse` module results in a compiler error caused by lines such as:

```c
HIPSPARSE_ORDER_COLUMN [[deprecated("Please use HIPSPARSE_ORDER_COL instead")]] = 1,
```

#### Workaround 1: Disable Build of 'hipsparse' Module

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

### The `hiprand` module Won't Compile

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

## Other Known Issues

### ROCm&trade; 5.5.0 and ROCm&trade; 5.5.1

On systems with ROCm&trade; HIP SDK 5.5.0 or 5.5.1, the examples

* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_no_args.py

abort with errors.

An upgrade to version HIP SDK 5.6 or later (or a downgrade to version 5.4) is advised if 
the showcased functionality is needed.

### Unspecific

On certain Ubuntu 20 systems, we encountered issues when running the examples:

* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
* hip-python/examples/0_Basic_Usage/rccl_comminitall_bcast.py

We could not identify the cause yet.

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html
