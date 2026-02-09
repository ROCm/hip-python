<!-- MIT License
  --
  -- Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
  * Prebuilt packages distributed via PyPI (or Test PyPI) are only provided for
    Linux systems that agree with the `manylinux_2_17_x86_64` tag.
* Requires that a compatible ROCm&trade; HIP SDK is installed on your system.
  * Source code is provided only for particular ROCm versions.
    * See the `git` branches tagged with `release/rocm-rel-X.Y[.Z]`
  * Prebuilt packages are built only for particular ROCm versions.

> [!NOTE]
> You may find that packages for one ROCm&trade; release are compatible with
> the ROCm&trade; HIP SDK of another release as the HIP Python functions load
> HIP C functions in a lazy manner.

### Build requirements

* A Linux operating system
* A C compiler
* `bash`, `python3` + `venv`
* The ROCm&trade; HIP SDK
* Python 3.8+.

## Install Prebuilt Packages

> [!NOTE]
> Prebuilt packages for some ROCm releases are published to Test PyPI first.
> Check the `simple` lists to see if your operating system and Python version
> is supported: [hip-python](https://test.pypi.org/simple/hip-python/),
> [hip-python-as-cuda](https://test.pypi.org/simple/hip-python-as-cuda/).

***

> [!IMPORTANT]
> Ensure that `pip` has at least version `24.0`, please upgrade it otherwise.

***

> [!CAUTION]
> We have only uploaded HIP Python **dummy** packages to PyPI for security reasons.
> Note that they do not distribute any HIP Python functionality.
> Please use the ones from Test PyPI for now.

### Via TestPyPI

First identify the first three digits of the version number of your
ROCm&trade; installation. Then install the HIP Python package(s) as follows:

<!-- markdownlint-disable  MD013 -->

```shell
python3 -m pip install -i https://test.pypi.org/simple hip-python~=$rocm_version.0
# if you want to install the CUDA Python interoperability package too, run:
python3 -m pip install -i https://test.pypi.org/simple hip-python-as-cuda~=$rocm_version.0
```

<!-- markdownlint-enable  MD013 -->

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

If you have HIP Python package wheels on your filesystem, you can run:

```shell
python3 -m pip install $path_to_hip_python.whl
# if you want to install the CUDA Python interoperability package too, run:
python3 -m pip install $path_to_hip_python_as_cuda.whl
```

> [!NOTE]
> See the HIP Python user guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

## Build from Source

1. Install ROCM.
2. Install `pip`, virtual environment and development headers for Python 3:

   ```shell
   # Ubuntu:
   sudo apt install python3-pip python3-venv python3-dev
   ```

3. Check out the feature branch `release/rocm-rel-X.Y[.Z]` for your particular
   ROCm&trade; installation:
4. Finally run:

   ```shell
   ./build.sh --hip --cuda --post-clean
   ```

The build process will produce Python binary wheels in the subdirectories
`hip-python/dist/` and `hip-python-as-cuda/dist`, which can be installed
as discussed in the previous section.

> [!NOTE]
> See the HIP Python developer guide for more details:
> <https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>

### Build Options

<!-- markdownlint-disable  MD013 -->

```text
Usage: ./build.sh [OPTIONS]

Options:
  --rocm-path          Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs               HIP Python libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                       Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --cuda-libs          HIP Python CUDA interop libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_CUDA_LIBS' if set or '*'.
                       Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --hip                Build package 'hip-python'.
  --cuda               Build package 'hip-python-as-cuda'.
  --docs               Build the docs.
  --no-api-docs        Temporarily move the 'docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs      Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  --docs-use-testpypi  Get the HIP Python packages for building the docs from Test PyPI.
  --docs-use-pypi      Get the HIP Python packages for building the docs from PyPI.
  --no-archive         Do not put previously created packages into the archive folder.
  --run-tests          Run the tests.
  -j,--num-jobs        Number of build jobs to use. Defaults to 1.
  --pre-clean          Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean         Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv        Do not create and use a virtual Python environment.
  -h, --help           Show this help message.
```

<!-- markdownlint-enable  MD013 -->

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

<https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html>
