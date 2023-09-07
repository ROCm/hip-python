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
# Installation

## Supported Hardware

Currently, only AMD GPUs are supported.

* See the ROCm&trade; [Hardware_and_Software_Support](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) page for a list of supported AMD GPUs.

## Supported Operation Systems

Currently, only Linux is supported by the HIP Python interfaces's library loader.
The next section lists additional constraints with respect to the required ROCm&trade; installation.

## Software Requirements

You must install a HIP Python version that is compatible with your  ROCm&trade; HIP SDK installation, or vice versa -- in particular, if you want to use the Cython interfaces. See the [ROCm&trade; documentation](https://rocm.docs.amd.com/en/latest/index.html) for more details on how to install the ROCm&trade; HIP SDK.

(subsec_hip_python_versioning)=
### HIP Python Versioning

The ROCm&trade; HIP SDK is versioned according to the below scheme:

``ROCM_VERSION_MAJOR.ROCM_VERSION_MINOR.ROCM_VERSION_PATCH[...]``:

While HIP Python packages are versioned according to:

``ROCM_VERSION_MAJOR.ROCM_VERSION_MINOR.ROCM_VERSION_PATCH.HIP_PYTHON_CODEGEN_VERSION.HIP_PYTHON_RELEASE_VERSION``

Any version of HIP Python that matches the first three numbers is suitable for your ROCm&trade; HIP SDK installation.

:::{admonition} Example

If you have the ROCm&trade; HIP SDK 5.6.0 installed, any
HIP Python package with version `5.6.0.X.Y` can be used.
:::

:::{note}

The HIP Python Python packages load HIP SDK functions in a lazy manner.
Therefore, you will likely "get away" with using "incompatible" ROCm&trade; and HIP Python pairs if the
following assumptions apply: 

* You are only using Python code, 
* the definitions of the types that you use have not changed between the respective ROCm&trade; releases, and 
* you are using a subset of functions that is present in both ROCm&trade; releases. 

Both assumptions often apply.
:::

### Installation Commands

:::{important}

Especially on older operating systems, ensure that your `pip` is upgraded to
the latest version. You can upgrade it, e.g., as follows:

```shell
python3 -m pip install --upgrade pip
```
:::

After having identified the correct package for your ROCm&trade; installation, type:

```shell
python3 -m pip install hip-python-<hip_version>.<hip_python_version>
```

or if you have a HIP Python wheel somewhere in your filesystem:

```shell
python3 -m pip install <path/to/hip_python>.whl
```

:::{note}

The first option will only be available after the public release on PyPI.
:::