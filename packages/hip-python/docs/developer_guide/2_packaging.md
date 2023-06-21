# Building and Packaging

This chapter describes the HIP Python packaging process.

## Building

### Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` files in
the top-level of the subfolders `packages/hip-python` and `packages/hip-python-as-cuda`.
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

<!---
Python >= 3.7 is required plus Python development files (e.g. via ``apt install python3-dev`` on Ubuntu).

To build ``pip`` packages (``.whl``) you need to install the ``pip`` package ``build``.
You further need to have `venv` installed (e.g. via ``apt install python3-venv`` on Ubuntu).
--->

### Known issues

* With older versions, fix C compiler error in hiprand_hcc.h
* With older versions when using older GCC compiler, fix C compiler error with `[[deprecated(...)]]`
  in hipsparse header.


## Quick Start

```bash
Usage: build_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --no-hip          Do not build package 'hip-python'.
  --no-cuda         Do not build package 'hip-python-as-cuda'.
  --no-docs         Do not build the docs of package 'hip-python'.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
```

## Packaging

### Requirements

* pypa build

