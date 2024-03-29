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
# Overview and High-Level Build Interfaces

This developer guide chapter shows you how to use the repository's 
high-level build interfaces to generate HIP Python's Cython sources, the final packages,
and the documentation. While this guide does not target the typical HIP Python user, an understanding of
the HIP Python build and packaging process might still be of interest to advanced
users that want to tailor the code generator or the autogenerated source code
to their needs.

:::{admonition} Requirement analysis details

* Commit used for evaluation: 1db035
* Date: 06/26/2023
:::

## HIP Python Repository Structure

```{eval-rst}
.. code-block::
   :caption: HIP Python Repository Structure. Main entry points and package folders are highlighted.
   :name: repository_structure
   :linenos:
   :emphasize-lines: 15, 18, 19, 28

    hip-python
    ├── _codegen
    │   ├── __init__.py
    │   ├── control.py
    │   ├── cparser.py
    │   ├── cython.py
    │   ├── doxyparser.py
    │   ├── test
    │   ├── tree.py
    ├── _controls.py
    ├── _cuda_interop_layer_gen.py
    ├── _gitversion.py
    ├── _parse_hipify_perl.py
    ├── codegen_hip_python.py
    ├── generate_hip_python_pkgs.sh
    ├── requirements.txt
    └───packages
      ├── build_hip_python_pkgs.sh
      ├── hip-python
      │   ├── dist
      │   ├── docs
      │   ├── examples
      │   ├── hip
      │   ├── pyproject.toml
      │   ├── requirements.txt
      │   ├── setup.cfg
      │   └── setup.py
      └───hip-python-as-cuda
            ├── cuda
            ├── dist
            ├── pyproject.toml
            ├── requirements.txt
            ├── setup.cfg
            └── setup.py
```

<!--
  --## Targeted Audience
  --
  --### DevOps Engineers
  --
  --As a DevOps engineer, you will work on the highest level of the HIP Python repository
  --and in the ``packages`` subfolder. You will mainly be  concerned with the scripts 
  --``generate_hip_python_pkgs.sh`` for code generation and ``packages/build_hip_python_pkgs.sh``
  --for package and docs building.
  --
  --### Developers
  --
  --As a developer, you will work on the highest level of the HIP Python repository.
  --You will likely tune the code generation inputs in ``codegen_hip_python.py`` and ``_controls.py``, 
  --or fix bugs/add features in folder `_codegen`.
  --For developers, all sections of this guide are likely of interest.
  --
  --### Advanced Users
  --
  --Most users will likely not want to get their hands on the code generation infrastructure.
  --Instead they will download and install prebuilt packages via `pip` or `conda`,
  --or they download the pregenerated package sources for a specific ROCm&trade; release from the
  --corresponding branch of the HIP Python git repository.
  --We refer users interested in the second approach to chapter [Building and Packaging](2_packaging.md) 
  --If you want to further tailor the code generation 
  --output according to your needs, then take a look into chapter [Code Generation](1_code_generation.md)
  --too. If you think your modifications should be made available to other users too and are about
  --to create a pull request, please first take a look at chapter [Commit Guidelines](4_commit_guide.md).
  -->

## Building from Source

Building the HIP Python packages from the HIP C/C++ sources is achieved in two steps:

1. Code generation via `generate_hip_python_pkgs.sh`.
2. Package and docs building via `packages/build_hip_python_pkgs.sh`.

The former script can be used as follows:

```
Usage: ./generate_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path       Path to a ROCm&trade; installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  -l, --libs        HIP Python modules to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
```

The latter can be used as shown below:

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
  --run-tests        Run the tests.
  -j,--num-jobs      Number of build jobs to use (currently only applied for building docs). Defaults to 1.
  --pre-clean        Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv      Do not create and use a virtual Python environment.
  -h, --help         Show this help message.
```

By default, both scripts will create a virtual Python environment named `_venv` in their
respective parent folder when run. All Python dependencies are automatically installed 
into these environments.

On systems with an ROCm&trade; installation installed to the standard location, it suffices to run:

```{eval-rst}
.. code-block:: bash
   :caption: Code generation and building (no removal of virtual Python environments).
   
   cd hip-python
   ./generate_hip_python_pkgs.sh
   cd packages
   ./build_hip_python_pkgs.sh
```

To remove the virtual environments after running the script, additionally add
the `--post-clean` flag as shown below:

```{eval-rst}
.. code-block:: bash
   :caption: Code generation and building plus removal of virtual Python environments.
   
   cd hip-python
   ./generate_hip_python_pkgs.sh --post-clean
   cd packages
   ./build_hip_python_pkgs.sh --post-clean
```

### Code Generation Step

In the code generation step, the header files in a ROCm&trade; installation are parsed
and the HIP Python's Cython modules are generated. Moreover, content of
the `hipify-perl` script is combined with the HIP header file parse trees in order to generate
the CUDA&reg; interoperability layer. Finally, package metadata and documentation files
per HIP and CUDA interoperability layer module are generated.

The script `./generate_hip_python_pkgs.sh` abstracts all these tasks plus the dependency management.
The full script is [printed below](generate_hip_python_pkgs_sh) for a deeper inspection.
Most options do not concern DevOps engineers, they are mainly intended for easing
development work.

```{eval-rst}
.. literalinclude:: ../../../../generate_hip_python_pkgs.sh
   :linenos:
   :language: bash
   :name: generate_hip_python_pkgs_sh
   :caption: Code generation script (``generate_hip_python_pkgs.sh``).
```

#### Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` file at the top level of the repository.
:::

* Python >= 3.7
* The following `pip` packages are required for running the code generation script
  * [`libclang`](https://pypi.org/project/libclang/) (>= 14.0.1, != 15.0.3, != 15.0.6, <= 15.0.6.1)
    
    Used for parsing the ROCm&trade; header files.
    
    We tested the following versions:
     * 9.0.1	-- fail
      * 10.0.1.0 -- fail
      * 11.0.0 -- fail
      * 11.0.1 -- fail
      * 11.1.0 -- fail
      * 12.0.0 -- fail
      * 13.0.0 -- fail
      * 14.0.1 -- **success**
      * 14.0.6 -- **success**
      * 15.0.3 -- fail
      * 15.0.6 -- fail
      * 15.0.6.1 -- **success**
      * 16.0.0 -- fail
  * [`Cython`](https://cython.org/)
    
    We use Cython's Tempita template engine for code generation.
  * [`pyparsing`](https://pyparsing-docs.readthedocs.io/en/latest/index.html)
    
    Used for constructing the [`doxygen`](https://www.doxygen.nl/) parser.

The following `pip` packages are optional:

* [`levenshtein`](https://pypi.org/project/Levenshtein/)

  For giving suggestions if a HIP name could not be mapped to a CUDA name
  when constructing the CUDA interoperability modules.

### Packaging and Docs Building Step

In the second step, the generated Cython modules are compiled to Python modules before
both are packaged via PyPA `~.build` into a `~.pip` wheel that can be 
installed directly or uploaded to a package repository such as PyPI.
Finally, the docs are built in this step which require the previously build packages as dependencies.

The script `./packages/build_hip_python_pkgs.sh` abstracts all these tasks 
plus the dependency management --- aside from uploading the package.
The full script is [printed below](build_hip_python_pkgs_sh) for a deeper inspection.
Most options do not concern DevOps engineers, they are mainly intended for easing
development work.

```{eval-rst}
.. literalinclude:: ../../../../packages/build_hip_python_pkgs.sh
   :linenos:
   :language: bash
   :name: build_hip_python_pkgs_sh
   :caption: Build script (``packages/build_hip_python_pkgs.sh``).
```
#### Packaging Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` files in
the subfolders `packages/hip-python` and `packages/hip-python-as-cuda`.
:::

* Python >= 3.7
* The following `pip` packages are required for running `setup.py`:
    * setuptools>=42
    * cython
    * wheel
    * build

#### Docs Building Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` file in
the subfolders `packages/docs/hip-python`.
:::

* Python >= 3.7
* The following `pip` packages are required for running building the documentation:
  * rocm-docs-core
  * myst-parser[linkify]