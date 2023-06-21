# Code Generation

This chapter shows how to use the code generation infrastructure
of the repository.

## Requirements

:::{note}
All Python requirements mentioned here are also "baked into" the `requirements.txt` file at the top level of the repository.
:::

:::{admonition} Requirement analysis details

* Commit used for evaluation: 9a5505be427b95ead61a45c0f6e89c1ba2edc0ef
* Date: 04/21/2023
:::

* Python >= 3.7
* The following `pip` packages are required for running the code generation script
  * [`libclang`](https://pypi.org/project/libclang/) (>= 14.0.1, != 15.0.3, != 15.0.6, <= 15.0.6.1)
    
    Used for parsing the ROCm header files.
    
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

## Quick start

```bash
Usage: generate_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  -l, --libs        HIP Python modules to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
```