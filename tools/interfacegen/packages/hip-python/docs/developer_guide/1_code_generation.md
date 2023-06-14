# Code Generation

This chapter shows how to use the code generation infrastructure
of the repository.

## Requirements

> **NOTE:** All Python requirements mentioned here are also baked into the `requirements.txt` file at the top level of the repository.

* Python >= 3.7
* The following ``pip`` package is required for running ``setup.py``:

    * ``libclang`` (>= 14.0.1, != 15.0.3, != 15.0.6, <= 15.0.6.1)

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


> **NOTE:** Requirement analysis details
> * Commit used for evaluation: 9a5505be427b95ead61a45c0f6e89c1ba2edc0ef
> * Date: 04/21/2023

<!---
Python >= 3.7 is required plus Python development files (e.g. via ``apt install python3-dev`` on Ubuntu).

To build ``pip`` packages (``.whl``) you need to install the ``pip`` package ``build``.
You further need to have `venv` installed (e.g. via ``apt install python3-venv`` on Ubuntu).
--->