# Overview

Ths developer guide shows how to generate the HIP Python source code
and build HIP Python packages. 
While this guide does not target the typical HIP Python user, an understanding of
the HIP Python build and packaging process might still be of interest to advanced
users that want to tailor the code generator or the autogenerated source code
to their needs.

## Targeted audience

### Developers

As a developer, you will work on the highest level of the HIP Python repository.
You will likely tune the code generation inputs in ``setup_python.py``
or fix bugs/add features in folder `_codegen`.
For developers, all sections of this guide are of interest.

### Advanced Users

Most users will likely not want to get their hands on the code generation infrastructure.
Instead they will download and install prebuilt packages via `pip` or `conda`,
or they download the pregenerated package sources for a specific ROCm release from the
corresponding branch of the HIP Python git repository.
We refer users interested in the second approach to chapter [Building and Packaging](2_packaging.md) 
If you want to further tailor the code generation 
output according to your needs, then take a look into chapter [Code Generation](1_code_generation.md)
too. If you think your modifications should be made available to other users too and are about
to create a pull request, please first take a look at chapter [Commit Guidelines](4_commit_guide.md).