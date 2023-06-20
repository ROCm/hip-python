# Installation

## Supported Hardware

* See the ROCm  [Hardware_and_Software_Support](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) page for a list of supported AMD GPUs.

* Currently, NVIDIA GPUs are not supported.

## Supported Operationg Systems

Currently, only Linux is supported by the HIP Python interfaces's library loader.
The next section lists additional constraints due to the required ROCm installation.

## Software Requirements

To use the `HIP Python` interfaces, you must install a HIP Python version
that matches the `ROCm` installation, or vice versa.
See the [ROCm installation guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.3/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html)
for more details on how to install ROCm.

:::{important}
Identifying matching `ROCm` and `HIP Python` pairs must be done via
the HIP (runtime) version! On a system with installed ROCm, you can, e.g., run
`hipconfig` to read out the HIP version.
:::

:::{note}

You might "get away" with using incompatible HIP-HIP Python pairs if the
definitions of the types you are using have not changed between HIP releases 
and you are using a subset of functions that is present in the "incompatible" HIP Python package's
interfaces too.
:::

(subsec_hip_python_versioning)=
### HIP Python Versioning

While, the HIP runtime is versioned according to the below scheme

``HIP_VERSION_MAJOR.HIP_VERSION_MINOR.HIP_VERSION_PATCH[...]``

HIP Python packages are versioned as follows:

``HIP_VERSION_MAJOR.HIP_VERSION_MINOR.HIP_VERSION_PATCH.HIP_PYTHON_VERSION``

The HIP Python version ``HIP_PYTHON_VERSION`` consists of the revision count on
the ``main`` branch plus an optional ``dev<number>`` that indicates
the deviation from the ``main`` branch. Such a suffix is typically appended
if the HIP Python packages have been built from a code generator 
on a development branch.


:::{admonition} Example

``ROCm 5.4.3`` comes with HIP runtime version `5.4.22804-474e8620`, 
which can, e.g., be obtained via `hipconfig`. 
Hence, any HIP Python package `>= 5.4.22804.0` can be used.
:::

:::{note}

You might "get away" with using incompatible HIP-HIP Python pairs if the
definitions of the types you are using have not changed between HIP releases 
and you are using a subset of functions that is present in the "incompatible" HIP Python package's
interfaces too.
:::

### Installation

After having identified the correct package for your ROCm installation, type:

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