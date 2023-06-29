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

**Welcome to the documentation pages for HIP Python!**

HIP Python provides low-level Cython and Python&reg; bindings for the HIP runtime, HIPRTC,
multiple math libraries and the communication library RCCL,
and further a CUDA&reg; Python Interoperability layer that aims to simplify
the porting of CUDA Python Python and Cython programs.

:::{note}
This documentation has been generated based on HIP version 5.6.31061-8c743ae5d.
:::

## Spotlight

::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} How to Install HIP Python

Learn how to identify the correct HIP Python packages 
to install for your ROCm&trade;  installation, 
where to retrieve the packages and what options
you have to install them.

- {doc}`/user_guide/0_install`

```{eval-rst}
.. image:: _static/pip_install.PNG
   :width: 240
   :align: center
```

:::

:::{grid-item-card} How to Use HIP Python

Learn how to use HIP Python's interfaces in your Python or Cython program.
We present a large number of examples that cover 
HIP and HIPRTC as well as multiple math libraries (HIPBLAS, HIPRAND, HIPFFT) 
and the communication library RCCL.
Learn how to launch your own kernels and how the different
libraries interact.

- {doc}`/user_guide/1_usage`

```{eval-rst}
.. image:: _static/hip_usage.PNG
   :width: 240
   :align: center
```

:::

:::{grid-item-card} How to Port CUDA Python Applications

Learn how you can use HIP Python's CUDA Python interoperability layer
to port or even directly run CUDA Python applications
on AMD GPUs. The chapter covers Python and Cython programs.

- {doc}`/user_guide/2_cuda_python_interop`

```{eval-rst}
.. image:: _static/cuda_interop.PNG
   :width: 240
   :align: center
```

:::

:::{grid-item-card} HIP Python's Adapter Types

Learn about the datatypes that HIP Python uses to translate between C and Python 
and that are designed to ease interoperability with other
packages such as [NumPy](https://numpy.org) and [Numba](https://numba.pydata.org/).

- {doc}`/user_guide/3_datatypes`
:::

:::{grid-item-card} HIP Python's Python API

The full list of HIP Python Python variables, classes
and functions.

- {doc}`python_api/hip`
- {doc}`python_api/hiprtc`
- {doc}`python_api/hipblas`
- {doc}`python_api/rccl`
- {doc}`python_api/hiprand`
- {doc}`python_api/hipfft`
- {doc}`python_api/hipsparse`
- {doc}`python_api_manual/_hip_helpers`
- {doc}`python_api_manual/_util_types`
:::

:::{grid-item-card} The CUDA Python Interoperability Layer's Python API

The full list of the CUDA Python interoperability layer's Python variables, classes
and functions.

- {doc}`python_api/cuda`
- {doc}`python_api/cudart`
- {doc}`python_api/nvrtc`
:::

::::
