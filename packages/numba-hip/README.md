# Numba HIP

This repository provides a ROCm™ HIP backend for Numba.

> **For AMD GPUs on Linux**
>
> Numba HIP is for AMD Radeon™ and AMD Instinct™ accelerators.
> CUDA® devices are not supported by Numba HIP.
>
> As Numba HIP generates LLVM bitcode device libraries on-the-fly via the
> ROCm compiler and delegates runtime tasks to the HIP runtime via the HIP
> Python bindings, it does itself not pose any limitation on the
> supported AMD GPU devices.
>
> The ROCm on Radeon QA team has tested Numba HIP 0.1.4 on systems
> with AMD Radeon™ RDNA3, and AMD Radeon™ RDNA4 accelerators ROCm 7.1.0.
> The QA team's testing focused on the following AMD Radeon™ cards:
>
> - AMD Radeon™ RX 7900 XTX
> - AMD Radeon™ RX 7900 XT
> - AMD Radeon™ PRO W7900D
> - AMD Radeon™ PRO V710
> - AMD Radeon™ RX 9060 XT
> - AMD Radeon™ AI PRO R9700
>
> The authors have tested all versions of Numba HIP before version 0.1.4
> (so for ROCM versions before 7.1.1) predominantly on systems with
> AMD Instinct™ MI210X (gfx90a).
>
> The authors have tested Numba HIP 0.1.6 on ROCm 7.1.1 and ROCm 7.2.0
> systems with the following architectures:
>
> - gfx1030v (AMD Radeon™ PRO V620)
> - gfx1100p (AMD Radeon™ PRO W7800)
> - gfx1102 (AMD Radeon™ RX 7600 XT)
> - gfx1201 (AMD Radeon™ RX 9070 XT)
> - gfx90a (AMD Instinct™ MI210X/MI250)
> - gfx942 (AMD Instinct™ MI300A/MI300X/MI308X/MI325X)

> **Experimental project**
>
> This project primarily aims to support the AMD ROCm™ Data Science toolkit
> ([ROCm-DS](https://rocm.docs.amd.com/projects/rocm-ds/en/latest/index.html)).
> Most features that have been implemented were driven by ROCm-DS.
>
> However, we are also happy to get feedback from early adopters on their
> experience with the new Numba HIP backend. So if you give Numba HIP a try,
> let us know about your experience. We are looking forward to receiving
> your suggestions, issue reports, and pull requests.

## About Numba: A Just-In-Time Compiler for Numerical Functions in Python

Numba is an open source, NumPy-aware optimizing compiler for Python sponsored
by Anaconda, Inc. It uses the LLVM compiler project to generate machine code
from Python syntax.

Numba can compile a large subset of numerically-focused Python, including many
NumPy functions. Additionally, Numba has support for automatic
parallelization of loops, generation of GPU-accelerated code, and creation of
ufuncs and C callbacks.

For more information about Numba, see the Numba homepage:
<https://numba.pydata.org> and the online documentation:
<https://numba.readthedocs.io/en/stable/index.html>

## Numba HIP: Basic Usage

Numba HIP's programming interfaces follow Numba CUDA's design very closely.
Aside from the module name `hip`, there is often no difference between
Numba CUDA to Numba HIP code.

**Example 1 (Numba HIP):**

```python
from numba import hip

@hip.jit
def f(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = hip.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]
```

**Example 2 (Numba CUDA):**

```python
from numba import cuda

@cuda.jit
def f(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = cuda.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]
```

## Numba HIP: Posing As Numba CUDA

As Numba HIP allows to use syntax that is so similar to that of Numba CUDA and
there are already many projects that use Numba CUDA, we have introduced a
feature to the Numba HIP backend that allows it to pose as the Numba CUDA
backend to dependent applications. We demonstrate the usage of this feature in
the example below:

**Example 3 (Numba HIP posing as Numba CUDA):**

```python
from numba import hip
hip.pose_as_cuda() # now 'from numba import cuda'
                   # and `numba.cuda` delegate to Numba HIP.

# unchanged Numba CUDA snippet (Example 2)

from numba import cuda

@cuda.jit
def f(a, b, c):
   # like threadIdx.x + (blockIdx.x * blockDim.x)
   tid = cuda.grid(1)
   size = len(c)

   if tid < size:
       c[tid] = a[tid] + b[tid]
```

## Numba HIP: Limitations

Generally, we aim for feature parity with Numba CUDA.

The following Numba CUDA features are not available via Numba HIP:

- Cooperative groups support (ex: `cg.this_grid()`, `cg.this_grid().sync()`)
- Atomic operations for tuple and array types,
- Runtime kernel debugging functionality,
- Device code printf,
- HIP Simulator equivalent to CUDA Simulator (low priority, users can
  potentially reuse CUDA simulator),
- Half precision (fp16) operations.

Note further that so far only limited effort has been spent on optimizing the
performance of the just-in-time compilation infrastructure.

## Numba HIP: Design Differences vs. Numba CUDA

- While Numba CUDA utilizes the `nvvm` IR library, Numba HIP generates
  an architecture-specific LLVM bitcode library from a HIP C++ header file
  at startup of a Numba HIP program. However, a filesystem cache ensures that
  this needs to be done only once for a given session. The presence of such an
  additional caching mechanism must be considered when benchmarking.

- While Numba CUDA manually/semi-automatically creates basic device function
  signatures and the respective lowering procedures, Numba HIP does this
  fully-automatically from the aforementioned HIP C++ header file via the LLVM
  `clang` Python bindings.

- Furthermore, Numba HIP automatically links the HIP device library functions
  with the `math` module and uses a mechanism for recursive attribute
  resolution.

## Installation

> **Supported Numba versions**
>
> The Numba HIP backend has been tested with the following Numba versions:
>
> - 0.58.\*
> - 0.59.\*
> - 0.60.0
> - 0.61.2 (Numba HIP 0.1.5+)
>
> Other versions have not been tested; using the Numba HIP backend with these
> versions might work or not.

Numba HIP is part of the [HIP Python](https://github.com/rocm/hip-python)
monorepo. Its runtime dependencies (`rocm-bindings-hip`,
`rocm-bindings-compiler`, `hip-python-interop`) are published on PyPI for
every supported ROCm release, so for most users a plain `pip install` is all
that is needed. Make sure your `pip` is current first:

```bash
pip install --upgrade pip
```

### Install from PyPI

```bash
pip install numba-hip
```

This pulls in the matching `rocm-bindings-*` and `hip-python-interop`
wheels for the most recent supported ROCm release. ROCm itself must already be
installed on the system (see the HIP Python install guide for details).

### Build from the hip-python monorepo

To build Numba HIP together with the rest of HIP Python from source, use the
unified CMake build at the repository root:

```bash
git clone https://github.com/rocm/hip-python.git
cd hip-python/packages
cmake -B build
cmake --build build --target numba_hip_wheel   # just numba-hip
# or: cmake --build build --target all_wheels   # every package
```

The wheel lands in `packages/build/dist/`. See the top-level `README.md`
for the full build instructions and options.

### Install with test dependencies

The test extras (`pytest`, `cffi`) are exposed as a dependency group:

```bash
cd packages/numba-hip
pip install --group test
```

## Contact

Numba has a discourse forum for discussions:

- <https://numba.discourse.group>
