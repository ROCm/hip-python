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

This repository provides low-level Python and Cython bindings for HIP.
Currently, only bindings for the AMD GPU backend of HIP are provided.

## Known Issues

### The `hiprand` Cython Module Causes Compiler Error

With all ROCm&trade; versions before and including version 5.6.0, compiling/using HIP Python's `hiprand` 
Cython module results in a compiler error.

The error is caused by the following statement in the C compilation
path of `<path_to_rocm>/hiprand/hiprand_hcc.h`, which is not legal in C
for aliasing a `struct` type:

```c
typedef rocrand_generator_base_type hiprandGenerator_st;
```

#### Workaround (Requires Access to Header File): Edit Header File

For this fix, you need write access to the ROCm&trade; header files.
Then, modify file `<path_to_rocm>/hiprand/hiprand_hcc.h` such that

```c
typedef rocrand_generator_base_type hiprandGenerator_st;
```

becomes 

```c
typedef struct rocrand_generator_base_type hiprandGenerator_st;
```

### ROCm&trade; 5.5.0 and ROCm&trade; 5.5.1

On systems with ROCm&trade; HIP SDK 5.5.0 or 5.5.1, the examples

* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_no_args.py

abort with errors.

An upgrade to version HIP SDK 5.6 or later (or a downgrade to version 5.4) is advised if 
the showcased functionality is needed.

### Unspecific

On certain Ubuntu 20 systems, we encountered issues when running the examples:

* hip-python/examples/0_Basic_Usage/hiprtc_launch_kernel_args.py
* hip-python/examples/0_Basic_Usage/rccl_comminitall_bcast.py

We could not identify the cause yet.

## Documentation

For examples, guides and API reference, please take a
look at the official HIP Python documentation pages:

https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html

## LICENSE

```
MIT License

Copyright (c) 2023 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
