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
(ch_datatypes)=
# Python Argument and Return Types

This chapter highlights some of the datatypes that HIP Python's Python interfaces
use to convert various Python objects to the C datatypes expected by the routines of the HIP C API.

While generic types are defined in module {py:obj}`hip._util.types`,
specific helper types for {py:obj}`hip.hip` are defined in {py:obj}`hip._helper_types`.

## {py:obj}`~.Pointer`

Whenever a C function signature argument of pointer type is too generic --- e.g., if it is a C ``void *``
pointer --- HIP Python's code generator uses the {py:obj}`.types.Pointer` adapter to convert
a select list of Python objects to the pointer type expected by the C function.

This adapter type can be instantiated from the following Python objects:

* {py:obj}`None`:
    This will set the adapter's pointer attribute to ``NULL``.
* {py:obj}`~.Pointer`:
    The adapter copies the pointer attribute.
    Python buffer handle ownership, if the input object has acquired one, is not transferred.
* {py:obj}`int`:
    The adapter interprets the {py:obj}`int` value as address and copies to its pointer attribute.
* {py:obj}`ctypes.c_void_p`:
    Copies the ``value`` member.
* {py:obj}`object` that implements the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol.
    Retrieves the address from the ``"data"`` tuple of the ``__cuda_array_interface`` member of the object.
* {py:obj}`object` that implements the [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html).
    If the object represents a simple contiguous array, the adapter acquires the buffer handle and retrieves the buffer.
    Releases the handle at time of destruction.

Type checks are performed in the above order.

## Usage in HIP Python

The type is typically used as argument in the following scenarios:

* When modeling generic first-degree pointers of basic --- however, typically not for (`const`) `char *` --- or generic first-degree 
  pointers of type ``void`` type.
* When modeling in-out scalar arguments, where the user supplies `ctypes.addressof(myscalar)` as argument.
* As place-holder whenever a complicated type expression of pointer type has not been analyzed yet.

## {py:obj}`~.types.DeviceArray`

This is a datatype for handling device buffers returned by `~.hipMalloc` and related device memory allocation routines.
It implements the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol.

In case it appears as the type of a function argument, it can also be initialized from the following Python objects
that you can pass instead of it:

* {py:obj}`None`:
    This will set the type's pointer attribute to `{py:obj}`NULL`.
    No shape and type information is available in this case!
* {py:obj}`object` that is accepted as input by `~.Pointer.__init__`:
    In this case, init code from `~.Pointer` is used.
    `~.Py_buffer` object ownership is not transferred
    See `~.Pointer.__init__` for more information.
    No shape and type information is available in this case!
* {py:obj}`int`:
    The type interprets the {py:obj}`int` value as address and copies to its pointer attribute.
    No shape and type information is available in this case!
* {py:obj}``ctypes.c_void_p`:
    The type takes the pointer address from ``pyobj.value``.
    No shape and type information is available in this case!
* {py:obj}`object` with ``__cuda_array_interface__`` member:
    The type takes the integer-valued pointer address, i.e. the first entry of the {py:obj}`data` tuple 
    from {py:obj}`pyobj`'s member ``__cuda_array_interface__``.
    Copies shape and type information.

Type checks are performed in the above order.

:::{note}

Shape and type information and other metadata can be modified or overwritten after creation via the `~.DeviceArray.configure`
member function. be aware that you might need to pass the ``_force=True`` keyword argument --- 
in particular if your instance was created from a type that does not implement the 
`CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol.
:::

:::{admonition} See
{py:obj}`~.DeviceArray.configure`
:::

## Usage in HIP Python

The type is used as return value for:

* {py:obj}`~.hipMalloc`
* {py:obj}`~.hipExtMallocWithFlags`
* {py:obj}`~.hipMallocManaged`
* {py:obj}`~.hipMallocAsync`
* {py:obj}`~.hipMallocFromPoolAsync`

It can be passed directly to all functions that expect
a `~.Pointer` attribute.

:::{note}

The example <project:#sec_example_hip_python_device_arrays>
showcases how to use `~.DeviceArray.configure` to change the shape and type
of a `~.DeviceArray` instance and how use the `[]` operator
to retrieve a subarray.
:::

## Simple Lists

The types:

{py:obj}`~.types.ListOfBytes`,
{py:obj}`~.types.ListOfPointer`,
{py:obj}`~.types.ListOfInt`,

