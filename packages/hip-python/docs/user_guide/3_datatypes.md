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
# HIP Python's Adapter Types

This chapter highlights some of the datatypes that HIP Python's Python interfaces
use to convert various Python objects to the C datatypes expected by the routines of the HIP C API.

While generic types are defined in module {py:obj}`hip._util.types`,
specific helper types for {py:obj}`hip.hip` are defined in {py:obj}`hip._hip_helpers`.

(sec_pointer)=
## {py:obj}`~.Pointer`

Whenever a C function signature argument of pointer type is too generic --- e.g., already if it is a C ``void *``
pointer --- HIP Python's code generator uses the {py:obj}`.types.Pointer` adapter to convert
a select list of Python objects to the pointer type expected by the C function:

* {py:obj}`None`:

  This will set the adapter's pointer attribute to ``NULL``.

* {py:obj}`~.Pointer`:
  
  The adapter copies the pointer attribute.
  Python buffer handle ownership, if the input object has acquired one, is not transferred.

* {py:obj}`int`:
  
  The adapter interprets the {py:obj}`int` value as address and copies to its pointer attribute.

* {py:obj}`ctypes.c_void_p`:
  
  Copies the ``value`` member.

* {py:obj}`object` that implements the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol:
  
  Retrieves the address from the ``"data"`` tuple of the ``__cuda_array_interface`` member of the object.

* {py:obj}`object` that implements the [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html):
  
  If the object represents a simple contiguous array, the adapter acquires the buffer handle and retrieves the buffer.
  Releases the handle at time of destruction.

Type checks are performed in the above order.

## Usage in HIP Python

The type is typically used as argument in the following scenarios:

* When modeling generic first-degree pointers of basic --- however, typically not for (`const`) `char *` --- or generic first-degree 
  pointers of type ``void`` type.
* When modeling in-out scalar arguments, where the user supplies `ctypes.addressof(myscalar)` as argument.
* As place-holder whenever a complicated type expression of pointer type has not been analyzed yet.

(sec_device_array)=
## {py:obj}`~.types.DeviceArray`

This is a datatype for handling device buffers returned by `~.hipMalloc` and related device memory allocation routines.
It implements the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol.

In case it appears as the type of a function argument, it can also be initialized from the following Python objects
that you can pass instead of it:

* {py:obj}`None`:

  This will set the type's pointer attribute to `{py:obj}`NULL`.
  No shape and type information is available in this case!

* {py:obj}`object` that is accepted as input by the `__init__` routine of {py:obj}`~.Pointer`:

  In this case, init code from `~.Pointer` is used.
  `~.Py_buffer` object ownership is not transferred
  See `~.Pointer.__init__` for more information.
  No shape and type information is available in this case!

* {py:obj}`int`:

  The type interprets the {py:obj}`int` value as address and copies to its pointer attribute.
  No shape and type information is available in this case!

* {py:obj}`ctypes.c_void_p`:

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
[CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol
or via a HIP Python device memory allocation routine.
:::

:::{admonition} See
{py:obj}`~.DeviceArray.configure`
:::

## Usage in HIP Python

The type is used as return value of:

* {py:obj}`~.hipMalloc`
* {py:obj}`~.hipExtMallocWithFlags`
* {py:obj}`~.hipMallocManaged`
* {py:obj}`~.hipMallocAsync`
* {py:obj}`~.hipMallocFromPoolAsync`

It can be passed to functions that expect a {py:obj}`~.Pointer` argument and where passing an instance of this type instead makes sense, e.g.
you can pass it as copy destination or copy source in {py:obj}`~.hip.hipMemcpy`.

:::{note}

The example <project:#sec_example_hip_python_device_arrays>
showcases how to use `~.DeviceArray.configure` to change the shape and type
of a `~.DeviceArray` instance and how use the `[]` operator
to retrieve a subarray.
:::

(sec_list_types)=
## List Types

The adapter types

* {py:obj}`~.types.ListOfBytes`,
* {py:obj}`~.types.ListOfPointer`,
* {py:obj}`~.types.ListOfInt`, {py:obj}`~.types.ListOfUnsigned`, {py:obj}`~.types.ListOfUnsignedLong`,

are used for simple Python `list` or `tuple` objects whose elements are 

* {py:obj}`bytes`, 
* can be used to construct a {py:obj}`~.types.Pointer`,
* or can be converted to the C types `int`, `unsigned`, `unsigned long`, respectively.

The types can be initialized from the following Python objects:

* {py:obj}`list` / {py:obj}`tuple` of the respective element type:

  In this case, adapter type allocates an array of C values wherein it stores the addresses/values 
  obtained from the `list`/`tuple` entries.
  Furthermore, the instance's owner flag is set in this case.

* {py:obj}`object` that is accepted as input by the `__init__` routine of {py:obj}`~.Pointer`:

  In this case, init code from `~.Pointer` is used and the C owner flag remains unset.
  See `~.Pointer` for more information.

### Usage in HIP Python

HIP Python employs the {py:obj}`~.types.ListOfBytes` type in scenarios
were a list of C `const char *` is expected, 
and the {py:obj}`~.types.ListOfPointer` type where a list of C pointer types is expected.

The type {py:obj}`~.types.ListOfBytes` is for example used to convert input Python types in the following routines:

* {py:obj}`hiprtcCompileProgram` for argument `options`,
* {py:obj}`hiprtcCreateProgram`for argument `headers`, and
* {py:obj}`hiprtcCreateProgram` for argument `includeNames`.

HIP Python functions that expect a C array of C `int`, C `unsigned`, or C `unsigned long` element type (`int *`, ...)
use {py:obj}`~.types.ListOfInt`, {py:obj}`~.types.ListOfUnsigned`, or {py:obj}`~.types.ListOfUnsignedLong`,
respectively, to handle the conversion from appropriate Python input types.

:::{note}

In the examples <project:#sec_launching_kernels> and <project:#sec_launching_kernels_with_args>,
we can pass lists directly to {py:obj}`~.hiprtcCreateProgram` and {py:obj}`~.hiprtcCompileProgram`
thanks to the type conversions performed by {py:obj}`~.types.ListOfBytes`.
:::

## Helper type for {py:obj}`~.hipModuleLaunchKernel`

The type {py:obj}`~.HipModuleLaunchKernel_extra` is a datatype for handling Python list or tuple objects with entries that 
are either ctypes datatypes or that can be converted to type Pointer.
This adapter type is used for converting Python objects into a shape
that is expected by the `extra` argument of {py:obj}`~.HipModuleLaunchKernel`

The type {py:obj}`~.HipModuleLaunchKernel_extra` used as similar init procedure
as the types presented in <project:#sec_list_types>.

:::{note}

In the example <project:#sec_launching_kernels_with_args>,
we can pass lists of {py:obj}`ctypes` types and {py:obj}`~.types.DeviceArray` directly to {py:obj}`~.hipModuleLaunchKernel`
thanks to the type conversions performed by {py:obj}`~.HipModuleLaunchKernel_extra`.
:::

## Autogenerated Wrapper Types

HIP Python's code generator autogenerates Python wrapper types
for records (C `struct` and `union`) and C function pointers.

As the other types presented in this chapter, these types
can be initialized from different Python objects --- but only if and only if they serve
as input adapter for a HIP Python function. 

Their initialization from within Python code differs with respect to
the other discussed types. Instead of passing a Python object, you can pass initial values of their
properties --- either as positional arguments, keyword arguments, or a combination
of both --- to the constructor.

:::{admonition} Example (Initialization of `dim3` in Python code)

In the example <project:#sec_launching_kernels_with_args>, the autogenerated type {py:obj}`~.dim3` is initialized
using keyword arguments and positional arguments, respectively, as shown below:

```python
block = hip.dim3(x=32)
grid = hip.dim3(math.ceil(n/block.x))
```
:::

If autogenerated types are used as adapter for a function argument, the following input types are also accepted 
instead of passing an instance of the type:

* {py:obj}`None`:

  This will set the type's pointer attribute to ``NULL``.

* {py:obj}`int`:
  
  The type interprets the {py:obj}`int` value as address and copies to its pointer attribute.
  
* {py:obj}`ctypes.c_void_p`:
  
  Takes the pointer address ``pyobj.value`` and writes it to its pointer attribute.

* {py:obj}`object` that implements the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) protocol:
  
  Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple 
  from `pyobj`'s member ``__cuda_array_interface__``  and writes it to its pointer attribute.
  
  (Input type is not allowed for function pointer wrapper types.)

* {py:obj}`object` that implements the [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html):
  
  If the object represents a simple contiguous array, the adapter acquires the buffer handle, retrieves the buffer
  writes the buffer's data address to its pointer attribute. It releases the handle again at time of destruction.
  
  (Input type is not allowed for function pointer wrapper types.)
  
Type checks are performed in the above order.

## Properties of the Adapter Types

Finally, we want to highlight some important properties that are shared by all adapter types
presented in this chapter:

1. Aside from the autogenerated wrapper types, all datatypes presented in this chapter are subclasses of type
   {py:obj}`~.types.Pointer`. Therefore, they can all be passed to functions 
   that use {py:obj}`~.types.Pointer` for type conversions for the given argument;
   read the section <project:#sec_pointer> again to understand why.
1. Due to the way subclasses are initialized --- the init procedure of {py:obj}`~.types.Pointer`
   is invoked if the specialized conversions were not applied (see <project:#sec_device_array> and <project:#sec_list_types>) --- {py:obj}`~.types.Pointer` and any of its subclasses can be passed to any function that uses
   a subclass of {py:obj}`~.types.Pointer` as adapter.
1. You can instantiate all of the datatypes presented in this chapter in Python and Cython code.
1. You can subclass all of the datatypes presented in this chapter in Python and Cython code.
1. You can always use `int(<adapter_obj>)` to obtain the address of
   the underlying data as integer. 
   
   (Hint: Print the integer address via Python routine {py:obj}`hex` to obtain an address that you can compare with AMD log output. AMD log output can be enabled via environment variable `AMD_LOG_LEVEL`. Set it to a value of at least `3`.)

1. You can always pass an {py:obj}`int` to functions that
   use an adapter type for translating the given argument.
   It will be interpreted as integer representation of an address.

### Implications

Due to the first two properties, you can use one of the subclasses as adapter and then pass the 
object to a HIP Python function that uses {py:obj}`~.types.Pointer` or one of its subclasses as converter:

```{eval-rst}
.. code-block:: python
   :caption: HIP Python Datatype Used as Adapter in Python Code

   err = hip.hipMyFunc(...,pointer_is_used_to_convert_this_arg=hip._util.types.ListOfInt([1,2,3]),...)
```

Due to properties 3 and 4, you can define your own adapter types and pass them to HIP Python's
functions wherever those use {py:obj}`~.types.Pointer` or one of its subclasses as converter.
Obviously, your adapter must create data or link to data that makes sense for the C function
wrapped by HIP Python's Python interfaces.

Properties 5-6 allow, e.g., to pass autogenerated wrapper types to functions such as {py:obj}`~.hipMemcpy`
by converting them to integer first.