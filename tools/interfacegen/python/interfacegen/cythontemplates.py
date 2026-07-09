# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Note: wrapper_class_decl_template must declare all ``@staticmethod`` ``cdef``
#       functions
# Note: Syntax ``bint owner=*`` is necessary to specify default value in
#       implementation part

wrapper_class_decl_template = """
{{default cptr_type = cname + "*"}}
{{default is_complete_type = True}}
{{default is_array = False}}
cdef class {{name}}({{util_types_prefix}}Pointer):
    cdef bint _is_ptr_owner

    cdef {{cptr_type}} getElementPtr(self)

    @staticmethod
    cdef {{name}} fromPtr(void* ptr, bint owner=*)
    @staticmethod
    cdef {{name}} fromPyobj(object pyobj)
{{if is_complete_type}}
    @staticmethod
    cdef __allocate(void* ptr)
    @staticmethod
    cdef {{name}} new()
{{if not is_array}}
    @staticmethod
    cdef {{name}} fromValue({{cname}} other)
{{endif}}
{{endif}}
"""

wrapper_class_impl_base_template = """
{{default cptr_type = cname + "*"}}
{{default is_funptr = False}}
{{default is_complete_type = True}}
{{default properties_name = None}}
{{default is_array = False}}
cdef class {{name}}({{util_types_prefix}}Pointer):
    \"""Python wrapper for cdef class {{cname}}.

    Python wrapper for cdef class {{cname}}.

    If this type is initialized via its `__init__` method, it allocates a
    member of the underlying C type and destroys it again if the wrapper
    type is deallocated.

    This type also serves as adapter when appearing as argument type in a
    function signature. In this case, the type can further be initialized
    from a number of Python objects:

    * `None`:

      This will set the ``self._ptr`` attribute to ``NULL``.

    * `int`:

      Interprets the integer value as pointer address and writes it to ``self._ptr``.
      No ownership is transferred.

    * `ctypes.c_void_p`:

      Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
      No ownership is transferred.

    * `object` that implements the `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_ protocol:

      Takes the integer-valued pointer address, i.e. the first entry of the `data` tuple
      from `pyobj`'s member ``__cuda_array_interface__``  and writes it to ``self._ptr``.

    * `object` that implements the Python buffer protocol:

      If the object represents a simple contiguous array,
      writes the `Py_buffer` associated with ``pyobj`` to `self._py_buffer`,
      sets the `self._py_buffer_acquired` flag to `True`, and
      writes `self._py_buffer.buf` to the data pointer `self._ptr`.

    * `{{util_types_prefix}}Pointer`:

      Takes the pointer address ``pyobj._ptr`` and writes it to ``self._ptr``.
      No ownership is transferred.

    Type checks are performed in the above order.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _is_ptr_owner (C type ``bint``, protected):
            If this wrapper is the owner of the underlying data.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    \"""
    # C members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self._is_ptr_owner = False
        self._py_buffer_acquired = False

    cdef {{cptr_type}} getElementPtr(self):
        return <{{cptr_type}}>self._ptr

    @staticmethod
    cdef {{name}} fromPtr(void* ptr, bint owner=False):
        \"""Factory function to create ``{{name}}`` objects from
        given ``{{cname}}`` pointer.
        {{if is_complete_type}}

        Setting ``owner`` flag to ``True`` causes
        the extension type to free the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        {{endif}}
        \"""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        wrapper._ptr = ptr
        wrapper._is_ptr_owner = owner
        return wrapper

    @staticmethod
    def fromObj(pyobj):
        \"""Creates a {{name}} from a Python object.

        Derives a {{name}} from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``{{name}}`` reference, this method
        returns it directly. No new ``{{name}}`` is created in this case.
        \"""
        return {{name}}.fromPyobj(pyobj)

    @staticmethod
    cdef {{name}} fromPyobj(object pyobj):
        \"""Creates a {{name}} from a Python object.

        Derives a {{name}} from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``{{name}}`` reference, this method
        returns it directly. No new ``{{name}}`` is created in this case.

        Args:
            pyobj (object): Must be either `None`; a `{{util_types_prefix}}Pointer`; a simple, contiguous buffer according to the buffer protocol;
                            or of type `{{name}}`; `int`; or `ctypes.c_void_p`.

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of {{name}}!
        \"""
        cdef {{name}} wrapper

        if isinstance(pyobj,{{name}}):
            return pyobj
        else:
            wrapper = {{name}}.__new__({{name}})
            wrapper.init_from_pyobj(pyobj)
            return wrapper

    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        {{if is_complete_type}}
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self._is_ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
        {{endif}}
    {{if is_complete_type}}

    @staticmethod
    cdef __allocate(void** ptr):
        ptr[0] = stdlib.malloc(sizeof({{cname}}))
        string.memset(<void*>ptr[0], 0, sizeof({{cname}}))

        if ptr[0] is NULL:
            raise MemoryError

    @staticmethod
    cdef {{name}} new():
        \"""Factory function to create {{name}} objects with
        newly allocated {{cname}}\"""
        cdef void* ptr
        {{name}}.__allocate(&ptr)
        return {{name}}.fromPtr(ptr, owner=True)

    @staticmethod
    def allocate(Py_ssize_t count=1):
        \"""Allocate an owned, zero-initialized array of ``count`` ``{{cname}}`` elements.

        ``count`` defaults to 1 (a single element). The returned ``{{name}}``
        owns the buffer and frees it when the wrapper is deallocated
        (``self._is_ptr_owner`` is set). Element ``i`` is read/written via the
        generated ``get_*(i)``/``set_*(i)`` accessors; the caller tracks
        ``count``.
        \"""
        if count < 1:
            raise ValueError("'count' must be positive")
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        wrapper._ptr = stdlib.malloc(count*sizeof({{cname}}))
        if wrapper._ptr is NULL:
            raise MemoryError()
        string.memset(wrapper._ptr, 0, count*sizeof({{cname}}))
        wrapper._is_ptr_owner = True
        return wrapper

{{if not is_array}}
    @staticmethod
    cdef {{name}} fromValue({{cname}} other):
        \"""Allocate new C type and copy from ``other``.
        \"""
        wrapper = {{name}}.new()
        string.memcpy(wrapper._ptr, &other, sizeof({{cname}}))
        return wrapper
{{endif}}

    def c_sizeof(self):
        \"""Returns the size of the underlying C type in bytes.
        Note:
            Implemented as function to not collide with
            autogenerated property names.
        \"""
        return sizeof({{cname}})
    {{endif}}

    def __int__(self):
        \"""Returns the data's address as long integer.
        \"""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)

    def __repr__(self):
        return f"<{{name}} object, ptr: {int(self)}>"

    def as_c_void_p(self):
        \"""Returns the data's address as `ctypes.c_void_p`
        Note:
            Implemented as function to not collide with
            autogenerated property names.
        \"""
        return ctypes.c_void_p(int(self))
"""

wrapper_class_record_init_template = """\
    {{if is_complete_type}}
    {{default defaults = dict()}}
    {{default all_properties_rendered = False}}
    {{default is_union = False}}
    {{py: all_properties_and_is_no_union = all_properties_rendered and not is_union}}
    {{if all_properties_and_is_no_union}}
    def __init__(self,*args,**kwargs):
    {{else}}
    def __init__(self,**kwargs):
    {{endif}}
        \"""Constructor type {{name}}.

        Constructor for type {{name}}.

        Args:
            {{if all_properties_and_is_no_union}}
            *args:
                Positional arguments. Initialize all or a subset of the member variables
                according to their order of declaration.
            {{endif}}
            **kwargs:
                Can be used to initialize member variables at construction,
                Just pass an argument expression of the form <member>=<value>
                per member that you want to initialize.
        \"""
        {{name}}.__allocate(&self._ptr)
        self._is_ptr_owner = True
        {{for k,v in defaults.items()}}
        self.{{k}} = {{v}}
        {{endfor}}
        attribs = self.{{properties_name}}()
        used_attribs = set()
        {{if all_properties_and_is_no_union}}
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        {{endif}}
        {{if is_union}}
        if len(kwargs) > 1:
            raise ValueError("Not more than one attribute might specified for Python types derived from C union types.")
        {{endif}}
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    {{endif}}
"""

# note: must be dedented
wrapper_class_record_property_template = """\
{{py: cptr_type = record_cname + "*"}}
{{py: element_ptr = "(<"+cptr_type+">self._ptr)"}}
{{if is_basic_type}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{element_ptr}}[i].{{attr}}
def set_{{attr}}(self, i, {{typename}} value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    {{element_ptr}}[i].{{attr}} = value
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, {{typename}} value):
    self.set_{{attr}}(0,value)
{{elif is_pointer_to_basic_type_or_void}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{handler}}.fromPtr({{element_ptr}}[i].{{attr}})
def set_{{attr}}(self, i, object value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.

    Note:
        This can be dangerous if the pointer is from a python object
        that is later on garbage collected.
    \"""
    {{element_ptr}}[i].{{attr}} = <{{typename}}>cpython.long.PyLong_AsVoidPtr(int({{handler}}.fromPyobj(value)))
@property
def {{attr}}(self):
    \"""{{brief_comment}}
    Note:
        Setting this {{attr}} can be dangerous if the underlying pointer is from a python object that
        is later on garbage collected.
    \"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, object value):
    self.set_{{attr}}(0,value)
{{elif is_basic_type_constantarray}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{element_ptr}}[i].{{attr}}
# TODO add setters
#def set_{{attr}}(self, i, {{typename}} value):
#    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.
#    \"""
#    {{element_ptr}}[i].{{attr}} = value
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
# TODO add setters
#@{{attr}}.setter
#def {{attr}}(self, {{typename}} value):
#    self.set_{{attr}}(0,value)
{{elif is_enum}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{typename}}({{element_ptr}}[i].{{attr}})
def set_{{attr}}(self, i, value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    if not isinstance(value, {{typename}}):
        raise TypeError("'value' must be of type '{{typename}}'")
    {{element_ptr}}[i].{{attr}} = value.value
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, value):
    self.set_{{attr}}(0,value)
{{elif is_record}}
def get_{{attr}}(self, i):
    \"""Get value of ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{typename}}.fromPtr(&{{element_ptr}}[i].{{attr}})
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
{{elif is_pointer_to_record}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{record_wrapper}}.fromPtr(<void*>{{element_ptr}}[i].{{attr}})
def set_{{attr}}(self, i, object value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.

    Note:
        This can be dangerous if the pointer is from a python object
        that is later on garbage collected.
    \"""
    {{element_ptr}}[i].{{attr}} = <{{pointer_ctype_no_const}}>cpython.long.PyLong_AsVoidPtr(int({{record_wrapper}}.fromPyobj(value)))
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, object value):
    self.set_{{attr}}(0,value)
{{elif is_pointer}}
def get_{{attr}}(self, i):
    \"""Get value ``{{attr}}`` of ``{{element_ptr}}[i]``.
    \"""
    return {{util_types_prefix}}Pointer.fromPtr(<void*>{{element_ptr}}[i].{{attr}})
def set_{{attr}}(self, i, object value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.

    Note:
        This can be dangerous if the pointer is from a python object
        that is later on garbage collected.
    \"""
    {{element_ptr}}[i].{{attr}} = <{{pointer_ctype_no_const}}>cpython.long.PyLong_AsVoidPtr(int({{util_types_prefix}}Pointer.fromPyobj(value)))
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
@{{attr}}.setter
def {{attr}}(self, object value):
    self.set_{{attr}}(0,value)
{{endif}}
"""

wrapper_class_constantarray_get_element_template = """\
    {{# var cname}}
    {{default is_basic_type = True}}
    {{default dim = 1}}
    {{default shape = (1,)}}
    {{py: cptr_type = cname + "*"}}
    {{py: is_dim_1 = (int(dim) == 1)}}
    {{py: cptr_type = cname + "*"}}
    {{py: element_ptr = "(<"+cptr_type+">self._ptr)"}}
    {{if is_basic_type}}
    def __getitem__(self,subscript):
        {{if is_dim_1}}
        cdef ssize_t index
        if isinstance(subscript,int):
            index = cpython.long.PyLong_AsSsize_t(subscript)
            if index < 0 or index >= {{shape[0]}}:
                raise IndexError(f"Index must be in range 0 .. {{shape[0]}}")
            return {{element_ptr}}[0][index]
        elif isinstance(subscript,slice):
            raise NotImplementedError(f"subscript of type 'slice' is not supported yet")
        else:
            raise IndexError(f"only a single index must be specified")
        {{else}}
        raise NotImplementedError(f"accessing values of multi-dimensional arrays not supported yet")
        {{endif}}
    {{endif}}
"""
