
# Note: wrapper_class_decl_template must declare all ``@staticmethod`` ``cdef`` functions
# Note: Syntax ``bint owner=*`` is necessary to specify default value in implementation part

wrapper_class_decl_template = """
{{default cptr_type = cname + "*"}}
{{default has_new = True}}
{{default has_from_pyobj = True}}
cdef class {{name}}({{util_types_prefix}}Pointer):
    cdef bint ptr_owner

    cdef {{cptr_type}} getElementPtr(self)

    @staticmethod
    cdef {{name}} from_ptr(void* ptr, bint owner=*)
    {{if has_from_pyobj}}
    @staticmethod
    cdef {{name}} from_pyobj(object pyobj)
    {{endif}}
    {{if has_new}}
    @staticmethod
    cdef __allocate(void* ptr)
    @staticmethod
    cdef {{name}} new()
    @staticmethod
    cdef {{name}} from_value({{cname}} other)
    {{endif}}
"""

# FIXME hide the pointer and metadata members better to prevent collisions with C type attributes

wrapper_class_impl_base_template = """
{{default cptr_type = cname + "*"}}
{{default is_funptr = False}}
{{default is_union = False}}
{{default has_new = True}}
{{default has_from_pyobj = True}}
{{default can_wrap_device_data = True}}
{{default defaults = dict()}}
{{default properties_name = None}}
{{default all_properties_rendered = False}}
cdef class {{name}}({{util_types_prefix}}Pointer):
    \"""Python wrapper for cdef class {{cname}}.

    Python wrapper for cdef class {{cname}}.

    If this type is initialized via its `__init__` method, it allocates a member of the underlying C type and
    destroys it again if the wrapper type is deallocted.

    This type also serves as adapter when appearing as argument type in a function signature.
    In this case, the type can further be initialized from the following Python objects
    that you can pass as argument instead:

    * `None`:

      This will set the ``self._ptr`` attribute to ``NULL``.

    * `int`:

      Interprets the integer value as pointer address and writes it to ``self._ptr``.
      No ownership is transferred.

    * `ctypes.c_void_p`:

      Takes the pointer address ``pyobj.value`` and writes it to ``self._ptr``.
      No ownership is transferred.

    {{if is_funptr}}
    {{else}}
    {{if can_wrap_device_data}}
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

    {{endif}}
    {{endif}}

    Type checks are performed in the above order.

    C Attributes:
        _ptr (C type ``void *``, protected):
            Stores a pointer to the data of the original Python object.
        _ptr_owner (C type ``bint``, protected):
            If this wrapper is the owner of the underlying data.
        _py_buffer (C type ``Py_buffer`, protected):
            Stores a pointer to the data of the original Python object.
        _py_buffer_acquired (C type ``bint``, protected):
            Stores a pointer to the data of the original Python object.
    \"""
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    cdef {{cptr_type}} getElementPtr(self):
        return <{{cptr_type}}>self._ptr

    @staticmethod
    cdef {{name}} from_ptr(void* ptr, bint owner=False):
        \"""Factory function to create ``{{name}}`` objects from
        given ``{{cname}}`` pointer.
        {{if has_new}}

        Setting ``owner`` flag to ``True`` causes
        the extension type to free the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        {{endif}}
        \"""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    {{if has_from_pyobj}}
    @staticmethod
    cdef {{name}} from_pyobj(object pyobj):
        \"""Derives a {{name}} from a Python object.

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
        cdef {{name}} wrapper = {{name}}.__new__({{name}})
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,{{name}}):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        {{if is_funptr}}
        elif str(type(pyobj)).startswith("<class 'ctypes.CFUNCTYPE.") and str(type(pyobj)).endswith(".CFunctionType'>" ):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(ctypes.cast(pyobj, ctypes.c_void_p).value)
        {{else}}
        {{if can_wrap_device_data}}
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        {{endif}}
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer(
                pyobj,
                &wrapper._py_buffer,
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = wrapper._py_buffer.buf
        {{endif}}
        elif isinstance(pyobj,{{util_types_prefix}}Pointer):
            wrapper._ptr = cpython.long.PyLong_AsVoidPtr(int(pyobj))
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    {{endif}}
    def __dealloc__(self):
        # Release the buffer handle
        {{if has_from_pyobj}}
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        {{endif}}
        {{if has_new}}
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
        {{endif}}
    {{if has_new}}

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
        return {{name}}.from_ptr(ptr, owner=True)

    @staticmethod
    cdef {{name}} from_value({{cname}} other):
        \"""Allocate new C type and copy from ``other``.
        \"""
        wrapper = {{name}}.new()
        string.memcpy(wrapper._ptr, &other, sizeof({{cname}}))
        return wrapper

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
        self.ptr_owner = True
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

    def __int__(self):
        \"""Returns the data's address as long integer.
        \"""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<{{name}} object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        \"""Returns the data's address as `ctypes.c_void_p`
        Note:
            Implemented as function to not collide with
            autogenerated property names.
        \"""
        return ctypes.c_void_p(int(self))
    {{if has_new}}
    def c_sizeof(self):
        \"""Returns the size of the underlying C type in bytes.
        Note:
            Implemented as function to not collide with
            autogenerated property names.
        \"""
        return sizeof({{cname}})
    {{endif}}
"""

# NOTE: This template is only used by RecordMixin not by FunctionPointerMixin
# Hence, cptr_type is not specified as {{default cptr_type = ...}}.
wrapper_class_property_template = """\
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
    return {{handler}}.from_ptr({{element_ptr}}[i].{{attr}})
def set_{{attr}}(self, i, object value):
    \"""Set value ``{{attr}}`` of ``{{element_ptr}}[i]``.

    Note:
        This can be dangerous if the pointer is from a python object
        that is later on garbage collected.
    \"""
    {{element_ptr}}[i].{{attr}} = <{{typename}}>cpython.long.PyLong_AsVoidPtr(int({{handler}}.from_pyobj(value)))
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
    return {{typename}}.from_ptr(&{{element_ptr}}[i].{{attr}})
@property
def {{attr}}(self):
    \"""{{brief_comment}}\"""
    return self.get_{{attr}}(0)
{{endif}}
"""