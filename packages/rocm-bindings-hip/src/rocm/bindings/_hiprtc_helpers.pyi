import _cython_3_2_4
import rocm.bindings.util.types
from typing import Any, ClassVar

__reduce_cython__: _cython_3_2_4.cython_function_or_method
__setstate_cython__: _cython_3_2_4.cython_function_or_method
__test__: dict

class HiprtcLinkCreate_option_ptr(rocm.bindings.util.types.Pointer):
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, pyobj) -> Any: ...
    @staticmethod
    def fromObj(pyobj) -> Any: ...
    def __reduce__(self): ...
