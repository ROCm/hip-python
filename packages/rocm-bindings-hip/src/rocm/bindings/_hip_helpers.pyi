import _cython_3_2_4
import rocm.bindings.util.types
from typing import Any, ClassVar

__reduce_cython__: _cython_3_2_4.cython_function_or_method
__setstate_cython__: _cython_3_2_4.cython_function_or_method
__test__: dict

class HipModuleLaunchKernel_extra(rocm.bindings.util.types.Pointer):
    """HipModuleLaunchKernel_extra(pyobj)

    Handler for `list`/`tuple` of `ctypes` types or types convertible to `~.Pointer`

    Datatype for handling Python `list` or `tuple` objects with entries that are either
    `ctypes` datatypes or that can be converted to type `~.Pointer`.

    The type can be initialized from the following Python objects:

    * `list` or `tuple` object:

      `list` or `tuple` object with entries that are either `ctypes` datatypes or that
      an be converted to type `~.Pointer`. In this case, this type allocates an
      appropriately sized buffer wherein it stores the values of all `ctypes` datatype
      entries of `pyobj` plus all the addresses from the entries that can be converted
      to type `~.Pointer`. The buffer is padded with additional bytes to account for
      the alignment requirements of each entry; for more details, see
      `~.hipModuleLaunchKernel`. Furthermore, the instance's ``self._is_ptr_owner`` C
      attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:

      In this case, init code from `~.Pointer` is used and the C attribute
      ``self._is_ptr_owner`` remains unchanged. See `~.Pointer.__init__` for more
      information.

    Note:
        Type checks are performed in the above order.

    See:
        `~.hipModuleLaunchKernel`"""
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    def __init__(self, pyobj) -> Any:
        """Constructor.

                The type can be initialized from the following Python objects:

                * `list` or `tuple` object:

                  `list` or `tuple` object with entries that are either `ctypes` datatypes or
                  that can be converted to type `~.Pointer`. In this case, this type allocates
                  an appropriately sized buffer wherein it stores the values of all `ctypes`
                  datatype entries of `pyobj` plus all the addresses from the entries that can
                  be converted to type `~.Pointer`. The buffer is padded with additional bytes
                  to account for the alignment requirements of each entry; for more details,
                  see `~.hipModuleLaunchKernel`. Furthermore, the instance's
                  ``self._is_ptr_owner`` C attribute is set to `True` in this case.

                * `object` that is accepted as input by `~.Pointer.__init__`:

                  In this case, init code from `~.Pointer` is used and the C attribute
                  ``self._is_ptr_owner`` remains unchanged. See `~.Pointer.__init__` for more
                  information.

                Note:
                    Type checks are performed in the above order.

                Args:
                    pyobj (`object`):
                        Must be either a `list` or `tuple` of objects that can be converted
                        to `~.Pointer`, or any other `object` that is accepted as input by
                        `~.Pointer.__init__`.

                See:
                    `~.hipModuleLaunchKernel`
        """
    @staticmethod
    def fromObj(pyobj) -> Any:
        """HipModuleLaunchKernel_extra.fromObj(pyobj)

        Creates a HipModuleLaunchKernel_extra from the given object.

        In case ``pyobj`` is itself a ``HipModuleLaunchKernel_extra`` instance,
        this method returns it directly. No new ``HipModuleLaunchKernel_extra`` is
        created."""
    def __reduce__(self):
        """HipModuleLaunchKernel_extra.__reduce_cython__(self)"""
