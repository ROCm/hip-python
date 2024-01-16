cimport libc.stdlib
cimport libc.string
cimport cpython.long
cimport cpython.list

cimport hip.chiprtc
import hip.hiprtc

import ctypes

__all__ = [
    # __all__ is important for generating the API documentation in source order
    "HiprtcLinkCreate_option_ptr",
    "HiprtcLinkCreate_option_vals_pptr",
]

# @param [in] extra Pointer to kernel arguments.   These are passed directly to the kernel and
#                   must be in the memory layout and alignment expected by the kernel.

cdef class HiprtcLinkCreate_option_ptr(hip._util.types.Pointer):
    """Datatype for handling Python `list` or `tuple` objects with `~.hiprtcJIT_option` entries.

    Datatype for handling Python `list` or `tuple` objects with `~.hiprtcJIT_option` entries.

    This type can be initialized from the following Python objects:
    
    * `list` or `tuple` object:

      `list` or `tuple` object with `~.hiprtcJIT_option` entries.

    * `object` that is accepted as input by `~.Pointer.__init__`:
      
      In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner ` remains unchanged.
      See `~.Pointer.__init__` for more information.
    
    Note:
        Type checks are performed in the above order.

    See:
        `~.hiprtcLinkCreate`
    """
    # members declared in declaration part (.pxd)

    @staticmethod
    cdef HiprtcLinkCreate_option_ptr fromPtr(void* ptr):
        cdef HiprtcLinkCreate_option_ptr wrapper = HiprtcLinkCreate_option_ptr.__new__(HiprtcLinkCreate_option_ptr)
        wrapper._ptr = ptr
        return wrapper

    cdef void init_from_pyobj(self, object pyobj):
        cdef Py_ssize_t num_entries = 0 
        cdef unsigned int* element_ptr = NULL
        cdef size_t entry_size = sizeof(unsigned int) # see: hip.hiprtc.hiprtcJIT_option.ctypes_type()

        self._py_buffer_acquired = False
        self._is_ptr_owner  = False
        if isinstance(pyobj,HiprtcLinkCreate_option_ptr):
            self._ptr = (<HiprtcLinkCreate_option_ptr>pyobj)._ptr
        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner  = True
            num_entries = len(pyobj)
            self._ptr = libc.stdlib.malloc(num_entries*entry_size)
            element_ptr = <unsigned int*>self._ptr # see: hip.hiprtc.hiprtcJIT_option.ctypes_type()
            
            # 3. Copy the data into the struct and calculate the offset
            for i in range(0,num_entries):
                if not isinstance(pyobj,hip.hiprtc.hiprtcJIT_option):
                    raise ValueError("list/tuple entries must be of type 'hip.hiprtc.hiprtcJIT_option'")
                element_ptr[i] = <unsigned int>cpython.long.PyLong_AsUnsignedLong(pyobj[i])
        else:
            hip._util.types.Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a HiprtcLinkCreate_option_ptr from the given object.

        In case ``pyobj`` is itself a ``HiprtcLinkCreate_option_ptr`` instance, this method
        returns it directly. No new ``HiprtcLinkCreate_option_ptr`` is created.
        """
        return HiprtcLinkCreate_option_ptr.fromPyobj(pyobj)

    @staticmethod
    cdef HiprtcLinkCreate_option_ptr fromPyobj(object pyobj):
        """Creates a HiprtcLinkCreate_option_ptr from the given object.

        In case ``pyobj`` is itself a ``HiprtcLinkCreate_option_ptr`` instance, this method
        returns it directly. No new ``HiprtcLinkCreate_option_ptr`` is created.

        Args:
            pyobj (`object`):
                Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                or of type ``_util.types.Pointer``, ``HiprtcLinkCreate_option_ptr``, ``int``, or ``ctypes.c_void_p``

                Furthermore, ``pyobj`` can be a list or tuple in the following shape, where
                each entry must be of type `~.hiprtcJIT_option`.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of HiprtcLinkCreate_option_ptr.
        """
        cdef HiprtcLinkCreate_option_ptr wrapper = HiprtcLinkCreate_option_ptr.__new__(HiprtcLinkCreate_option_ptr)
        
        if isinstance(pyobj,HiprtcLinkCreate_option_ptr):
            return pyobj
        else:
            wrapper = HiprtcLinkCreate_option_ptr.__new__(HiprtcLinkCreate_option_ptr)
            wrapper.init_from_pyobj(pyobj)
            return wrapper
    
    def __dealloc__(self):
        if self._is_ptr_owner :
            libc.stdlib.free(self._ptr)

    def __init__(self,object pyobj):
        """Constructor.

        This type can be initialized from the following Python objects:
    
        * `list` or `tuple` object:

          `list` or `tuple` object with `~.hiprtcJIT_option` entries.

        * `object` that is accepted as input by `~.Pointer.__init__`:
        
          In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner ` remains unchanged.
          See `~.Pointer.__init__` for more information.
        
        Note:
            Type checks are performed in the above order.

        Args:
            pyobj (`object`): 
                Must be either a `list` or `tuple` of objects that can be converted
                to `~.Pointer`, or any other `object` that is accepted as input by `~.Pointer.__init__`.

        See:
            `~.hipModuleLaunchKernel`
        """
        HiprtcLinkCreate_option_ptr.init_from_pyobj(self,pyobj)
