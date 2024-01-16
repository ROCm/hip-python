cimport libc.stdlib
cimport libc.string
cimport cpython.long

cimport hip.chip

import ctypes

__all__ = [
    # __all__ is important for generating the API documentation in source order
    "HipModuleLaunchKernel_extra",
]

# @param [in] extra Pointer to kernel arguments.   These are passed directly to the kernel and
#                   must be in the memory layout and alignment expected by the kernel.

cdef class HipModuleLaunchKernel_extra(hip._util.types.Pointer):
    """Datatype for handling Python `list` or `tuple` objects with entries that are either `ctypes` datatypes or that can be converted to type `~.Pointer`.

    Datatype for handling Python `list` or `tuple` objects with entries that are either `ctypes` datatypes or that can be converted to type `~.Pointer`.

    The type can be initialized from the following Python objects:
    
    * `list` or `tuple` object:

      `list` or `tuple` object with entries that are either `ctypes` datatypes or that can be converted to type `~.Pointer`.
      In this case, this type allocates an appropriately sized buffer wherein it stores the 
      values of all `ctypes` datatype entries of `pyobj` plus all the addresses from the
      entries that can be converted to type `~.Pointer`. The buffer is padded with additional bytes to account 
      for the alignment requirements of each entry; for more details, see `~.hipModuleLaunchKernel`.
      Furthermore, the instance's `self._is_ptr_owner ` C attribute is set to `True` in this case.

    * `object` that is accepted as input by `~.Pointer.__init__`:
      
      In this case, init code from `~.Pointer` is used and the C attribute `self._is_ptr_owner ` remains unchanged.
      See `~.Pointer.__init__` for more information.
    
    Note:
        Type checks are performed in the above order.

    See:
        `~.hipModuleLaunchKernel`
    """
    # members declared in declaration part (.pxd)

    def __cinit__(self):
        self._ptr = <void*>&self._config
        self._config[0] = <void*>hip.chip.HIP_LAUNCH_PARAM_BUFFER_POINTER
        self._config[2] = <void*>hip.chip.HIP_LAUNCH_PARAM_BUFFER_SIZE
        self._config[3] = <void*>&self._args_buffer_size
        self._config[4] = <void*>hip.chip.HIP_LAUNCH_PARAM_END

    @staticmethod
    cdef HipModuleLaunchKernel_extra fromPtr(void* ptr):
        cdef HipModuleLaunchKernel_extra wrapper = HipModuleLaunchKernel_extra.__new__(HipModuleLaunchKernel_extra)
        wrapper._ptr = ptr
        return wrapper

    cdef size_t _aligned_size(self, size_t size, size_t factor):
        return factor * ((size) // (factor) + ((size) % (factor) != 0)) # //: floor division

    cdef void init_from_pyobj(self, object pyobj):
        cdef tuple ctypes_types = (
            ctypes.c_bool,
            ctypes.c_char,
            ctypes.c_wchar,
            ctypes.c_byte,
            ctypes.c_ubyte,
            ctypes.c_short,
            ctypes.c_ushort,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_long,
            ctypes.c_ulong,
            ctypes.c_longlong,
            ctypes.c_ulonglong,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
            ctypes.c_float,
            ctypes.c_double,
            ctypes.c_longdouble,
            ctypes.c_char_p,
            ctypes.c_wchar_p,
            ctypes.c_void_p,
            ctypes.Structure,
            ctypes.Union,
        )
        cdef size_t buffer_size = 0 
        cdef char* args = NULL
        cdef size_t num_bytes = 0
        cdef void* ptr = NULL
        cdef size_t MAX_ALIGN = 8

        self._py_buffer_acquired = False
        self._is_ptr_owner  = False
        if isinstance(pyobj,HipModuleLaunchKernel_extra):
            self._ptr = (<HipModuleLaunchKernel_extra>pyobj)._ptr
        elif isinstance(pyobj,(tuple,list)):
            self._is_ptr_owner  = True
            # 1. Calcualte number of bytes needed
            for entry in pyobj:
                if isinstance(entry,ctypes_types):
                    buffer_size += self._aligned_size(ctypes.sizeof(entry),MAX_ALIGN) # overestimate
                else:
                    ptr = hip._util.types.Pointer(entry)._ptr # fails if cannot be converted to pointer
                    buffer_size += sizeof(void *)
            #print(f"required: {buffer_size}")
            # 2. Allocate the args array
            args = <char*>libc.stdlib.malloc(buffer_size)
            self._config[1] = <void*>args
            
            # 3. Copy the data into the struct and calculate the offset
            self._args_buffer_size = 0
            for entry in pyobj:
                if isinstance(entry,ctypes_types):
                    ptr = cpython.long.PyLong_AsVoidPtr(ctypes.addressof(entry))
                    num_bytes = ctypes.sizeof(entry)
                else:
                    ptr = &hip._util.types.Pointer(entry)._ptr # fails if cannot be converted to pointer
                    num_bytes = sizeof(void *)
                # TODO it is not confirmed that this is the correct alignment algorithm!
                self._args_buffer_size = self._aligned_size(self._args_buffer_size, min(num_bytes,MAX_ALIGN))
                libc.string.memcpy(<void*>&args[self._args_buffer_size], ptr, num_bytes)
                self._args_buffer_size += num_bytes # TODO: self._aligned_size(num_bytes) # must be 8-byte aligned
        else:
            hip._util.types.Pointer.init_from_pyobj(self,pyobj)

    @staticmethod
    def fromObj(pyobj):
        """Creates a HipModuleLaunchKernel_extra from the given object.

        In case ``pyobj`` is itself a ``HipModuleLaunchKernel_extra`` instance, this method
        returns it directly. No new ``HipModuleLaunchKernel_extra`` is created.
        """
        return HipModuleLaunchKernel_extra.fromPyobj(pyobj)

    @staticmethod
    cdef HipModuleLaunchKernel_extra fromPyobj(object pyobj):
        """Creates a HipModuleLaunchKernel_extra from the given object.

        In case ``pyobj`` is itself a ``HipModuleLaunchKernel_extra`` instance, this method
        returns it directly. No new ``HipModuleLaunchKernel_extra`` is created.

        Args:
            pyobj (`object`):
                Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                or of type ``_util.types.Pointer``, ``HipModuleLaunchKernel_extra``, ``int``, or ``ctypes.c_void_p``

                Furthermore, ``pyobj`` can be a list or tuple in the following shape, where
                each entry can be either be

                * (1) pointer-like, i.e. can be directly translated to ``_util.types.Pointer``,
                * (2) A ctypes C datatype, which is always passed by value.

        Note:
            This routine does not perform a copy but returns the original pyobj
            if ``pyobj`` is an instance of HipModuleLaunchKernel_extra.
        Note:
            This routines assumes that the original input is not garbage
            collected before the deletion of this object.
        """
        cdef HipModuleLaunchKernel_extra wrapper = HipModuleLaunchKernel_extra.__new__(HipModuleLaunchKernel_extra)
        
        if isinstance(pyobj,HipModuleLaunchKernel_extra):
            return pyobj
        else:
            wrapper = HipModuleLaunchKernel_extra.__new__(HipModuleLaunchKernel_extra)
            wrapper.init_from_pyobj(pyobj)
            return wrapper
    
    def __dealloc__(self):
        if self._is_ptr_owner :
            libc.stdlib.free(<void*>self._config[1])

    def __init__(self,object pyobj):
        """Constructor.

        The type can be initialized from the following Python objects:

        * `list` or `tuple` object:

          `list` or `tuple` object with entries that are either `ctypes` datatypes or that can be converted to type `~.Pointer`.
          In this case, this type allocates an appropriately sized buffer wherein it stores the 
          values of all `ctypes` datatype entries of `pyobj` plus all the addresses from the
          entries that can be converted to type `~.Pointer`. The buffer is padded with additional bytes to account 
          for the alignment requirements of each entry; for more details, see `~.hipModuleLaunchKernel`.
          Furthermore, the instance's `self._is_ptr_owner ` C attribute is set to `True` in this case.

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
        HipModuleLaunchKernel_extra.init_from_pyobj(self,pyobj)
