# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
import enum

from . cimport chiprtc
class hiprtcResult(enum.IntEnum):
    HIPRTC_SUCCESS = chiprtc.HIPRTC_SUCCESS
    HIPRTC_ERROR_OUT_OF_MEMORY = chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    HIPRTC_ERROR_INVALID_INPUT = chiprtc.HIPRTC_ERROR_INVALID_INPUT
    HIPRTC_ERROR_INVALID_PROGRAM = chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    HIPRTC_ERROR_INVALID_OPTION = chiprtc.HIPRTC_ERROR_INVALID_OPTION
    HIPRTC_ERROR_COMPILATION = chiprtc.HIPRTC_ERROR_COMPILATION
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    HIPRTC_ERROR_INTERNAL_ERROR = chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
    HIPRTC_ERROR_LINKING = chiprtc.HIPRTC_ERROR_LINKING

class hiprtcJIT_option(enum.IntEnum):
    HIPRTC_JIT_MAX_REGISTERS = chiprtc.HIPRTC_JIT_MAX_REGISTERS
    HIPRTC_JIT_THREADS_PER_BLOCK = chiprtc.HIPRTC_JIT_THREADS_PER_BLOCK
    HIPRTC_JIT_WALL_TIME = chiprtc.HIPRTC_JIT_WALL_TIME
    HIPRTC_JIT_INFO_LOG_BUFFER = chiprtc.HIPRTC_JIT_INFO_LOG_BUFFER
    HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = chiprtc.HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    HIPRTC_JIT_ERROR_LOG_BUFFER = chiprtc.HIPRTC_JIT_ERROR_LOG_BUFFER
    HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = chiprtc.HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    HIPRTC_JIT_OPTIMIZATION_LEVEL = chiprtc.HIPRTC_JIT_OPTIMIZATION_LEVEL
    HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = chiprtc.HIPRTC_JIT_TARGET_FROM_HIPCONTEXT
    HIPRTC_JIT_TARGET = chiprtc.HIPRTC_JIT_TARGET
    HIPRTC_JIT_FALLBACK_STRATEGY = chiprtc.HIPRTC_JIT_FALLBACK_STRATEGY
    HIPRTC_JIT_GENERATE_DEBUG_INFO = chiprtc.HIPRTC_JIT_GENERATE_DEBUG_INFO
    HIPRTC_JIT_LOG_VERBOSE = chiprtc.HIPRTC_JIT_LOG_VERBOSE
    HIPRTC_JIT_GENERATE_LINE_INFO = chiprtc.HIPRTC_JIT_GENERATE_LINE_INFO
    HIPRTC_JIT_CACHE_MODE = chiprtc.HIPRTC_JIT_CACHE_MODE
    HIPRTC_JIT_NEW_SM3X_OPT = chiprtc.HIPRTC_JIT_NEW_SM3X_OPT
    HIPRTC_JIT_FAST_COMPILE = chiprtc.HIPRTC_JIT_FAST_COMPILE
    HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_NAMES
    HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS
    HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = chiprtc.HIPRTC_JIT_GLOBAL_SYMBOL_COUNT
    HIPRTC_JIT_LTO = chiprtc.HIPRTC_JIT_LTO
    HIPRTC_JIT_FTZ = chiprtc.HIPRTC_JIT_FTZ
    HIPRTC_JIT_PREC_DIV = chiprtc.HIPRTC_JIT_PREC_DIV
    HIPRTC_JIT_PREC_SQRT = chiprtc.HIPRTC_JIT_PREC_SQRT
    HIPRTC_JIT_FMA = chiprtc.HIPRTC_JIT_FMA
    HIPRTC_JIT_NUM_OPTIONS = chiprtc.HIPRTC_JIT_NUM_OPTIONS

class hiprtcJITInputType(enum.IntEnum):
    HIPRTC_JIT_INPUT_CUBIN = chiprtc.HIPRTC_JIT_INPUT_CUBIN
    HIPRTC_JIT_INPUT_PTX = chiprtc.HIPRTC_JIT_INPUT_PTX
    HIPRTC_JIT_INPUT_FATBINARY = chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    HIPRTC_JIT_INPUT_OBJECT = chiprtc.HIPRTC_JIT_INPUT_OBJECT
    HIPRTC_JIT_INPUT_LIBRARY = chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    HIPRTC_JIT_INPUT_NVVM = chiprtc.HIPRTC_JIT_INPUT_NVVM
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    HIPRTC_JIT_INPUT_LLVM_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_BITCODE
    HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE
    HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = chiprtc.HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
    HIPRTC_JIT_NUM_INPUT_TYPES = chiprtc.HIPRTC_JIT_NUM_INPUT_TYPES

cdef class ihiprtcLinkState:
    cdef chiprtc.ihiprtcLinkState* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef ihiprtcLinkState from_ptr(chiprtc.ihiprtcLinkState *_ptr, bint owner=False):
        """Factory function to create ihiprtcLinkState objects from
        given chiprtc.ihiprtcLinkState pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihiprtcLinkState wrapper = ihiprtcLinkState.__new__(ihiprtcLinkState)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL


cdef class _hiprtcProgram:
    cdef chiprtc._hiprtcProgram* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr_owner = False

    @staticmethod
    cdef _hiprtcProgram from_ptr(chiprtc._hiprtcProgram *_ptr, bint owner=False):
        """Factory function to create _hiprtcProgram objects from
        given chiprtc._hiprtcProgram pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef _hiprtcProgram wrapper = _hiprtcProgram.__new__(_hiprtcProgram)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
