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
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef ihiprtcLinkState from_ptr(chiprtc.ihiprtcLinkState *_ptr, bint owner=False):
        """Factory function to create ``ihiprtcLinkState`` objects from
        given ``chiprtc.ihiprtcLinkState`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ihiprtcLinkState wrapper = ihiprtcLinkState.__new__(ihiprtcLinkState)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hiprtcLinkState = ihiprtcLinkState

def hiprtcGetErrorString(result):
    """@brief Returns text string message to explain the error which occurred
    @param [in] result  code to convert to string.
    @return  const char pointer to the NULL-terminated error string
    @warning In HIP, this function returns the name of the error,
    if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
    @see hiprtcResult
    """
    pass

def hiprtcVersion(major, minor):
    """@brief Sets the parameters as major and minor version.
    @param [out] major  HIP Runtime Compilation major version.
    @param [out] minor  HIP Runtime Compilation minor version.
    """
    pass


cdef class _hiprtcProgram:
    cdef chiprtc._hiprtcProgram* _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    @staticmethod
    cdef _hiprtcProgram from_ptr(chiprtc._hiprtcProgram *_ptr, bint owner=False):
        """Factory function to create ``_hiprtcProgram`` objects from
        given ``chiprtc._hiprtcProgram`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef _hiprtcProgram wrapper = _hiprtcProgram.__new__(_hiprtcProgram)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


hiprtcProgram = _hiprtcProgram

def hiprtcAddNameExpression(prog, const char * name_expression):
    """@brief Adds the given name exprssion to the runtime compilation program.
    @param [in] prog  runtime compilation program instance.
    @param [in] name_expression  const char pointer to the name expression.
    @return  HIPRTC_SUCCESS
    If const char pointer is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    pass

def hiprtcCompileProgram(prog, int numOptions):
    """@brief Compiles the given runtime compilation program.
    @param [in] prog  runtime compilation program instance.
    @param [in] numOptions  number of compiler options.
    @param [in] options  compiler options as const array of strins.
    @return HIPRTC_SUCCESS
    If the compiler failed to build the runtime compilation program,
    it will return HIPRTC_ERROR_COMPILATION.
    @see hiprtcResult
    """
    pass

def hiprtcCreateProgram(const char * src, const char * name, int numHeaders):
    """@brief Creates an instance of hiprtcProgram with the given input parameters,
    and sets the output hiprtcProgram prog with it.
    @param [in, out] prog  runtime compilation program instance.
    @param [in] src  const char pointer to the program source.
    @param [in] name  const char pointer to the program name.
    @param [in] numHeaders  number of headers.
    @param [in] headers  array of strings pointing to headers.
    @param [in] includeNames  array of strings pointing to names included in program source.
    @return HIPRTC_SUCCESS
    Any invalide input parameter, it will return HIPRTC_ERROR_INVALID_INPUT
    or HIPRTC_ERROR_INVALID_PROGRAM.
    If failed to create the program, it will return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE.
    @see hiprtcResult
    """
    pass

def hiprtcDestroyProgram():
    """@brief Destroys an instance of given hiprtcProgram.
    @param [in] prog  runtime compilation program instance.
    @return HIPRTC_SUCCESS
    If prog is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    pass

def hiprtcGetLoweredName(prog, const char * name_expression):
    """@brief Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input parameters,
    and sets the output lowered_name with it.
    @param [in] prog  runtime compilation program instance.
    @param [in] name_expression  const char pointer to the name expression.
    @param [in, out] lowered_name  const char array to the lowered (mangled) name.
    @return HIPRTC_SUCCESS
    If any invalide nullptr input parameters, it will return HIPRTC_ERROR_INVALID_INPUT
    If name_expression is not found, it will return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    If failed to get lowered_name from the program, it will return HIPRTC_ERROR_COMPILATION.
    @see hiprtcResult
    """
    pass

def hiprtcGetProgramLog(prog, char * log):
    """@brief Gets the log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] log  memory pointer to the generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetProgramLogSize(prog, logSizeRet):
    """@brief Gets the size of log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] logSizeRet  size of generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetCode(prog, char * code):
    """@brief Gets the pointer of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetCodeSize(prog, codeSizeRet):
    """@brief Gets the size of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetBitcode(prog, char * bitcode):
    """@brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetBitcodeSize(prog, bitcode_size):
    """@brief Gets the size of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcLinkCreate(unsigned int num_options, option_ptr):
    """@brief Creates the link instance via hiprtc APIs.
    @param [in] hip_jit_options
    @param [out] hiprtc link state instance
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcLinkAddFile(hip_link_state, input_type, const char * file_path, unsigned int num_options, options_ptr):
    """@brief Adds a file with bit code to be linked with options
    @param [in] hiprtc link state, jit input type, file path,
    option reated parameters.
    @param [out] None.
    @return HIPRTC_SUCCESS
    If input values are invalid, it will
    @return HIPRTC_ERROR_INVALID_INPUT
    @see hiprtcResult
    """
    pass

def hiprtcLinkAddData(hip_link_state, input_type, image, int image_size, const char * name, unsigned int num_options, options_ptr):
    """@brief Completes the linking of the given program.
    @param [in] hiprtc link state, jit input type, image_ptr ,
    option reated parameters.
    @param [out] None.
    @return HIPRTC_SUCCESS
    If adding the file fails, it will
    @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    @see hiprtcResult
    """
    pass

def hiprtcLinkComplete(hip_link_state, size_out):
    """@brief Completes the linking of the given program.
    @param [in] hiprtc link state instance
    @param [out] linked_binary, linked_binary_size.
    @return HIPRTC_SUCCESS
    If adding the data fails, it will
    @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    @see hiprtcResult
    """
    pass

def hiprtcLinkDestroy(hip_link_state):
    """@brief Deletes the link instance via hiprtc APIs.
    @param [in] hiprtc link state instance
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    If linking fails, it will
    @return HIPRTC_ERROR_LINKING
    @see hiprtcResult
    """
    pass