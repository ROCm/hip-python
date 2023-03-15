# AMD_COPYRIGHT
from . cimport chiprtc

class hiprtcResult(enum.IntEnum):
    HIPRTC_SUCCESS = 0
    HIPRTC_ERROR_OUT_OF_MEMORY = 1
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
    HIPRTC_ERROR_INVALID_INPUT = 3
    HIPRTC_ERROR_INVALID_PROGRAM = 4
    HIPRTC_ERROR_INVALID_OPTION = 5
    HIPRTC_ERROR_COMPILATION = 6
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
    HIPRTC_ERROR_INTERNAL_ERROR = 11
    HIPRTC_ERROR_LINKING = 100

class hiprtcJIT_option(enum.IntEnum):
    HIPRTC_JIT_MAX_REGISTERS = 0
    HIPRTC_JIT_THREADS_PER_BLOCK = 1
    HIPRTC_JIT_WALL_TIME = 2
    HIPRTC_JIT_INFO_LOG_BUFFER = 3
    HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
    HIPRTC_JIT_ERROR_LOG_BUFFER = 5
    HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
    HIPRTC_JIT_OPTIMIZATION_LEVEL = 7
    HIPRTC_JIT_TARGET_FROM_HIPCONTEXT = 8
    HIPRTC_JIT_TARGET = 9
    HIPRTC_JIT_FALLBACK_STRATEGY = 10
    HIPRTC_JIT_GENERATE_DEBUG_INFO = 11
    HIPRTC_JIT_LOG_VERBOSE = 12
    HIPRTC_JIT_GENERATE_LINE_INFO = 13
    HIPRTC_JIT_CACHE_MODE = 14
    HIPRTC_JIT_NEW_SM3X_OPT = 15
    HIPRTC_JIT_FAST_COMPILE = 16
    HIPRTC_JIT_GLOBAL_SYMBOL_NAMES = 17
    HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS = 18
    HIPRTC_JIT_GLOBAL_SYMBOL_COUNT = 19
    HIPRTC_JIT_LTO = 20
    HIPRTC_JIT_FTZ = 21
    HIPRTC_JIT_PREC_DIV = 22
    HIPRTC_JIT_PREC_SQRT = 23
    HIPRTC_JIT_FMA = 24
    HIPRTC_JIT_NUM_OPTIONS = 25

class hiprtcJITInputType(enum.IntEnum):
    HIPRTC_JIT_INPUT_CUBIN = 0
    HIPRTC_JIT_INPUT_PTX = 1
    HIPRTC_JIT_INPUT_FATBINARY = 2
    HIPRTC_JIT_INPUT_OBJECT = 3
    HIPRTC_JIT_INPUT_LIBRARY = 4
    HIPRTC_JIT_INPUT_NVVM = 5
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = 6
    HIPRTC_JIT_INPUT_LLVM_BITCODE = 100
    HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101
    HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102
    HIPRTC_JIT_NUM_INPUT_TYPES = 9

def hiprtcGetErrorString(result) nogil:
    """@brief Returns text string message to explain the error which occurred
    @param [in] result  code to convert to string.
    @return  const char pointer to the NULL-terminated error string
    @warning In HIP, this function returns the name of the error,
    if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
    @see hiprtcResult
    """
    pass

def hiprtcVersion(major,minor) nogil:
    """@brief Sets the parameters as major and minor version.
    @param [out] major  HIP Runtime Compilation major version.
    @param [out] minor  HIP Runtime Compilation minor version.
    """
    pass

def hiprtcAddNameExpression(prog,name_expression) nogil:
    """@brief Adds the given name exprssion to the runtime compilation program.
    @param [in] prog  runtime compilation program instance.
    @param [in] name_expression  const char pointer to the name expression.
    @return  HIPRTC_SUCCESS
    If const char pointer is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    pass

def hiprtcCompileProgram(prog,numOptions,options) nogil:
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

def hiprtcCreateProgram(prog,src,name,numHeaders,headers,includeNames) nogil:
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

def hiprtcDestroyProgram(prog) nogil:
    """@brief Destroys an instance of given hiprtcProgram.
    @param [in] prog  runtime compilation program instance.
    @return HIPRTC_SUCCESS
    If prog is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
    @see hiprtcResult
    """
    pass

def hiprtcGetLoweredName(prog,name_expression,lowered_name) nogil:
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

def hiprtcGetProgramLog(prog,log) nogil:
    """@brief Gets the log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] log  memory pointer to the generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetProgramLogSize(prog,logSizeRet) nogil:
    """@brief Gets the size of log generated by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] logSizeRet  size of generated log.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetCode(prog,code) nogil:
    """@brief Gets the pointer of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetCodeSize(prog,codeSizeRet) nogil:
    """@brief Gets the size of compilation binary by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetBitcode(prog,bitcode) nogil:
    """@brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  char pointer to bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcGetBitcodeSize(prog,bitcode_size) nogil:
    """@brief Gets the size of compiled bitcode by the runtime compilation program instance.
    @param [in] prog  runtime compilation program instance.
    @param [out] code  the size of bitcode.
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcLinkCreate(num_options,option_ptr,option_vals_pptr,hip_link_state_ptr) nogil:
    """@brief Creates the link instance via hiprtc APIs.
    @param [in] hip_jit_options
    @param [out] hiprtc link state instance
    @return HIPRTC_SUCCESS
    @see hiprtcResult
    """
    pass

def hiprtcLinkAddFile(hip_link_state,input_type,file_path,num_options,options_ptr,option_values) nogil:
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

def hiprtcLinkAddData(hip_link_state,input_type,image,image_size,name,num_options,options_ptr,option_values) nogil:
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

def hiprtcLinkComplete(hip_link_state,bin_out,size_out) nogil:
    """@brief Completes the linking of the given program.
    @param [in] hiprtc link state instance
    @param [out] linked_binary, linked_binary_size.
    @return HIPRTC_SUCCESS
    If adding the data fails, it will
    @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    @see hiprtcResult
    """
    pass

def hiprtcLinkDestroy(hip_link_state) nogil:
    """@brief Deletes the link instance via hiprtc APIs.
    @param [in] hiprtc link state instance
    @param [out] code  the size of binary.
    @return HIPRTC_SUCCESS
    If linking fails, it will
    @return HIPRTC_ERROR_LINKING
    @see hiprtcResult
    """
    pass