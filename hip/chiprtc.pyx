# AMD_COPYRIGHT
from libc.stdint cimport *

cimport hip._util.posixloader as loader


cdef void* _lib_handle = loader.open_library("libhiprtc.so")


cdef void* hiprtcGetErrorString_funptr = NULL
# @brief Returns text string message to explain the error which occurred
# @param [in] result  code to convert to string.
# @return  const char pointer to the NULL-terminated error string
# @warning In HIP, this function returns the name of the error,
# if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
# @see hiprtcResult
cdef const char * hiprtcGetErrorString(hiprtcResult result) nogil:
    global _lib_handle
    global hiprtcGetErrorString_funptr
    if hiprtcGetErrorString_funptr == NULL:
        with gil:
            hiprtcGetErrorString_funptr = loader.load_symbol(_lib_handle, "hiprtcGetErrorString")
    return (<const char * (*)(hiprtcResult) nogil> hiprtcGetErrorString_funptr)(result)


cdef void* hiprtcVersion_funptr = NULL
# @brief Sets the parameters as major and minor version.
# @param [out] major  HIP Runtime Compilation major version.
# @param [out] minor  HIP Runtime Compilation minor version.
cdef hiprtcResult hiprtcVersion(int * major,int * minor) nogil:
    global _lib_handle
    global hiprtcVersion_funptr
    if hiprtcVersion_funptr == NULL:
        with gil:
            hiprtcVersion_funptr = loader.load_symbol(_lib_handle, "hiprtcVersion")
    return (<hiprtcResult (*)(int *,int *) nogil> hiprtcVersion_funptr)(major,minor)


cdef void* hiprtcAddNameExpression_funptr = NULL
# @brief Adds the given name exprssion to the runtime compilation program.
# @param [in] prog  runtime compilation program instance.
# @param [in] name_expression  const char pointer to the name expression.
# @return  HIPRTC_SUCCESS
# If const char pointer is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
# @see hiprtcResult
cdef hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,const char * name_expression) nogil:
    global _lib_handle
    global hiprtcAddNameExpression_funptr
    if hiprtcAddNameExpression_funptr == NULL:
        with gil:
            hiprtcAddNameExpression_funptr = loader.load_symbol(_lib_handle, "hiprtcAddNameExpression")
    return (<hiprtcResult (*)(hiprtcProgram,const char *) nogil> hiprtcAddNameExpression_funptr)(prog,name_expression)


cdef void* hiprtcCompileProgram_funptr = NULL
# @brief Compiles the given runtime compilation program.
# @param [in] prog  runtime compilation program instance.
# @param [in] numOptions  number of compiler options.
# @param [in] options  compiler options as const array of strins.
# @return HIPRTC_SUCCESS
# If the compiler failed to build the runtime compilation program,
# it will return HIPRTC_ERROR_COMPILATION.
# @see hiprtcResult
cdef hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,int numOptions,const char ** options) nogil:
    global _lib_handle
    global hiprtcCompileProgram_funptr
    if hiprtcCompileProgram_funptr == NULL:
        with gil:
            hiprtcCompileProgram_funptr = loader.load_symbol(_lib_handle, "hiprtcCompileProgram")
    return (<hiprtcResult (*)(hiprtcProgram,int,const char **) nogil> hiprtcCompileProgram_funptr)(prog,numOptions,options)


cdef void* hiprtcCreateProgram_funptr = NULL
# @brief Creates an instance of hiprtcProgram with the given input parameters,
# and sets the output hiprtcProgram prog with it.
# @param [in, out] prog  runtime compilation program instance.
# @param [in] src  const char pointer to the program source.
# @param [in] name  const char pointer to the program name.
# @param [in] numHeaders  number of headers.
# @param [in] headers  array of strings pointing to headers.
# @param [in] includeNames  array of strings pointing to names included in program source.
# @return HIPRTC_SUCCESS
# Any invalide input parameter, it will return HIPRTC_ERROR_INVALID_INPUT
# or HIPRTC_ERROR_INVALID_PROGRAM.
# If failed to create the program, it will return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE.
# @see hiprtcResult
cdef hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog,const char * src,const char * name,int numHeaders,const char ** headers,const char ** includeNames) nogil:
    global _lib_handle
    global hiprtcCreateProgram_funptr
    if hiprtcCreateProgram_funptr == NULL:
        with gil:
            hiprtcCreateProgram_funptr = loader.load_symbol(_lib_handle, "hiprtcCreateProgram")
    return (<hiprtcResult (*)(hiprtcProgram*,const char *,const char *,int,const char **,const char **) nogil> hiprtcCreateProgram_funptr)(prog,src,name,numHeaders,headers,includeNames)


cdef void* hiprtcDestroyProgram_funptr = NULL
# @brief Destroys an instance of given hiprtcProgram.
# @param [in] prog  runtime compilation program instance.
# @return HIPRTC_SUCCESS
# If prog is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
# @see hiprtcResult
cdef hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) nogil:
    global _lib_handle
    global hiprtcDestroyProgram_funptr
    if hiprtcDestroyProgram_funptr == NULL:
        with gil:
            hiprtcDestroyProgram_funptr = loader.load_symbol(_lib_handle, "hiprtcDestroyProgram")
    return (<hiprtcResult (*)(hiprtcProgram*) nogil> hiprtcDestroyProgram_funptr)(prog)


cdef void* hiprtcGetLoweredName_funptr = NULL
# @brief Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input parameters,
# and sets the output lowered_name with it.
# @param [in] prog  runtime compilation program instance.
# @param [in] name_expression  const char pointer to the name expression.
# @param [in, out] lowered_name  const char array to the lowered (mangled) name.
# @return HIPRTC_SUCCESS
# If any invalide nullptr input parameters, it will return HIPRTC_ERROR_INVALID_INPUT
# If name_expression is not found, it will return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
# If failed to get lowered_name from the program, it will return HIPRTC_ERROR_COMPILATION.
# @see hiprtcResult
cdef hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog,const char * name_expression,const char ** lowered_name) nogil:
    global _lib_handle
    global hiprtcGetLoweredName_funptr
    if hiprtcGetLoweredName_funptr == NULL:
        with gil:
            hiprtcGetLoweredName_funptr = loader.load_symbol(_lib_handle, "hiprtcGetLoweredName")
    return (<hiprtcResult (*)(hiprtcProgram,const char *,const char **) nogil> hiprtcGetLoweredName_funptr)(prog,name_expression,lowered_name)


cdef void* hiprtcGetProgramLog_funptr = NULL
# @brief Gets the log generated by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] log  memory pointer to the generated log.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog,char * log) nogil:
    global _lib_handle
    global hiprtcGetProgramLog_funptr
    if hiprtcGetProgramLog_funptr == NULL:
        with gil:
            hiprtcGetProgramLog_funptr = loader.load_symbol(_lib_handle, "hiprtcGetProgramLog")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> hiprtcGetProgramLog_funptr)(prog,log)


cdef void* hiprtcGetProgramLogSize_funptr = NULL
# @brief Gets the size of log generated by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] logSizeRet  size of generated log.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog,int * logSizeRet) nogil:
    global _lib_handle
    global hiprtcGetProgramLogSize_funptr
    if hiprtcGetProgramLogSize_funptr == NULL:
        with gil:
            hiprtcGetProgramLogSize_funptr = loader.load_symbol(_lib_handle, "hiprtcGetProgramLogSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> hiprtcGetProgramLogSize_funptr)(prog,logSizeRet)


cdef void* hiprtcGetCode_funptr = NULL
# @brief Gets the pointer of compilation binary by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  char pointer to binary.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCode(hiprtcProgram prog,char * code) nogil:
    global _lib_handle
    global hiprtcGetCode_funptr
    if hiprtcGetCode_funptr == NULL:
        with gil:
            hiprtcGetCode_funptr = loader.load_symbol(_lib_handle, "hiprtcGetCode")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> hiprtcGetCode_funptr)(prog,code)


cdef void* hiprtcGetCodeSize_funptr = NULL
# @brief Gets the size of compilation binary by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  the size of binary.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog,int * codeSizeRet) nogil:
    global _lib_handle
    global hiprtcGetCodeSize_funptr
    if hiprtcGetCodeSize_funptr == NULL:
        with gil:
            hiprtcGetCodeSize_funptr = loader.load_symbol(_lib_handle, "hiprtcGetCodeSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> hiprtcGetCodeSize_funptr)(prog,codeSizeRet)


cdef void* hiprtcGetBitcode_funptr = NULL
# @brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  char pointer to bitcode.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcode(hiprtcProgram prog,char * bitcode) nogil:
    global _lib_handle
    global hiprtcGetBitcode_funptr
    if hiprtcGetBitcode_funptr == NULL:
        with gil:
            hiprtcGetBitcode_funptr = loader.load_symbol(_lib_handle, "hiprtcGetBitcode")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> hiprtcGetBitcode_funptr)(prog,bitcode)


cdef void* hiprtcGetBitcodeSize_funptr = NULL
# @brief Gets the size of compiled bitcode by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  the size of bitcode.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog,int * bitcode_size) nogil:
    global _lib_handle
    global hiprtcGetBitcodeSize_funptr
    if hiprtcGetBitcodeSize_funptr == NULL:
        with gil:
            hiprtcGetBitcodeSize_funptr = loader.load_symbol(_lib_handle, "hiprtcGetBitcodeSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> hiprtcGetBitcodeSize_funptr)(prog,bitcode_size)


cdef void* hiprtcLinkCreate_funptr = NULL
# @brief Creates the link instance via hiprtc APIs.
# @param [in] hip_jit_options
# @param [out] hiprtc link state instance
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkCreate(unsigned int num_options,hiprtcJIT_option * option_ptr,void ** option_vals_pptr,hiprtcLinkState* hip_link_state_ptr) nogil:
    global _lib_handle
    global hiprtcLinkCreate_funptr
    if hiprtcLinkCreate_funptr == NULL:
        with gil:
            hiprtcLinkCreate_funptr = loader.load_symbol(_lib_handle, "hiprtcLinkCreate")
    return (<hiprtcResult (*)(unsigned int,hiprtcJIT_option *,void **,hiprtcLinkState*) nogil> hiprtcLinkCreate_funptr)(num_options,option_ptr,option_vals_pptr,hip_link_state_ptr)


cdef void* hiprtcLinkAddFile_funptr = NULL
# @brief Adds a file with bit code to be linked with options
# @param [in] hiprtc link state, jit input type, file path,
# option reated parameters.
# @param [out] None.
# @return HIPRTC_SUCCESS
# If input values are invalid, it will
# @return HIPRTC_ERROR_INVALID_INPUT
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,const char * file_path,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil:
    global _lib_handle
    global hiprtcLinkAddFile_funptr
    if hiprtcLinkAddFile_funptr == NULL:
        with gil:
            hiprtcLinkAddFile_funptr = loader.load_symbol(_lib_handle, "hiprtcLinkAddFile")
    return (<hiprtcResult (*)(hiprtcLinkState,hiprtcJITInputType,const char *,unsigned int,hiprtcJIT_option *,void **) nogil> hiprtcLinkAddFile_funptr)(hip_link_state,input_type,file_path,num_options,options_ptr,option_values)


cdef void* hiprtcLinkAddData_funptr = NULL
# @brief Completes the linking of the given program.
# @param [in] hiprtc link state, jit input type, image_ptr ,
# option reated parameters.
# @param [out] None.
# @return HIPRTC_SUCCESS
# If adding the file fails, it will
# @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,void * image,int image_size,const char * name,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil:
    global _lib_handle
    global hiprtcLinkAddData_funptr
    if hiprtcLinkAddData_funptr == NULL:
        with gil:
            hiprtcLinkAddData_funptr = loader.load_symbol(_lib_handle, "hiprtcLinkAddData")
    return (<hiprtcResult (*)(hiprtcLinkState,hiprtcJITInputType,void *,int,const char *,unsigned int,hiprtcJIT_option *,void **) nogil> hiprtcLinkAddData_funptr)(hip_link_state,input_type,image,image_size,name,num_options,options_ptr,option_values)


cdef void* hiprtcLinkComplete_funptr = NULL
# @brief Completes the linking of the given program.
# @param [in] hiprtc link state instance
# @param [out] linked_binary, linked_binary_size.
# @return HIPRTC_SUCCESS
# If adding the data fails, it will
# @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state,void ** bin_out,int * size_out) nogil:
    global _lib_handle
    global hiprtcLinkComplete_funptr
    if hiprtcLinkComplete_funptr == NULL:
        with gil:
            hiprtcLinkComplete_funptr = loader.load_symbol(_lib_handle, "hiprtcLinkComplete")
    return (<hiprtcResult (*)(hiprtcLinkState,void **,int *) nogil> hiprtcLinkComplete_funptr)(hip_link_state,bin_out,size_out)


cdef void* hiprtcLinkDestroy_funptr = NULL
# @brief Deletes the link instance via hiprtc APIs.
# @param [in] hiprtc link state instance
# @param [out] code  the size of binary.
# @return HIPRTC_SUCCESS
# If linking fails, it will
# @return HIPRTC_ERROR_LINKING
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state) nogil:
    global _lib_handle
    global hiprtcLinkDestroy_funptr
    if hiprtcLinkDestroy_funptr == NULL:
        with gil:
            hiprtcLinkDestroy_funptr = loader.load_symbol(_lib_handle, "hiprtcLinkDestroy")
    return (<hiprtcResult (*)(hiprtcLinkState) nogil> hiprtcLinkDestroy_funptr)(hip_link_state)
