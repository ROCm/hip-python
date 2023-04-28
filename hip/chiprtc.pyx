# AMD_COPYRIGHT
from libc.stdint cimport *

cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libhiprtc.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _hiprtcGetErrorString__funptr = NULL
# @brief Returns text string message to explain the error which occurred
# @param [in] result  code to convert to string.
# @return  const char pointer to the NULL-terminated error string
# @warning In HIP, this function returns the name of the error,
# if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
# @see hiprtcResult
cdef const char * hiprtcGetErrorString(hiprtcResult result) nogil:
    global _hiprtcGetErrorString__funptr
    __init_symbol(&_hiprtcGetErrorString__funptr,"hiprtcGetErrorString")
    return (<const char * (*)(hiprtcResult) nogil> _hiprtcGetErrorString__funptr)(result)


cdef void* _hiprtcVersion__funptr = NULL
# @brief Sets the parameters as major and minor version.
# @param [out] major  HIP Runtime Compilation major version.
# @param [out] minor  HIP Runtime Compilation minor version.
cdef hiprtcResult hiprtcVersion(int * major,int * minor) nogil:
    global _hiprtcVersion__funptr
    __init_symbol(&_hiprtcVersion__funptr,"hiprtcVersion")
    return (<hiprtcResult (*)(int *,int *) nogil> _hiprtcVersion__funptr)(major,minor)


cdef void* _hiprtcAddNameExpression__funptr = NULL
# @brief Adds the given name exprssion to the runtime compilation program.
# @param [in] prog  runtime compilation program instance.
# @param [in] name_expression  const char pointer to the name expression.
# @return  HIPRTC_SUCCESS
# If const char pointer is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
# @see hiprtcResult
cdef hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,const char * name_expression) nogil:
    global _hiprtcAddNameExpression__funptr
    __init_symbol(&_hiprtcAddNameExpression__funptr,"hiprtcAddNameExpression")
    return (<hiprtcResult (*)(hiprtcProgram,const char *) nogil> _hiprtcAddNameExpression__funptr)(prog,name_expression)


cdef void* _hiprtcCompileProgram__funptr = NULL
# @brief Compiles the given runtime compilation program.
# @param [in] prog  runtime compilation program instance.
# @param [in] numOptions  number of compiler options.
# @param [in] options  compiler options as const array of strins.
# @return HIPRTC_SUCCESS
# If the compiler failed to build the runtime compilation program,
# it will return HIPRTC_ERROR_COMPILATION.
# @see hiprtcResult
cdef hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,int numOptions,const char ** options) nogil:
    global _hiprtcCompileProgram__funptr
    __init_symbol(&_hiprtcCompileProgram__funptr,"hiprtcCompileProgram")
    return (<hiprtcResult (*)(hiprtcProgram,int,const char **) nogil> _hiprtcCompileProgram__funptr)(prog,numOptions,options)


cdef void* _hiprtcCreateProgram__funptr = NULL
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
    global _hiprtcCreateProgram__funptr
    __init_symbol(&_hiprtcCreateProgram__funptr,"hiprtcCreateProgram")
    return (<hiprtcResult (*)(hiprtcProgram*,const char *,const char *,int,const char **,const char **) nogil> _hiprtcCreateProgram__funptr)(prog,src,name,numHeaders,headers,includeNames)


cdef void* _hiprtcDestroyProgram__funptr = NULL
# @brief Destroys an instance of given hiprtcProgram.
# @param [in] prog  runtime compilation program instance.
# @return HIPRTC_SUCCESS
# If prog is NULL, it will return HIPRTC_ERROR_INVALID_INPUT.
# @see hiprtcResult
cdef hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) nogil:
    global _hiprtcDestroyProgram__funptr
    __init_symbol(&_hiprtcDestroyProgram__funptr,"hiprtcDestroyProgram")
    return (<hiprtcResult (*)(hiprtcProgram*) nogil> _hiprtcDestroyProgram__funptr)(prog)


cdef void* _hiprtcGetLoweredName__funptr = NULL
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
    global _hiprtcGetLoweredName__funptr
    __init_symbol(&_hiprtcGetLoweredName__funptr,"hiprtcGetLoweredName")
    return (<hiprtcResult (*)(hiprtcProgram,const char *,const char **) nogil> _hiprtcGetLoweredName__funptr)(prog,name_expression,lowered_name)


cdef void* _hiprtcGetProgramLog__funptr = NULL
# @brief Gets the log generated by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] log  memory pointer to the generated log.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog,char * log) nogil:
    global _hiprtcGetProgramLog__funptr
    __init_symbol(&_hiprtcGetProgramLog__funptr,"hiprtcGetProgramLog")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> _hiprtcGetProgramLog__funptr)(prog,log)


cdef void* _hiprtcGetProgramLogSize__funptr = NULL
# @brief Gets the size of log generated by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] logSizeRet  size of generated log.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog,int * logSizeRet) nogil:
    global _hiprtcGetProgramLogSize__funptr
    __init_symbol(&_hiprtcGetProgramLogSize__funptr,"hiprtcGetProgramLogSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> _hiprtcGetProgramLogSize__funptr)(prog,logSizeRet)


cdef void* _hiprtcGetCode__funptr = NULL
# @brief Gets the pointer of compilation binary by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  char pointer to binary.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCode(hiprtcProgram prog,char * code) nogil:
    global _hiprtcGetCode__funptr
    __init_symbol(&_hiprtcGetCode__funptr,"hiprtcGetCode")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> _hiprtcGetCode__funptr)(prog,code)


cdef void* _hiprtcGetCodeSize__funptr = NULL
# @brief Gets the size of compilation binary by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  the size of binary.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog,int * codeSizeRet) nogil:
    global _hiprtcGetCodeSize__funptr
    __init_symbol(&_hiprtcGetCodeSize__funptr,"hiprtcGetCodeSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> _hiprtcGetCodeSize__funptr)(prog,codeSizeRet)


cdef void* _hiprtcGetBitcode__funptr = NULL
# @brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  char pointer to bitcode.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcode(hiprtcProgram prog,char * bitcode) nogil:
    global _hiprtcGetBitcode__funptr
    __init_symbol(&_hiprtcGetBitcode__funptr,"hiprtcGetBitcode")
    return (<hiprtcResult (*)(hiprtcProgram,char *) nogil> _hiprtcGetBitcode__funptr)(prog,bitcode)


cdef void* _hiprtcGetBitcodeSize__funptr = NULL
# @brief Gets the size of compiled bitcode by the runtime compilation program instance.
# @param [in] prog  runtime compilation program instance.
# @param [out] code  the size of bitcode.
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog,int * bitcode_size) nogil:
    global _hiprtcGetBitcodeSize__funptr
    __init_symbol(&_hiprtcGetBitcodeSize__funptr,"hiprtcGetBitcodeSize")
    return (<hiprtcResult (*)(hiprtcProgram,int *) nogil> _hiprtcGetBitcodeSize__funptr)(prog,bitcode_size)


cdef void* _hiprtcLinkCreate__funptr = NULL
# @brief Creates the link instance via hiprtc APIs.
# @param [in] hip_jit_options
# @param [out] hiprtc link state instance
# @return HIPRTC_SUCCESS
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkCreate(unsigned int num_options,hiprtcJIT_option * option_ptr,void ** option_vals_pptr,hiprtcLinkState* hip_link_state_ptr) nogil:
    global _hiprtcLinkCreate__funptr
    __init_symbol(&_hiprtcLinkCreate__funptr,"hiprtcLinkCreate")
    return (<hiprtcResult (*)(unsigned int,hiprtcJIT_option *,void **,hiprtcLinkState*) nogil> _hiprtcLinkCreate__funptr)(num_options,option_ptr,option_vals_pptr,hip_link_state_ptr)


cdef void* _hiprtcLinkAddFile__funptr = NULL
# @brief Adds a file with bit code to be linked with options
# @param [in] hiprtc link state, jit input type, file path,
# option reated parameters.
# @param [out] None.
# @return HIPRTC_SUCCESS
# If input values are invalid, it will
# @return HIPRTC_ERROR_INVALID_INPUT
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,const char * file_path,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil:
    global _hiprtcLinkAddFile__funptr
    __init_symbol(&_hiprtcLinkAddFile__funptr,"hiprtcLinkAddFile")
    return (<hiprtcResult (*)(hiprtcLinkState,hiprtcJITInputType,const char *,unsigned int,hiprtcJIT_option *,void **) nogil> _hiprtcLinkAddFile__funptr)(hip_link_state,input_type,file_path,num_options,options_ptr,option_values)


cdef void* _hiprtcLinkAddData__funptr = NULL
# @brief Completes the linking of the given program.
# @param [in] hiprtc link state, jit input type, image_ptr ,
# option reated parameters.
# @param [out] None.
# @return HIPRTC_SUCCESS
# If adding the file fails, it will
# @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,void * image,int image_size,const char * name,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil:
    global _hiprtcLinkAddData__funptr
    __init_symbol(&_hiprtcLinkAddData__funptr,"hiprtcLinkAddData")
    return (<hiprtcResult (*)(hiprtcLinkState,hiprtcJITInputType,void *,int,const char *,unsigned int,hiprtcJIT_option *,void **) nogil> _hiprtcLinkAddData__funptr)(hip_link_state,input_type,image,image_size,name,num_options,options_ptr,option_values)


cdef void* _hiprtcLinkComplete__funptr = NULL
# @brief Completes the linking of the given program.
# @param [in] hiprtc link state instance
# @param [out] linked_binary, linked_binary_size.
# @return HIPRTC_SUCCESS
# If adding the data fails, it will
# @return HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state,void ** bin_out,int * size_out) nogil:
    global _hiprtcLinkComplete__funptr
    __init_symbol(&_hiprtcLinkComplete__funptr,"hiprtcLinkComplete")
    return (<hiprtcResult (*)(hiprtcLinkState,void **,int *) nogil> _hiprtcLinkComplete__funptr)(hip_link_state,bin_out,size_out)


cdef void* _hiprtcLinkDestroy__funptr = NULL
# @brief Deletes the link instance via hiprtc APIs.
# @param [in] hiprtc link state instance
# @param [out] code  the size of binary.
# @return HIPRTC_SUCCESS
# If linking fails, it will
# @return HIPRTC_ERROR_LINKING
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state) nogil:
    global _hiprtcLinkDestroy__funptr
    __init_symbol(&_hiprtcLinkDestroy__funptr,"hiprtcLinkDestroy")
    return (<hiprtcResult (*)(hiprtcLinkState) nogil> _hiprtcLinkDestroy__funptr)(hip_link_state)
