# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
cdef extern from "hip/hiprtc.h":

    cdef enum hiprtcResult:
        HIPRTC_SUCCESS
        HIPRTC_ERROR_OUT_OF_MEMORY
        HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
        HIPRTC_ERROR_INVALID_INPUT
        HIPRTC_ERROR_INVALID_PROGRAM
        HIPRTC_ERROR_INVALID_OPTION
        HIPRTC_ERROR_COMPILATION
        HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
        HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
        HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
        HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
        HIPRTC_ERROR_INTERNAL_ERROR
        HIPRTC_ERROR_LINKING

    cdef enum hiprtcJIT_option:
        HIPRTC_JIT_MAX_REGISTERS
        HIPRTC_JIT_THREADS_PER_BLOCK
        HIPRTC_JIT_WALL_TIME
        HIPRTC_JIT_INFO_LOG_BUFFER
        HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
        HIPRTC_JIT_ERROR_LOG_BUFFER
        HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
        HIPRTC_JIT_OPTIMIZATION_LEVEL
        HIPRTC_JIT_TARGET_FROM_HIPCONTEXT
        HIPRTC_JIT_TARGET
        HIPRTC_JIT_FALLBACK_STRATEGY
        HIPRTC_JIT_GENERATE_DEBUG_INFO
        HIPRTC_JIT_LOG_VERBOSE
        HIPRTC_JIT_GENERATE_LINE_INFO
        HIPRTC_JIT_CACHE_MODE
        HIPRTC_JIT_NEW_SM3X_OPT
        HIPRTC_JIT_FAST_COMPILE
        HIPRTC_JIT_GLOBAL_SYMBOL_NAMES
        HIPRTC_JIT_GLOBAL_SYMBOL_ADDRESS
        HIPRTC_JIT_GLOBAL_SYMBOL_COUNT
        HIPRTC_JIT_LTO
        HIPRTC_JIT_FTZ
        HIPRTC_JIT_PREC_DIV
        HIPRTC_JIT_PREC_SQRT
        HIPRTC_JIT_FMA
        HIPRTC_JIT_NUM_OPTIONS
        HIPRTC_JIT_IR_TO_ISA_OPT_EXT
        HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT

    cdef enum hiprtcJITInputType:
        HIPRTC_JIT_INPUT_CUBIN
        HIPRTC_JIT_INPUT_PTX
        HIPRTC_JIT_INPUT_FATBINARY
        HIPRTC_JIT_INPUT_OBJECT
        HIPRTC_JIT_INPUT_LIBRARY
        HIPRTC_JIT_INPUT_NVVM
        HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
        HIPRTC_JIT_INPUT_LLVM_BITCODE
        HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE
        HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
        HIPRTC_JIT_NUM_INPUT_TYPES

    cdef struct ihiprtcLinkState:
        pass

    ctypedef ihiprtcLinkState * hiprtcLinkState

# 
#  @ingroup Runtime
# 
# @brief Returns text string message to explain the error which occurred
# 
# @param [in] result  code to convert to string.
# @return  const char pointer to the NULL-terminated error string
# 
# @warning In HIP, this function returns the name of the error,
# if the hiprtc result is defined, it will return "Invalid HIPRTC error code"
# 
# @see hiprtcResult
cdef const char * hiprtcGetErrorString(hiprtcResult result) nogil


# 
# @ingroup Runtime
# @brief Sets the parameters as major and minor version.
# 
# @param [out] major  HIP Runtime Compilation major version.
# @param [out] minor  HIP Runtime Compilation minor version.
#
cdef hiprtcResult hiprtcVersion(int * major,int * minor) nogil


cdef extern from "hip/hiprtc.h":

    cdef struct _hiprtcProgram:
        pass

    ctypedef _hiprtcProgram * hiprtcProgram

# 
# @ingroup Runtime
# @brief Adds the given name exprssion to the runtime compilation program.
# 
# @param [in] prog  runtime compilation program instance.
# @param [in] name_expression  const char pointer to the name expression.
# @return  #HIPRTC_SUCCESS
# 
# If const char pointer is NULL, it will return #HIPRTC_ERROR_INVALID_INPUT.
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,const char * name_expression) nogil


# 
# @ingroup Runtime
# @brief Compiles the given runtime compilation program.
# 
# @param [in] prog  runtime compilation program instance.
# @param [in] numOptions  number of compiler options.
# @param [in] options  compiler options as const array of strins.
# @return #HIPRTC_SUCCESS
# 
# If the compiler failed to build the runtime compilation program,
# it will return #HIPRTC_ERROR_COMPILATION.
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcCompileProgram(hiprtcProgram prog,int numOptions,const char ** options) nogil


# 
# @ingroup Runtime
# @brief Creates an instance of hiprtcProgram with the given input parameters,
# and sets the output hiprtcProgram prog with it.
# 
# @param [in, out] prog  runtime compilation program instance.
# @param [in] src  const char pointer to the program source.
# @param [in] name  const char pointer to the program name.
# @param [in] numHeaders  number of headers.
# @param [in] headers  array of strings pointing to headers.
# @param [in] includeNames  array of strings pointing to names included in program source.
# @return #HIPRTC_SUCCESS
# 
# Any invalide input parameter, it will return #HIPRTC_ERROR_INVALID_INPUT
# or #HIPRTC_ERROR_INVALID_PROGRAM.
# 
# If failed to create the program, it will return #HIPRTC_ERROR_PROGRAM_CREATION_FAILURE.
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog,const char * src,const char * name,int numHeaders,const char ** headers,const char ** includeNames) nogil


# 
# @brief Destroys an instance of given hiprtcProgram.
# @ingroup Runtime
# @param [in] prog  runtime compilation program instance.
# @return #HIPRTC_SUCCESS
# 
# If prog is NULL, it will return #HIPRTC_ERROR_INVALID_INPUT.
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) nogil


# 
# @brief Gets the lowered (mangled) name from an instance of hiprtcProgram with the given input parameters,
# and sets the output lowered_name with it.
# @ingroup Runtime
# @param [in] prog  runtime compilation program instance.
# @param [in] name_expression  const char pointer to the name expression.
# @param [in, out] lowered_name  const char array to the lowered (mangled) name.
# @return #HIPRTC_SUCCESS
# 
# If any invalide nullptr input parameters, it will return #HIPRTC_ERROR_INVALID_INPUT
# 
# If name_expression is not found, it will return #HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
# 
# If failed to get lowered_name from the program, it will return #HIPRTC_ERROR_COMPILATION.
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog,const char * name_expression,const char ** lowered_name) nogil


# 
# @brief Gets the log generated by the runtime compilation program instance.
# @ingroup Runtime
# @param [in] prog  runtime compilation program instance.
# @param [out] log  memory pointer to the generated log.
# @return HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog,char * log) nogil


# 
# @brief Gets the size of log generated by the runtime compilation program instance.
# 
# @param [in] prog  runtime compilation program instance.
# @param [out] logSizeRet  size of generated log.
# @return HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog,unsigned long * logSizeRet) nogil


# 
# @brief Gets the pointer of compilation binary by the runtime compilation program instance.
# @ingroup Runtime
# @param [in] prog  runtime compilation program instance.
# @param [out] code  char pointer to binary.
# @return HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCode(hiprtcProgram prog,char * code) nogil


# 
# @brief Gets the size of compilation binary by the runtime compilation program instance.
# @ingroup Runtime
# @param [in] prog  runtime compilation program instance.
# @param [out] codeSizeRet  the size of binary.
# @return HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog,unsigned long * codeSizeRet) nogil


# 
# @brief Gets the pointer of compiled bitcode by the runtime compilation program instance.
# 
# @param [in] prog  runtime compilation program instance.
# @param [out] bitcode  char pointer to bitcode.
# @return HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcode(hiprtcProgram prog,char * bitcode) nogil


# 
# @brief Gets the size of compiled bitcode by the runtime compilation program instance.
# @ingroup Runtime
# 
# @param [in] prog  runtime compilation program instance.
# @param [out] bitcode_size  the size of bitcode.
# @return #HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcGetBitcodeSize(hiprtcProgram prog,unsigned long * bitcode_size) nogil


# 
# @brief Creates the link instance via hiprtc APIs.
# @ingroup Runtime
# @param [in] num_options  Number of options
# @param [in] option_ptr  Array of options
# @param [in] option_vals_pptr  Array of option values cast to void*
# @param [out] hip_link_state_ptr  hiprtc link state created upon success
# 
# @return #HIPRTC_SUCCESS, #HIPRTC_ERROR_INVALID_INPUT, #HIPRTC_ERROR_INVALID_OPTION
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkCreate(unsigned int num_options,hiprtcJIT_option * option_ptr,void ** option_vals_pptr,hiprtcLinkState* hip_link_state_ptr) nogil


# 
# @brief Adds a file with bit code to be linked with options
# @ingroup Runtime
# @param [in] hip_link_state  hiprtc link state
# @param [in] input_type  Type of the input data or bitcode
# @param [in] file_path  Path to the input file where bitcode is present
# @param [in] num_options  Size of the options
# @param [in] options_ptr  Array of options applied to this input
# @param [in] option_values  Array of option values cast to void*
# 
# @return #HIPRTC_SUCCESS
# 
# If input values are invalid, it will
# @return #HIPRTC_ERROR_INVALID_INPUT
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddFile(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,const char * file_path,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil


# 
# @brief Completes the linking of the given program.
# @ingroup Runtime
# @param [in] hip_link_state  hiprtc link state
# @param [in] input_type  Type of the input data or bitcode
# @param [in] image  Input data which is null terminated
# @param [in] image_size  Size of the input data
# @param [in] name  Optional name for this input
# @param [in] num_options  Size of the options
# @param [in] options_ptr  Array of options applied to this input
# @param [in] option_values  Array of option values cast to void*
# 
# @return #HIPRTC_SUCCESS, #HIPRTC_ERROR_INVALID_INPUT
# 
# If adding the file fails, it will
# @return #HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkAddData(hiprtcLinkState hip_link_state,hiprtcJITInputType input_type,void * image,unsigned long image_size,const char * name,unsigned int num_options,hiprtcJIT_option * options_ptr,void ** option_values) nogil


# 
# @brief Completes the linking of the given program.
# @ingroup Runtime
# @param [in]  hip_link_state  hiprtc link state
# @param [out]  bin_out  Upon success, points to the output binary
# @param [out]  size_out  Size of the binary is stored (optional)
# 
# @return #HIPRTC_SUCCESS
# 
# If adding the data fails, it will
# @return #HIPRTC_ERROR_LINKING
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkComplete(hiprtcLinkState hip_link_state,void ** bin_out,unsigned long * size_out) nogil


# 
# @brief Deletes the link instance via hiprtc APIs.
# @ingroup Runtime
# @param [in] hip_link_state link state instance
# 
# @return #HIPRTC_SUCCESS
# 
# @see hiprtcResult
cdef hiprtcResult hiprtcLinkDestroy(hiprtcLinkState hip_link_state) nogil
