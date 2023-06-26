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


cimport hip.chiprtc

from hip.chiprtc cimport hiprtcResult as nvrtcResult
from hip.chiprtc cimport HIPRTC_SUCCESS
from hip.chiprtc cimport HIPRTC_SUCCESS as NVRTC_SUCCESS
from hip.chiprtc cimport HIPRTC_ERROR_OUT_OF_MEMORY
from hip.chiprtc cimport HIPRTC_ERROR_OUT_OF_MEMORY as NVRTC_ERROR_OUT_OF_MEMORY
from hip.chiprtc cimport HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
from hip.chiprtc cimport HIPRTC_ERROR_PROGRAM_CREATION_FAILURE as NVRTC_ERROR_PROGRAM_CREATION_FAILURE
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_INPUT
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_INPUT as NVRTC_ERROR_INVALID_INPUT
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_PROGRAM
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_PROGRAM as NVRTC_ERROR_INVALID_PROGRAM
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_OPTION
from hip.chiprtc cimport HIPRTC_ERROR_INVALID_OPTION as NVRTC_ERROR_INVALID_OPTION
from hip.chiprtc cimport HIPRTC_ERROR_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_COMPILATION as NVRTC_ERROR_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
from hip.chiprtc cimport HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE as NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
from hip.chiprtc cimport HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION as NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION as NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
from hip.chiprtc cimport HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
from hip.chiprtc cimport HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID as NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
from hip.chiprtc cimport HIPRTC_ERROR_INTERNAL_ERROR
from hip.chiprtc cimport HIPRTC_ERROR_INTERNAL_ERROR as NVRTC_ERROR_INTERNAL_ERROR
from hip.chiprtc cimport HIPRTC_ERROR_LINKING
from hip.chiprtc cimport hiprtcJITInputType as CUjitInputType
from hip.chiprtc cimport HIPRTC_JIT_INPUT_CUBIN
from hip.chiprtc cimport HIPRTC_JIT_INPUT_CUBIN as CU_JIT_INPUT_CUBIN
from hip.chiprtc cimport HIPRTC_JIT_INPUT_PTX
from hip.chiprtc cimport HIPRTC_JIT_INPUT_PTX as CU_JIT_INPUT_PTX
from hip.chiprtc cimport HIPRTC_JIT_INPUT_FATBINARY
from hip.chiprtc cimport HIPRTC_JIT_INPUT_FATBINARY as CU_JIT_INPUT_FATBINARY
from hip.chiprtc cimport HIPRTC_JIT_INPUT_OBJECT
from hip.chiprtc cimport HIPRTC_JIT_INPUT_OBJECT as CU_JIT_INPUT_OBJECT
from hip.chiprtc cimport HIPRTC_JIT_INPUT_LIBRARY
from hip.chiprtc cimport HIPRTC_JIT_INPUT_LIBRARY as CU_JIT_INPUT_LIBRARY
from hip.chiprtc cimport HIPRTC_JIT_INPUT_NVVM
from hip.chiprtc cimport HIPRTC_JIT_INPUT_NVVM as CU_JIT_INPUT_NVVM
from hip.chiprtc cimport HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
from hip.chiprtc cimport HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES as CU_JIT_NUM_INPUT_TYPES
from hip.chiprtc cimport HIPRTC_JIT_INPUT_LLVM_BITCODE
from hip.chiprtc cimport HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE
from hip.chiprtc cimport HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
from hip.chiprtc cimport HIPRTC_JIT_NUM_INPUT_TYPES
ctypedef CUjitInputType CUjitInputType_enum
from hip.chiprtc cimport ihiprtcLinkState
from hip.chiprtc cimport ihiprtcLinkState as CUlinkState_st
from hip.chiprtc cimport hiprtcLinkState
from hip.chiprtc cimport hiprtcLinkState as CUlinkState
from hip.chiprtc cimport hiprtcGetErrorString
from hip.chiprtc cimport hiprtcGetErrorString as nvrtcGetErrorString
from hip.chiprtc cimport hiprtcVersion
from hip.chiprtc cimport hiprtcVersion as nvrtcVersion
from hip.chiprtc cimport hiprtcProgram
from hip.chiprtc cimport hiprtcProgram as nvrtcProgram
from hip.chiprtc cimport hiprtcAddNameExpression
from hip.chiprtc cimport hiprtcAddNameExpression as nvrtcAddNameExpression
from hip.chiprtc cimport hiprtcCompileProgram
from hip.chiprtc cimport hiprtcCompileProgram as nvrtcCompileProgram
from hip.chiprtc cimport hiprtcCreateProgram
from hip.chiprtc cimport hiprtcCreateProgram as nvrtcCreateProgram
from hip.chiprtc cimport hiprtcDestroyProgram
from hip.chiprtc cimport hiprtcDestroyProgram as nvrtcDestroyProgram
from hip.chiprtc cimport hiprtcGetLoweredName
from hip.chiprtc cimport hiprtcGetLoweredName as nvrtcGetLoweredName
from hip.chiprtc cimport hiprtcGetProgramLog
from hip.chiprtc cimport hiprtcGetProgramLog as nvrtcGetProgramLog
from hip.chiprtc cimport hiprtcGetProgramLogSize
from hip.chiprtc cimport hiprtcGetProgramLogSize as nvrtcGetProgramLogSize
from hip.chiprtc cimport hiprtcGetCode
from hip.chiprtc cimport hiprtcGetCode as nvrtcGetPTX
from hip.chiprtc cimport hiprtcGetCodeSize
from hip.chiprtc cimport hiprtcGetCodeSize as nvrtcGetPTXSize
from hip.chiprtc cimport hiprtcGetBitcode
from hip.chiprtc cimport hiprtcGetBitcode as nvrtcGetCUBIN
from hip.chiprtc cimport hiprtcGetBitcodeSize
from hip.chiprtc cimport hiprtcGetBitcodeSize as nvrtcGetCUBINSize
from hip.chiprtc cimport hiprtcLinkCreate
from hip.chiprtc cimport hiprtcLinkCreate as cuLinkCreate
from hip.chiprtc cimport hiprtcLinkCreate as cuLinkCreate_v2
from hip.chiprtc cimport hiprtcLinkAddFile
from hip.chiprtc cimport hiprtcLinkAddFile as cuLinkAddFile
from hip.chiprtc cimport hiprtcLinkAddFile as cuLinkAddFile_v2
from hip.chiprtc cimport hiprtcLinkAddData
from hip.chiprtc cimport hiprtcLinkAddData as cuLinkAddData
from hip.chiprtc cimport hiprtcLinkAddData as cuLinkAddData_v2
from hip.chiprtc cimport hiprtcLinkComplete
from hip.chiprtc cimport hiprtcLinkComplete as cuLinkComplete
from hip.chiprtc cimport hiprtcLinkDestroy
from hip.chiprtc cimport hiprtcLinkDestroy as cuLinkDestroy