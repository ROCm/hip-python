# MIT License
# 
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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


cimport rocm.bindings.cyhiprtc as cyhiprtc

from rocm.bindings.cyhiprtc cimport hiprtcResult as nvrtcResult
from rocm.bindings.cyhiprtc cimport HIPRTC_SUCCESS
from rocm.bindings.cyhiprtc cimport HIPRTC_SUCCESS as NVRTC_SUCCESS
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_OUT_OF_MEMORY
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_OUT_OF_MEMORY as NVRTC_ERROR_OUT_OF_MEMORY
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_PROGRAM_CREATION_FAILURE as NVRTC_ERROR_PROGRAM_CREATION_FAILURE
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_INPUT
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_INPUT as NVRTC_ERROR_INVALID_INPUT
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_PROGRAM
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_PROGRAM as NVRTC_ERROR_INVALID_PROGRAM
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_OPTION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INVALID_OPTION as NVRTC_ERROR_INVALID_OPTION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_COMPILATION as NVRTC_ERROR_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE as NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION as NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION as NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID as NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INTERNAL_ERROR
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_INTERNAL_ERROR as NVRTC_ERROR_INTERNAL_ERROR
from rocm.bindings.cyhiprtc cimport HIPRTC_ERROR_LINKING
from rocm.bindings.cyhiprtc cimport ihiprtcLinkState as CUlinkState_st
from rocm.bindings.cyhiprtc cimport hiprtcLinkState as CUlinkState
from rocm.bindings.cyhiprtc cimport hiprtcGetErrorString as nvrtcGetErrorString
from rocm.bindings.cyhiprtc cimport hiprtcVersion as nvrtcVersion
from rocm.bindings.cyhiprtc cimport _hiprtcProgram as _nvrtcProgram
from rocm.bindings.cyhiprtc cimport hiprtcProgram as nvrtcProgram
from rocm.bindings.cyhiprtc cimport hiprtcAddNameExpression as nvrtcAddNameExpression
from rocm.bindings.cyhiprtc cimport hiprtcCompileProgram as nvrtcCompileProgram
from rocm.bindings.cyhiprtc cimport hiprtcCreateProgram as nvrtcCreateProgram
from rocm.bindings.cyhiprtc cimport hiprtcDestroyProgram as nvrtcDestroyProgram
from rocm.bindings.cyhiprtc cimport hiprtcGetLoweredName as nvrtcGetLoweredName
from rocm.bindings.cyhiprtc cimport hiprtcGetProgramLog as nvrtcGetProgramLog
from rocm.bindings.cyhiprtc cimport hiprtcGetProgramLogSize as nvrtcGetProgramLogSize
from rocm.bindings.cyhiprtc cimport hiprtcGetCode as nvrtcGetPTX
from rocm.bindings.cyhiprtc cimport hiprtcGetCodeSize as nvrtcGetPTXSize
from rocm.bindings.cyhiprtc cimport hiprtcGetBitcode as nvrtcGetCUBIN
from rocm.bindings.cyhiprtc cimport hiprtcGetBitcodeSize as nvrtcGetCUBINSize
from rocm.bindings.cyhiprtc cimport hiprtcLinkCreate as cuLinkCreate
from rocm.bindings.cyhiprtc cimport hiprtcLinkCreate as cuLinkCreate_v2
from rocm.bindings.cyhiprtc cimport hiprtcLinkAddFile as cuLinkAddFile
from rocm.bindings.cyhiprtc cimport hiprtcLinkAddFile as cuLinkAddFile_v2
from rocm.bindings.cyhiprtc cimport hiprtcLinkAddData as cuLinkAddData
from rocm.bindings.cyhiprtc cimport hiprtcLinkAddData as cuLinkAddData_v2
from rocm.bindings.cyhiprtc cimport hiprtcLinkComplete as cuLinkComplete
from rocm.bindings.cyhiprtc cimport hiprtcLinkDestroy as cuLinkDestroy