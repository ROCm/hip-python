# AMD_COPYRIGHT

cimport hip.chiprtc

cdef enum nvrtcResult:
    NVRTC_SUCCESS = hip.chiprtc.HIPRTC_SUCCESS
    HIPRTC_SUCCESS = hip.chiprtc.HIPRTC_SUCCESS
    NVRTC_ERROR_OUT_OF_MEMORY = hip.chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    HIPRTC_ERROR_OUT_OF_MEMORY = hip.chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    NVRTC_ERROR_INVALID_INPUT = hip.chiprtc.HIPRTC_ERROR_INVALID_INPUT
    HIPRTC_ERROR_INVALID_INPUT = hip.chiprtc.HIPRTC_ERROR_INVALID_INPUT
    NVRTC_ERROR_INVALID_PROGRAM = hip.chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    HIPRTC_ERROR_INVALID_PROGRAM = hip.chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    NVRTC_ERROR_INVALID_OPTION = hip.chiprtc.HIPRTC_ERROR_INVALID_OPTION
    HIPRTC_ERROR_INVALID_OPTION = hip.chiprtc.HIPRTC_ERROR_INVALID_OPTION
    NVRTC_ERROR_COMPILATION = hip.chiprtc.HIPRTC_ERROR_COMPILATION
    HIPRTC_ERROR_COMPILATION = hip.chiprtc.HIPRTC_ERROR_COMPILATION
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hip.chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hip.chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    NVRTC_ERROR_INTERNAL_ERROR = hip.chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
    HIPRTC_ERROR_INTERNAL_ERROR = hip.chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
    HIPRTC_ERROR_LINKING = hip.chiprtc.HIPRTC_ERROR_LINKING
cdef enum CUjitInputType:
    CU_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    HIPRTC_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    CU_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    HIPRTC_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    CU_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    HIPRTC_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    CU_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    HIPRTC_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    CU_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    HIPRTC_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    CU_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    HIPRTC_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    CU_JIT_NUM_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    HIPRTC_JIT_INPUT_LLVM_BITCODE = hip.chiprtc.HIPRTC_JIT_INPUT_LLVM_BITCODE
    HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = hip.chiprtc.HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE
    HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = hip.chiprtc.HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE
    HIPRTC_JIT_NUM_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_INPUT_TYPES
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