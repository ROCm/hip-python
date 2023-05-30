# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import os
import enum

import hip.hiprtc
hiprtc = hip.hiprtc # makes hiprtc types and routines accessible without import
                            # allows checks such as `hasattr(cuda.nvrtc,"hiprtc")`

HIP_PYTHON_MOD = hiprtc
globals()["HIP_PYTHON"] = True

def _hip_python_get_bool_environ_var(env_var, default):
    yes_vals = ("true", "1", "t", "y", "yes")
    no_vals = ("false", "0", "f", "n", "no")
    value = os.environ.get(env_var, default).lower()
    if value in yes_vals:
        return True
    elif value in no_vals:
        return False
    else:
        allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
        raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")


HIP_PYTHON_nvrtcResult_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_nvrtcResult_HALLUCINATE","false")

class _nvrtcResult_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_nvrtcResult_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_nvrtcResult_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hiprtc.hiprtcResult):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return nvrtcResult
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class nvrtcResult(hiprtc._hiprtcResult__Base,metaclass=_nvrtcResult_EnumMeta):
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

HIP_PYTHON_CUjitInputType_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjitInputType_HALLUCINATE","false")

class _CUjitInputType_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjitInputType_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjitInputType_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hiprtc.hiprtcJITInputType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUjitInputType
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUjitInputType(hiprtc._hiprtcJITInputType__Base,metaclass=_CUjitInputType_EnumMeta):
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

HIP_PYTHON_CUjitInputType_enum_HALLUCINATE = _hip_python_get_bool_environ_var("HIP_PYTHON_CUjitInputType_enum_HALLUCINATE","false")

class _CUjitInputType_enum_EnumMeta(enum.EnumMeta):

    def __getattribute__(cls,name):
        global _get_hip_name
        global HIP_PYTHON_CUjitInputType_enum_HALLUCINATE
        try:
            result = super().__getattribute__(name)
            return result
        except AttributeError as ae:
            if not HIP_PYTHON_CUjitInputType_enum_HALLUCINATE:
                raise ae
            else:
                used_vals = list(cls._value2member_map_.keys())
                if not len(used_vals):
                    raise ae
                new_val = min(used_vals)
                while new_val in used_vals: # find a free enum value
                    new_val += 1

                class HallucinatedEnumConstant():
                    """Mimicks the orginal enum type this is derived from.
                    """
                    def __init__(self):
                        pass

                    @property
                    def name(self):
                        return self._name_

                    @property
                    def value(self):
                        return self._value_

                    def __eq__(self,other):
                        if isinstance(other,hiprtc.hiprtcJITInputType):
                            return self.value == other.value
                        return False

                    def __repr__(self):        
                        """Mimicks enum.Enum.__repr__"""
                        return "<%s.%s: %r>" % (
                                self.__class__._name_, self._name_, self._value_)

                    def __str__(self):
                        """Mimicks enum.Enum.__str__"""
                        return "%s.%s" % (self.__class__._name_, self._name_)

                    def __hash__(self):
                        return hash(str(self))

                    @property
                    def __class__(self):
                        """Make this type appear as a constant of the actual 
                        CUDA enum type in isinstance checks.
                        """
                        return CUjitInputType_enum
                setattr(HallucinatedEnumConstant,"_name_",name)
                setattr(HallucinatedEnumConstant,"_value_",new_val)
                return HallucinatedEnumConstant()


class CUjitInputType_enum(hiprtc._hiprtcJITInputType__Base,metaclass=_CUjitInputType_enum_EnumMeta):
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
cdef class CUlinkState_st(hip.hiprtc.ihiprtcLinkState):
    pass
CUlinkState = hiprtc.hiprtcLinkState
nvrtcGetErrorString = hiprtc.hiprtcGetErrorString
nvrtcVersion = hiprtc.hiprtcVersion
nvrtcProgram = hiprtc.hiprtcProgram
nvrtcAddNameExpression = hiprtc.hiprtcAddNameExpression
nvrtcCompileProgram = hiprtc.hiprtcCompileProgram
nvrtcCreateProgram = hiprtc.hiprtcCreateProgram
nvrtcDestroyProgram = hiprtc.hiprtcDestroyProgram
nvrtcGetLoweredName = hiprtc.hiprtcGetLoweredName
nvrtcGetProgramLog = hiprtc.hiprtcGetProgramLog
nvrtcGetProgramLogSize = hiprtc.hiprtcGetProgramLogSize
nvrtcGetPTX = hiprtc.hiprtcGetCode
nvrtcGetPTXSize = hiprtc.hiprtcGetCodeSize
nvrtcGetCUBIN = hiprtc.hiprtcGetBitcode
nvrtcGetCUBINSize = hiprtc.hiprtcGetBitcodeSize
cuLinkCreate = hiprtc.hiprtcLinkCreate
cuLinkCreate_v2 = hiprtc.hiprtcLinkCreate
cuLinkAddFile = hiprtc.hiprtcLinkAddFile
cuLinkAddFile_v2 = hiprtc.hiprtcLinkAddFile
cuLinkAddData = hiprtc.hiprtcLinkAddData
cuLinkAddData_v2 = hiprtc.hiprtcLinkAddData
cuLinkComplete = hiprtc.hiprtcLinkComplete
cuLinkDestroy = hiprtc.hiprtcLinkDestroy