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

"""
Attributes:
    HIP_PYTHON (`.bool`):
        `True`.
    hip_python_mod (module):
        A reference to the package `.hip.hiprtc`.
    hiprtc (module):
        A reference to the package `.hip.hiprtc`.
    HIP_PYTHON_nvrtcResult_HALLUCINATE:
        Make `.nvrtcResult` hallucinate values for non-existing enum constants. Disabled by default
        if default is not modified via environment variable.

        Default value can be set/unset via environment variable ``HIP_PYTHON_nvrtcResult_HALLUCINATE``.

        * Environment variable values that result in `True` are: ``yes``, ``1``, ``y``, ``true`` 
        * Those that result in `False` are: ``no``, ``0``, ``n``, ``false``.
    HIP_PYTHON_CUjitInputType_HALLUCINATE:
        Make `.CUjitInputType` hallucinate values for non-existing enum constants. Disabled by default
        if default is not modified via environment variable.

        Default value can be set/unset via environment variable ``HIP_PYTHON_CUjitInputType_HALLUCINATE``.

        * Environment variable values that result in `True` are: ``yes``, ``1``, ``y``, ``true`` 
        * Those that result in `False` are: ``no``, ``0``, ``n``, ``false``.
    HIP_PYTHON_CUjitInputType_enum_HALLUCINATE:
        Make `.CUjitInputType_enum` hallucinate values for non-existing enum constants. Disabled by default
        if default is not modified via environment variable.

        Default value can be set/unset via environment variable ``HIP_PYTHON_CUjitInputType_enum_HALLUCINATE``.

        * Environment variable values that result in `True` are: ``yes``, ``1``, ``y``, ``true`` 
        * Those that result in `False` are: ``no``, ``0``, ``n``, ``false``.
    CUlinkState:
        alias of `.hiprtcLinkState`
    nvrtcGetErrorString:
        alias of `.hiprtcGetErrorString`
    nvrtcVersion:
        alias of `.hiprtcVersion`
    nvrtcProgram:
        alias of `.hiprtcProgram`
    nvrtcAddNameExpression:
        alias of `.hiprtcAddNameExpression`
    nvrtcCompileProgram:
        alias of `.hiprtcCompileProgram`
    nvrtcCreateProgram:
        alias of `.hiprtcCreateProgram`
    nvrtcDestroyProgram:
        alias of `.hiprtcDestroyProgram`
    nvrtcGetLoweredName:
        alias of `.hiprtcGetLoweredName`
    nvrtcGetProgramLog:
        alias of `.hiprtcGetProgramLog`
    nvrtcGetProgramLogSize:
        alias of `.hiprtcGetProgramLogSize`
    nvrtcGetPTX:
        alias of `.hiprtcGetCode`
    nvrtcGetPTXSize:
        alias of `.hiprtcGetCodeSize`
    nvrtcGetCUBIN:
        alias of `.hiprtcGetBitcode`
    nvrtcGetCUBINSize:
        alias of `.hiprtcGetBitcodeSize`
    cuLinkCreate:
        alias of `.hiprtcLinkCreate`
    cuLinkCreate_v2:
        alias of `.hiprtcLinkCreate`
    cuLinkAddFile:
        alias of `.hiprtcLinkAddFile`
    cuLinkAddFile_v2:
        alias of `.hiprtcLinkAddFile`
    cuLinkAddData:
        alias of `.hiprtcLinkAddData`
    cuLinkAddData_v2:
        alias of `.hiprtcLinkAddData`
    cuLinkComplete:
        alias of `.hiprtcLinkComplete`
    cuLinkDestroy:
        alias of `.hiprtcLinkDestroy`

"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import os
import enum

import hip.hiprtc
hiprtc = hip.hiprtc # makes hiprtc types and routines accessible without import
                            # allows checks such as `hasattr(cuda.nvrtc,"hiprtc")`

hip_python_mod = hiprtc
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
    HIPRTC_SUCCESS = hip.chiprtc.HIPRTC_SUCCESS
    NVRTC_SUCCESS = hip.chiprtc.HIPRTC_SUCCESS
    HIPRTC_ERROR_OUT_OF_MEMORY = hip.chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    NVRTC_ERROR_OUT_OF_MEMORY = hip.chiprtc.HIPRTC_ERROR_OUT_OF_MEMORY
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
    HIPRTC_ERROR_INVALID_INPUT = hip.chiprtc.HIPRTC_ERROR_INVALID_INPUT
    NVRTC_ERROR_INVALID_INPUT = hip.chiprtc.HIPRTC_ERROR_INVALID_INPUT
    HIPRTC_ERROR_INVALID_PROGRAM = hip.chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    NVRTC_ERROR_INVALID_PROGRAM = hip.chiprtc.HIPRTC_ERROR_INVALID_PROGRAM
    HIPRTC_ERROR_INVALID_OPTION = hip.chiprtc.HIPRTC_ERROR_INVALID_OPTION
    NVRTC_ERROR_INVALID_OPTION = hip.chiprtc.HIPRTC_ERROR_INVALID_OPTION
    HIPRTC_ERROR_COMPILATION = hip.chiprtc.HIPRTC_ERROR_COMPILATION
    NVRTC_ERROR_COMPILATION = hip.chiprtc.HIPRTC_ERROR_COMPILATION
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = hip.chiprtc.HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = hip.chiprtc.HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hip.chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = hip.chiprtc.HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
    HIPRTC_ERROR_INTERNAL_ERROR = hip.chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
    NVRTC_ERROR_INTERNAL_ERROR = hip.chiprtc.HIPRTC_ERROR_INTERNAL_ERROR
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
    HIPRTC_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    CU_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    HIPRTC_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    CU_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    HIPRTC_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    CU_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    HIPRTC_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    CU_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    HIPRTC_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    CU_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    HIPRTC_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    CU_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    CU_JIT_NUM_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
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
    HIPRTC_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    CU_JIT_INPUT_CUBIN = hip.chiprtc.HIPRTC_JIT_INPUT_CUBIN
    HIPRTC_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    CU_JIT_INPUT_PTX = hip.chiprtc.HIPRTC_JIT_INPUT_PTX
    HIPRTC_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    CU_JIT_INPUT_FATBINARY = hip.chiprtc.HIPRTC_JIT_INPUT_FATBINARY
    HIPRTC_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    CU_JIT_INPUT_OBJECT = hip.chiprtc.HIPRTC_JIT_INPUT_OBJECT
    HIPRTC_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    CU_JIT_INPUT_LIBRARY = hip.chiprtc.HIPRTC_JIT_INPUT_LIBRARY
    HIPRTC_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    CU_JIT_INPUT_NVVM = hip.chiprtc.HIPRTC_JIT_INPUT_NVVM
    HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
    CU_JIT_NUM_INPUT_TYPES = hip.chiprtc.HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
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

__all__ = [
    "HIP_PYTHON",
    "hip_python_mod",
    "hiprtc",
    "_nvrtcResult_EnumMeta",
    "HIP_PYTHON_nvrtcResult_HALLUCINATE",
    "nvrtcResult",
    "_CUjitInputType_EnumMeta",
    "HIP_PYTHON_CUjitInputType_HALLUCINATE",
    "CUjitInputType",
    "_CUjitInputType_enum_EnumMeta",
    "HIP_PYTHON_CUjitInputType_enum_HALLUCINATE",
    "CUjitInputType_enum",
    "CUlinkState_st",
    "CUlinkState",
    "nvrtcGetErrorString",
    "nvrtcVersion",
    "nvrtcProgram",
    "nvrtcAddNameExpression",
    "nvrtcCompileProgram",
    "nvrtcCreateProgram",
    "nvrtcDestroyProgram",
    "nvrtcGetLoweredName",
    "nvrtcGetProgramLog",
    "nvrtcGetProgramLogSize",
    "nvrtcGetPTX",
    "nvrtcGetPTXSize",
    "nvrtcGetCUBIN",
    "nvrtcGetCUBINSize",
    "cuLinkCreate",
    "cuLinkCreate_v2",
    "cuLinkAddFile",
    "cuLinkAddFile_v2",
    "cuLinkAddData",
    "cuLinkAddData_v2",
    "cuLinkComplete",
    "cuLinkDestroy",
]