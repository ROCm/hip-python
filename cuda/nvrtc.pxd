# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import hip.hiprtc
cimport hip.chiprtc
import os
import enum
cimport hip.hiprtc

cimport cuda.cnvrtc
from hip.chiprtc cimport ihiprtcLinkState as CUlinkState_st
from hip.chiprtc cimport hiprtcGetErrorString as nvrtcGetErrorString
from hip.chiprtc cimport hiprtcVersion as nvrtcVersion
from hip.chiprtc cimport hiprtcAddNameExpression as nvrtcAddNameExpression
from hip.chiprtc cimport hiprtcCompileProgram as nvrtcCompileProgram
from hip.chiprtc cimport hiprtcCreateProgram as nvrtcCreateProgram
from hip.chiprtc cimport hiprtcDestroyProgram as nvrtcDestroyProgram
from hip.chiprtc cimport hiprtcGetLoweredName as nvrtcGetLoweredName
from hip.chiprtc cimport hiprtcGetProgramLog as nvrtcGetProgramLog
from hip.chiprtc cimport hiprtcGetProgramLogSize as nvrtcGetProgramLogSize
from hip.chiprtc cimport hiprtcGetCode as nvrtcGetPTX
from hip.chiprtc cimport hiprtcGetCodeSize as nvrtcGetPTXSize
from hip.chiprtc cimport hiprtcGetBitcode as nvrtcGetCUBIN
from hip.chiprtc cimport hiprtcGetBitcodeSize as nvrtcGetCUBINSize
from hip.chiprtc cimport hiprtcLinkCreate as cuLinkCreate
from hip.chiprtc cimport hiprtcLinkCreate as cuLinkCreate_v2
from hip.chiprtc cimport hiprtcLinkAddFile as cuLinkAddFile
from hip.chiprtc cimport hiprtcLinkAddFile as cuLinkAddFile_v2
from hip.chiprtc cimport hiprtcLinkAddData as cuLinkAddData
from hip.chiprtc cimport hiprtcLinkAddData as cuLinkAddData_v2
from hip.chiprtc cimport hiprtcLinkComplete as cuLinkComplete
from hip.chiprtc cimport hiprtcLinkDestroy as cuLinkDestroy