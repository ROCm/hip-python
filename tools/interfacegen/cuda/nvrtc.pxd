# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport hip.chiprtc
cimport hip.hiprtc

DEF HIP_PYTHON = True

cimport cuda.cnvrtc
from hip.hiprtc cimport ihiprtcLinkState # here
cdef class CUlinkState_st(hip.hiprtc.ihiprtcLinkState):
    pass