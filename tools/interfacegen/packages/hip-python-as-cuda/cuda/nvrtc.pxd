# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport hip.chiprtc
cimport hip.hiprtc

cimport cuda.cnvrtc
from hip.hiprtc cimport ihiprtcLinkState
cdef class CUlinkState_st(hip.hiprtc.ihiprtcLinkState):
    pass