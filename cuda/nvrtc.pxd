# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

cimport hip.chiprtc
cimport hip.hiprtc

cimport cuda.cnvrtc
cimport cuda.nvrtc
from hip.hiprtc cimport ihiprtcLinkState # here
cdef class CUlinkState_st(hip.hiprtc.ihiprtcLinkState):
    pass