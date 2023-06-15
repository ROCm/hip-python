# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

from ._version import *
HIP_VERSION = 50422804
HIP_VERSION_NAME = hip_version_name = "5.4.22804-474e8620"
HIP_VERSION_TUPLE = hip_version_tuple = (5,4,22804,"474e8620")


from . import _util
from . import hip
from . import hiprtc
from . import hipblas
from . import rccl
from . import hiprand
from . import hipfft
from . import hipsparse