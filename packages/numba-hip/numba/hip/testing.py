import os
import platform
# import shutil

from numba.tests.support import SerialMixin
# from numba.hip.rocmpaths import get_rocm_path
from numba.hip.hipdrv import driver # , devices, libs
from numba.core import config
from numba.tests.support import TestCase
from pathlib import Path
import unittest

numba_hip_dir = Path(__file__).parent
test_data_dir = numba_hip_dir / 'tests' / 'data'


class HIPTestCase(SerialMixin, TestCase):
    """
    For tests that use a HIP device. Test methods in a HIPTestCase must not
    be run out of module order, because the ContextResettingTestCase may reset
    the context and destroy resources used by a normal HIPTestCase if any of
    its tests are run between tests from a HIPTestCase.
    """

    def setUp(self):
        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY

        # Disable warnings about low gpu utilization in the test suite
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        # Disable warnings about host arrays in the test suite
        config.CUDA_WARN_ON_IMPLICIT_COPY = 0

    def tearDown(self):
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy

# CUDA interoperability
CUDATestCase = HIPTestCase


class ContextResettingTestCase(HIPTestCase):
    """
    For tests where the context needs to be reset after each test. Typically
    these inspect or modify parts of the context that would usually be expected
    to be internal implementation details (such as the state of allocations and
    deallocations, etc.).
    """

    def tearDown(self):
        super().tearDown()
        from numba.hip.hipdrv.devices import reset
        reset()

def _hipify_reason(reason: str):
    return reason.replace("CUDA","HIP")

def skip_on_hipsim(reason):
    """Skip this test if running on the HIP simulator"""
    return unittest.skipIf(config.ENABLE_CUDASIM, _hipify_reason(reason))

# CUDA interoperability
skip_on_cudasim = skip_on_hipsim

def skip_unless_hipsim(reason):
    """Skip this test if running on HIP hardware"""
    return unittest.skipUnless(config.ENABLE_CUDASIM, _hipify_reason(reason))

# HIP not supported
# def skip_unless_conda_hiptoolkit(reason):
#     """Skip test if the HIP toolkit was not installed by Conda"""
#     return unittest.skipUnless(get_conda_ctk() is not None, reason)


def skip_if_external_memmgr(reason):
    """Skip test if an EMM Plugin is in use"""
    return unittest.skipIf(config.CUDA_MEMORY_MANAGER != 'default', _hipify_reason(reason))


def skip_under_hip_memcheck(reason):
    return unittest.skipIf(os.environ.get('CUDA_MEMCHECK') is not None, _hipify_reason(reason))

# TODO HIP not supported
# def skip_without_nvdisasm(reason):
#     nvdisasm_path = shutil.which('nvdisasm')
#     return unittest.skipIf(nvdisasm_path is None, reason)


# TODO HIP not supported
# def skip_with_nvdisasm(reason):
#     nvdisasm_path = shutil.which('nvdisasm')
#     return unittest.skipIf(nvdisasm_path is not None, reason)


def skip_on_arm(reason):
    cpu = platform.processor()
    is_arm = cpu.startswith('arm') or cpu.startswith('aarch')
    return unittest.skipIf(is_arm, reason)


def skip_if_hip_includes_missing(fn):
    # Skip when hip/hip_runtime.h is not available - generally this should indicate
    # whether the HIP includes are available or not
    hip_runtime_h = os.path.join(config.CUDA_INCLUDE_PATH, 'hip/hip_runtime.h')
    hip_runtime_h_file = (os.path.exists(hip_runtime_h) and os.path.isfile(hip_runtime_h))
    reason = 'HIP include dir not available on this system'
    return unittest.skipUnless(hip_runtime_h_file, reason)(fn)

skip_if_cuda_includes_missing = skip_if_hip_includes_missing

# TODO HIP not supported
# def skip_if_mvc_enabled(reason):
#     """Skip a test if Minor Version Compatibility is enabled"""
#     return unittest.skipIf(config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY,
#                            reason)

# TODO HIP not supported
# def skip_if_mvc_libraries_unavailable(fn):
#     libs_available = False
#     try:
#         import cubinlinker  # noqa: F401
#         import ptxcompiler  # noqa: F401
#         libs_available = True
#     except ImportError:
#         pass

#     return unittest.skipUnless(libs_available,
#                                "Requires cubinlinker and ptxcompiler")(fn)

# TODO HIP not supported
# def cc_X_or_above(major, minor):
#     if not config.ENABLE_CUDASIM:
#         cc = devices.get_context().device.compute_capability
#         return cc >= (major, minor)
#     else:
#         return True


# def skip_unless_cc_50(fn):
#     return unittest.skipUnless(cc_X_or_above(5, 0), "requires cc >= 5.0")(fn)


# def skip_unless_cc_53(fn):
#     return unittest.skipUnless(cc_X_or_above(5, 3), "requires cc >= 5.3")(fn)


# def skip_unless_cc_60(fn):
#     return unittest.skipUnless(cc_X_or_above(6, 0), "requires cc >= 6.0")(fn)


# def skip_unless_cc_75(fn):
#     return unittest.skipUnless(cc_X_or_above(7, 5), "requires cc >= 7.5")(fn)

# TODO HIP not supported
# def xfail_unless_hipsim(fn):
#     if config.ENABLE_CUDASIM:
#         return fn
#     else:
#         return unittest.expectedFailure(fn)

def skip_with_cuda_python(reason):
    return unittest.skipIf(driver.USE_NV_BINDING, reason)

# TODO HIP not supported
# def hipdevrt_missing():
#     if config.ENABLE_CUDASIM:
#         return False
#     try:
#         libs.check_static_lib('hipdevrt')
#     except FileNotFoundError:
#         return True
#     return False


# def skip_if_hipdevrt_missing(fn):
#     return unittest.skipIf(hipdevrt_missing(), 'hipdevrt missing')(fn)


class ForeignArray(object):
    """
    Class for emulating an array coming from another library through the CUDA
    Array interface. This just hides a DeviceNDArray so that it doesn't look
    like a DeviceNDArray.
    """

    def __init__(self, arr):
        self._arr = arr
        self.__cuda_array_interface__ = arr.__cuda_array_interface__
