# AMD_COPYRIGHT

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

from _codegen import PackageGenerator, Node, MacroDefinition, FieldDecl, FunctionDecl

__author__ = "AMD_AUTHOR"

ROCM_INC = "/opt/rocm/include"

def generate_files(pkg_gen: PackageGenerator):
    cython_c_preamble = """\
# AMD_COPYRIGHT
from libc.stdint import *

"""
    python_cython_preamble_template = """\
# AMD_COPYRIGHT
from . cimport c{pkg}

"""
    with open(f"hip/c{pkg_gen.pkg_name}.pxd", "w") as outfile:
        outfile.write(cython_c_preamble)
        outfile.write(pkg_gen.render_cython_bindings())

    with open(f"hip/{pkg_gen.pkg_name}.pyx", "w") as outfile:
        outfile.write(python_cython_preamble_template.format(pkg=pkg_gen.pkg_name))
        outfile.write(pkg_gen.render_python_interfaces(f"c{pkg_gen.pkg_name}"))

# hiprtc
def hiprtc_node_filter(node: Node):
    if isinstance(node,FieldDecl):
        return False
    if isinstance(node,MacroDefinition):
        return node.name.startswith("hiprtc")
    if node.file.endswith("hiprtc.h"):
        return True
    return False

pkg_gen = PackageGenerator(
    "hiprtc", ROCM_INC, ["hip/hiprtc.h"], "libhiprtc.so", hiprtc_node_filter
)

generate_files(pkg_gen)

# hip
hip_int_macros = (
    #  from hip/hip_version.h
    "HIP_VERSION_MAJOR",
    "HIP_VERSION_MINOR",
    "HIP_VERSION_PATCH",
    "HIP_VERSION_GITHASH",
    "HIP_VERSION_BUILD_ID",
    "HIP_VERSION_BUILD_NAME",
    "HIP_VERSION",
    # from hip/hip_texture_types.h
    "hipTextureType1D", 
    "hipTextureType2D",
    "hipTextureType3D",
    "hipTextureTypeCubemap",
    "hipTextureType1DLayered",
    "hipTextureType2DLayered",
    "hipTextureTypeCubemapLayered",
    "HIP_IMAGE_OBJECT_SIZE_DWORD",
    "HIP_SAMPLER_OBJECT_SIZE_DWORD",
    "HIP_SAMPLER_OBJECT_OFFSET_DWORD",
    "HIP_TEXTURE_OBJECT_SIZE_DWORD",
    # from hip/driver_types.h
    "HIP_TRSA_OVERRIDE_FORMAT", 
    "HIP_TRSF_READ_AS_INTEGER",
    "HIP_TRSF_NORMALIZED_COORDINATES",
    "HIP_TRSF_SRGB",
    # from hip/hip_runtime_api.h
    "hipIpcMemLazyEnablePeerAccess",
    "HIP_IPC_HANDLE_SIZE",
    "hipStreamDefault",
    "hipStreamNonBlocking",
    "hipEventDefault",
    "hipEventBlockingSync",
    "hipEventDisableTiming",
    "hipEventInterprocess",
    "hipEventReleaseToDevice",
    "hipEventReleaseToSystem",
    "hipHostMallocDefault",
    "hipHostMallocPortable",
    "hipHostMallocMapped",
    "hipHostMallocWriteCombined",
    "hipHostMallocNumaUser",
    "hipHostMallocCoherent",
    "hipHostMallocNonCoherent",
    "hipMemAttachGlobal",
    "hipMemAttachHost",
    "hipMemAttachSingle",
    "hipDeviceMallocDefault",
    "hipDeviceMallocFinegrained",
    "hipMallocSignalMemory",
    "hipHostRegisterDefault",
    "hipHostRegisterPortable",
    "hipHostRegisterMapped",
    "hipHostRegisterIoMemory",
    "hipExtHostRegisterCoarseGrained",
    "hipDeviceScheduleAuto",
    "hipDeviceScheduleSpin",
    "hipDeviceScheduleYield",
    "hipDeviceScheduleBlockingSync",
    "hipDeviceScheduleMask",
    "hipDeviceMapHost",
    "hipDeviceLmemResizeToMax",
    "hipArrayDefault",
    "hipArrayLayered",
    "hipArraySurfaceLoadStore",
    "hipArrayCubemap",
    "hipArrayTextureGather",
    "hipOccupancyDefault",
    "hipCooperativeLaunchMultiDeviceNoPreSync",
    "hipCooperativeLaunchMultiDeviceNoPostSync",
    "hipCpuDeviceId",
    "hipInvalidDeviceId",
    "hipExtAnyOrderLaunch",
    "hipStreamWaitValueGte",
    "hipStreamWaitValueEq",
    "hipStreamWaitValueAnd",
    "hipStreamWaitValueNor",
    "hipStreamPerThread",
    #"USE_PEER_NON_UNIFIED",
)

def hip_node_filter(node: Node):
    if isinstance(node,FieldDecl):
        return False
    if isinstance(node,FunctionDecl):
        if not node.name.startswith("hip"):
            return False
    if node.name in hip_int_macros:
        return True
    if not isinstance(node,MacroDefinition):
        if "hip/" in node.file:
            return True
    return False

pkg_gen = PackageGenerator(
    "hip",
    ROCM_INC,
    ["hip/hip_runtime_api.h","hip/hip_ext.h"],
    "libhipamd64.so",
    hip_node_filter,
)

generate_files(pkg_gen)
