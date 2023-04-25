# AMD_COPYRIGHT

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

import sys
import os
import enum
import textwrap

from setuptools import setup, Extension
from Cython.Build import cythonize

from _codegen import (
    CythonPackageGenerator,
    Node,
    MacroDefinition,
    Function,
    Parm,
    Field,
    PointerParamIntent,
)

__author__ = "AMD_AUTHOR"

# Configuration
ROCM_PATH = os.environ.get("ROCM_PATH", None)
if not ROCM_PATH:
    ROCM_PATH = os.environ.get("ROCM_HOME")
if not ROCM_PATH:
    raise RuntimeError("Environment variable ROCM_PATH is not set")
rocm_inc = os.path.join(ROCM_PATH, "include")
CFLAGS = os.environ.get("CFLAGS", None)

HIP_PLATFORM = os.environ.get("HIP_PLATFORM", "amd")
if HIP_PLATFORM not in ("amd", "hcc"):
    raise RuntimeError("Currently only HIP_PLATFORM=amd is supported")


class HipPlatform(enum.IntEnum):
    AMD = 0
    NVIDIA = 1

    @staticmethod
    def from_string(key: str):
        valid_inputs = ("amd", "hcc", "nvidia", "nvcc")
        key = key.lower()
        if key in valid_inputs[0:2]:
            return HipPlatform.AMD
        elif key in valid_inputs[2:4]:
            return HipPlatform.NVIDIA
        else:
            raise ValueError(
                f"Input must be one of: {','.join(valid_inputs)} (any case)"
            )

    @property
    def cflags(self):
        return ["-D", f"__HIP_PLATFORM_{self.name}__"]


hip_platform = HipPlatform.from_string(HIP_PLATFORM)


def get_bool_environ_var(env_var, default):
    return os.environ.get(env_var, default).lower() in ("true", "1", "t", "y", "yes")


HIP_PYTHON_SETUP_GENERATE = get_bool_environ_var("HIP_PYTHON_SETUP_GENERATE", "true")
HIP_PYTHON_SETUP_BUILD = get_bool_environ_var("HIP_PYTHON_SETUP_BUILD", "true")
HIP_PYTHON_SETUP_RUNTIME_LINKING = get_bool_environ_var(
    "HIP_PYTHON_SETUP_RUNTIME_LINKING", "true"
)
HIP_PYTHON_SETUP_VERBOSE = get_bool_environ_var("HIP_PYTHON_SETUP_VERBOSE", "true")

if HIP_PYTHON_SETUP_VERBOSE:
    print("Environment variables:")
    print(f"{ROCM_PATH=}")
    print(f"{HIP_PLATFORM=}")
    print(f"{HIP_PYTHON_SETUP_GENERATE=}")
    print(f"{HIP_PYTHON_SETUP_BUILD=}")
    print(f"{HIP_PYTHON_SETUP_RUNTIME_LINKING=}")
    print(f"{HIP_PYTHON_SETUP_VERBOSE=}")

if HIP_PYTHON_SETUP_GENERATE:
    # hiprtc
    def hiprtc_node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("hiprtc")
        if node.file is None:
            print(f"node.file is None: {node.cursor.kind}")
        if node.file.endswith("hiprtc.h"):
            return True
        return False

    CythonPackageGenerator(
        "hiprtc",
        rocm_inc,
        "hip/hiprtc.h",
        runtime_linking=HIP_PYTHON_SETUP_RUNTIME_LINKING,
        dll="libhiprtc.so",
        node_filter=hiprtc_node_filter,
        cflags=hip_platform.cflags,
    ).write_package_files(output_dir="hip")

    # hip
    hip_str_macros = (
        "HIP_VERSION_GITHASH",
        "HIP_VERSION_BUILD_NAME",
    )

    hip_int_macros = (
        #  from hip/hip_version.h
        "HIP_VERSION_MAJOR",
        "HIP_VERSION_MINOR",
        "HIP_VERSION_PATCH",
        # "HIP_VERSION_GITHASH", # no int, is char*
        "HIP_VERSION_BUILD_ID",
        # "HIP_VERSION_BUILD_NAME", # is char*
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
        # "hipStreamPerThread", # no int, type is struct ihipStream_t *
        # "USE_PEER_NON_UNIFIED",
    )

    def hip_node_filter(node: Node):
        if isinstance(node, Function):
            if not node.name.startswith("hip"):
                return False
        if node.name in hip_int_macros:
            return True
        if node.name in hip_str_macros:
            return True
        if not isinstance(node, MacroDefinition):
            if "hip/" in node.file:
                return True
        return False

    def hip_macro_type(node: MacroDefinition):
        if node.name in hip_int_macros:
            return "int"
        if node.name in hip_str_macros:
            return "char*"
        assert False, "Not implemented!"

    def hip_ptr_parm_intent(parm: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.

        Rules
        -----

        1. We exploit that ``hip/hip_runtime_api.h``` does not
        work with typed arrays, so every pointer
        of basic type is actually a return value
        that is created internally by the function.
        Exceptions are ``char*`` parameters, which
        are C-style strings.

        2. All ``void``, ``struct``, ``union``, ``enum`` double (``**``) pointers are
        return values that are created internally by the respective function.
        """
        if (
            parm.is_pointer_to_record(degree=2)
            or parm.is_pointer_to_enum(degree=1)
            or (
                parm.is_pointer_to_basic_type(degree=1)
                and not parm.is_pointer_to_char(degree=1)
            )
        ):
            return PointerParamIntent.OUT
        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr","ptr","dev_ptr","data"]:
                return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hip_ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if (
                node.is_pointer_to_basic_type(degree=1)
                or node.is_pointer_to_enum(degree=1)
                or node.is_pointer_to_record(degree=1)
                or node.is_pointer_to_record(degree=2)
            ):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    generator = CythonPackageGenerator(
        "hip",
        rocm_inc,
        "hip/hip_runtime_api.h",
        runtime_linking=HIP_PYTHON_SETUP_RUNTIME_LINKING,
        dll="libamdhip64.so",
        node_filter=hip_node_filter,
        ptr_parm_intent=hip_ptr_parm_intent,
        ptr_rank=hip_ptr_rank,
        macro_type=hip_macro_type,
        cflags=hip_platform.cflags,
    )
    generator.write_package_files
    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""
    for node in generator.backend.root.walk():
        if isinstance(node,MacroDefinition):
            last_token = list(node.cursor.get_tokens())[-1].spelling
            if node.name == "HIP_VERSION_MAJOR":
                HIP_VERSION_MAJOR = int(last_token)
            elif node.name == "HIP_VERSION_MINOR":
                HIP_VERSION_MINOR = int(last_token)
            elif node.name == "HIP_VERSION_PATCH":
                HIP_VERSION_PATCH = int(last_token)
            elif node.name == "HIP_VERSION_GITHASH":
                HIP_VERSION_GITHASH = last_token.strip('"')
    HIP_VERSION_NAME = f"{HIP_VERSION_MAJOR}.{HIP_VERSION_MINOR}.{HIP_VERSION_PATCH}-{HIP_VERSION_GITHASH}"
    HIP_VERSION= (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH)

    with open("hip/__init__.py","w") as f:
        f.write(textwrap.dedent(f"""\
            from ._version import *
            HIP_VERSION = {HIP_VERSION}
            HIP_VERSION_NAME = hip_version_name = "{HIP_VERSION_NAME}"
            HIP_VERSION_TUPLE = hip_version_tuple = ({HIP_VERSION_MAJOR},{HIP_VERSION_MINOR},{HIP_VERSION_PATCH},"{HIP_VERSION_GITHASH}")

            from . import _util
            from . import hip
            from . import hiprtc
            from . import hipblas""")
        )


    # hipblas
    def hipblas_node_filter(node: Node):
        if node.name in ("__int16_t","__uint16_t"):
            return True
        if not isinstance(node, MacroDefinition):
            if node.name[0:7] in ("hipblas,HIPBLAS"):
                if "Batched" in node.name:
                    return False
                return True
        elif node.name in (
            "hipblasVersionMajor",
            "hipblaseVersionMinor",
            "hipblasVersionMinor",
            "hipblasVersionPatch",
            # "hipblasVersionTweak", # double?
        ):
            return True
        return False

    def hipblas_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hipblas_ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.
        
        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.name == "handle":
                return 0
            elif node.name in (
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                "iota",
                "kappa",
                "lambda",
                "mu",
                "nu",
                "xi",
                "omicron",
                "pi",
                "rho",
                "sigma",
                "tau",
                "upsilon",
                "phi",
                "chi",
                "psi",
                "omega",
            ):
                return 0
            elif len(node.name) == 1 and node.name.lower() in "abcdefghijklmnopqrstuvwxyz":
                return 1
            elif len(node.name) == 1 and node.name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return 2
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    generator = CythonPackageGenerator(
        "hipblas",
        rocm_inc,
        "hipblas/hipblas.h",
        runtime_linking=HIP_PYTHON_SETUP_RUNTIME_LINKING,
        dll="libhipblas.so",
        node_filter=hipblas_node_filter,
        ptr_parm_intent=hipblas_ptr_parm_intent,
        ptr_rank=hipblas_ptr_rank,
        cflags=hip_platform.cflags,
    )
    generator.c_interface_preamble += """\
from .chip cimport hipStream_t
"""
    generator.python_interface_preamble += """\
#ctypedef int16_t __int16_t
#ctypedef uint16_t __uint16_t
from .hip cimport ihipStream_t
"""
    generator.write_package_files(output_dir="hip")

# Build Cython packages
if HIP_PYTHON_SETUP_BUILD:
    if HIP_PYTHON_SETUP_RUNTIME_LINKING:
        libraries = []
        library_dirs = []
    else:
        library_dirs = [os.path.join(ROCM_PATH, "lib")]
        libraries = ["hiprtc", "amdhip64"]

    extra_compile_args = hip_platform.cflags
    if CFLAGS == None:
        extra_compile_args += ["-O3"]

    def create_extension(name, sources):
        return Extension(
            name,
            sources=sources,
            include_dirs=[rocm_inc],
            library_dirs=library_dirs,
            libraries=libraries,
            language="c",
            extra_compile_args=extra_compile_args,
        )

    cython_module_sources = [
        ("hip._util.datahandle", ["./hip/_util/datahandle.pyx"]),
        ("hip.chiprtc", ["./hip/chiprtc.pyx"]),
        ("hip.chip", ["./hip/chip.pyx"]),
        ("hip.chipblas", ["./hip/chipblas.pyx"]),
        ("hip.hiprtc", ["./hip/hiprtc.pyx"]),
        ("hip.hip", ["./hip/hip.pyx"]),
        ("hip.hipblas", ["./hip/hipblas.pyx"]),
    ]
    if HIP_PYTHON_SETUP_RUNTIME_LINKING:
        cython_module_sources.insert(
            0, ("hip._util.posixloader", ["./hip/_util/posixloader.pyx"])
        )

    ext_modules = []
    for name, sources in cython_module_sources:
        extension = create_extension(name, sources)
        ext_modules += cythonize(
            [extension],
            compiler_directives=dict(
                embedsignature=True,
                language_level=3,
            ),
        )

    setup(
        ext_modules=ext_modules,
        use_scm_version = {
          "write_to": "hip/_version.py",
          "local_scheme": lambda v: f"+{HIP_VERSION_NAME.replace('-','.')}"
        }
    )