from _codegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from _codegen.tree import (
    Node,
    MacroDefinition,
    Function,
    Parm,
    Field,
    Record,
)

from _codegen.control import ParmIntent

# HIP

class hip:

    str_macros = (
        "HIP_VERSION_GITHASH",
        "HIP_VERSION_BUILD_NAME",
    )
    void_p_macros = (
        "HIP_LAUNCH_PARAM_BUFFER_POINTER",
        "HIP_LAUNCH_PARAM_BUFFER_SIZE",
        "HIP_LAUNCH_PARAM_END",
    )
    int_macros = (
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

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, Function):
            if not node.name.startswith("hip"):
                return False
        if node.name in hip.int_macros:
            return True
        if node.name in hip.str_macros:
            return True
        if node.name in hip.void_p_macros:
            return True
        if not isinstance(node, MacroDefinition):
            if "hip/" in node.file:
                # some modifications:
                if isinstance(node, Record) and node.name == "dim3":
                    node.set_defaults(x=1, y=1, z=1)
                return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        if node.name in hip.int_macros:
            return "int"
        if node.name in hip.void_p_macros:
            return "unsigned long int"
        if node.name in hip.str_macros:
            return "char*"
        assert False, "Not implemented!"

    @staticmethod
    def ptr_parm_intent(parm: Parm):
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
            if (parm.parent.name, parm.name) == ("hipDeviceGetAttribute", "pi"):
                return ParmIntent.INOUT
            return ParmIntent.OUT
        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr", "ptr", "dev_ptr", "data", "dptr"]:
                return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
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

# HIPRTC

class hiprtc:

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("hiprtc")
        if node.file is None:
            print(f"node.file is None: {node.cursor.kind}")
        if node.file.endswith("hiprtc.h"):
            return True
        return False

    @staticmethod
    def ptr_parm_intent(parm: Parm):
        """ """
        out_parms = (
            ("hiprtcVersion", "major"),
            ("hiprtcVersion", "minor"),
            ("hiprtcCreateProgram", "prog"),
            ("hiprtcGetLoweredName", "lowered_name"),  # rank == 1
            ("hiprtcGetProgramLogSize", "logSizeRet"),
            ("hiprtcGetCodeSize", "codeSizeRet"),
            ("hiprtcGetBitcodeSize", "bitcode_size"),
            ("hiprtcLinkCreate", "hip_link_state_ptr"),
            ("hiprtcLinkComplete", "bin_out"),  # rank == 1
            ("hiprtcLinkComplete", "size_out"),
        )
        inout_parms = (  # these buffers must be allocated by user
            ("hiprtcGetCode", "code"),
            ("hiprtcGetProgramLog", "log"),
            ("hiprtcGetBitcode", "bitcode"),
        )
        if (parm.parent.name, parm.name) in out_parms:
            return ParmIntent.OUT
        if (parm.parent.name, parm.name) in inout_parms:
            return ParmIntent.INOUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if (
                node.is_pointer_to_basic_type(degree=1)
                and not node.is_pointer_to_char(degree=1)
                or node.is_pointer_to_record(degree=1)
                or node.is_pointer_to_record(degree=2)
            ):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

# HIPBLAS

class hipblas:

    @staticmethod
    def node_filter(node: Node):
        if node.name in ("__int16_t", "__uint16_t"):
            return True
        if not isinstance(node, MacroDefinition):
            if node.name[0:7] in ("hipblas,HIPBLAS"):
                # if "Batched" in node.name:
                #    return False
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

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
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
            elif (
                len(node.name) == 1
                and node.name.lower() in "abcdefghijklmnopqrstuvwxyz"
            ):
                categories = list(node.categorized_type_layer_kinds())
                if categories in (
                    [TypeCategory.ARRAY, TypeCategory.POINTER, TypeCategory.BASIC],
                    [TypeCategory.ARRAY, TypeCategory.POINTER, TypeCategory.VOID],
                ):
                    return 2
                return 1
            elif len(node.name) == 1 and node.name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return 2
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

# RCCL

class rccl:

    @staticmethod
    def node_filter(node: Node):
        if not isinstance(node, MacroDefinition):
            if node.name.startswith("nccl") or node.name.startswith("pnccl"):
                return True
        elif node.name in (
            "NCCL_MAJOR",
            "NCCL_MINOR",
            "NCCL_PATCH",
            "NCCL_SUFFIX",
            "NCCL_VERSION_CODE",
            "RCCL_BFLOAT16",
            "RCCL_GATHER_SCATTER",
            "RCCL_ALLTOALLV",
            "RCCL_MULTIRANKPERGPU",
            "NCCL_UNIQUE_ID_BYTES",
        ):
            return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        rccl_str_macros = "NCCL_SUFFIX"
        if node.name in rccl_str_macros:
            return "char*"
        return "int"

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "comm"),
                ("pncclCommInitAll", "comm"),
            ):
                return ParmIntent.INOUT
            return ParmIntent.OUT
        if node.is_pointer_to_record(degree=1):
            if (node.parent.name, node.name) == "ncclGetUniqueId":
                return ParmIntent.OUT
        if node.is_pointer_to_basic_type(degree=1):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "devlist"),
                ("pncclCommInitAll", "devlist"),
            ):
                return ParmIntent.IN
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=1):
                return 0
            if node.is_pointer_to_record(degree=2):
                if (node.parent.name, node.name) in (
                    ("ncclCommInitAll", "comm"),
                    ("pncclCommInitAll", "comm"),
                ):
                    return 1
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                if (node.parent.name, node.name) in (
                    ("ncclCommInitAll", "devlist"),
                    ("pncclCommInitAll", "devlist"),
                ):
                    return 1
                return 0
        return 1

# HIPRAND
class hiprand:

    @staticmethod
    def node_filter(node: Node):
        if not isinstance(node, MacroDefinition):
            if node.name.startswith("hiprand") or node.name == "uint4":
                return True
            if node.name.startswith("rocrand"):
                if not isinstance(node, Function):
                    return True
        elif node.name in (
            "HIPRAND_VERSION",
            "HIPRAND_DEFAULT_MAX_BLOCK_SIZE",
            "HIPRAND_DEFAULT_MIN_WARPS_PER_EU",
        ):
            return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        return "int"

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if node.is_pointer_to_basic_type(degree=1):
            if node.name == "output_data":
                return ParmIntent.INOUT
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=1):
                return 0
            if node.is_pointer_to_record(degree=2):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

# HIPFFT

class hipfft:

    @staticmethod
    def node_filter(node: Node):
        if not isinstance(node, MacroDefinition):
            if node.name.startswith("hipfft"):
                return True
        elif node.name in (
            "HIPFFT_FORWARD",
            "HIPFFT_BACKWARD",
        ):
            return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        return "int"

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if node.name == "workSize":
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=(1, 2)):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        return 1

# HIPSPARSE

class hipsparse:

    @staticmethod
    def node_filter(node: Node):
        if not isinstance(node, MacroDefinition):
            if (
                node.name.startswith("hipsparse")
                or node.name.endswith("Info_t")
                or (
                    node.name.endswith("Info")
                    and not isinstance(node, Function)
                    and not node.name == "hipArrayMapInfo"
                )
            ):
                return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        return "int"

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=(1, 2)):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1