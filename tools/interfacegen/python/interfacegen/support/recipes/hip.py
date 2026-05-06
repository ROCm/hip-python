# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import re

import pyparsing as pyp

from interfacegen.cparser import TypeHandler
from interfacegen.support.recipes.control import ParmIntent
from interfacegen.tree import (
    Field,
    Function,
    MacroDefinition,
    Node,
    Parm,
    Record,
)

TypeCategory = TypeHandler.TypeCategory

# HIP


class hip:

    str_macros = (
        "HIP_VERSION_GITHASH",
        "HIP_VERSION_BUILD_NAME",
    )
    # NOTE: Uses void* macro values from ROCm 7.1.1
    # NOTE: Newer versions of GCC do not allow
    #       to simply convert ``void*`` to ``unsigned long long``.
    #       This requires us to hardcode these values into the generated #       interfaces for certain languages like Cython.
    void_p_macros = dict(
        HIP_LAUNCH_PARAM_BUFFER_POINTER=0x01,
        HIP_LAUNCH_PARAM_BUFFER_SIZE=0x02,
        HIP_LAUNCH_PARAM_END=0x03,
    )
    int_macros = (
        #  from hip/hip_version.h
        "HIP_VERSION_MAJOR",
        "HIP_VERSION_MINOR",
        "HIP_VERSION_PATCH",
        # "HIP_VERSION_GITHASH", # no int, is char *
        "HIP_VERSION_BUILD_ID",
        # "HIP_VERSION_BUILD_NAME", # is char *
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
                # some modifications: # TODO move this into node_init
                if isinstance(node, Record) and node.name == "dim3":
                    node.set_defaults(x=1, y=1, z=1)
                return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        if node.name in hip.int_macros:
            return "int"
        if node.name in hip.void_p_macros:
            return "void *"
        if node.name in hip.str_macros:
            return "char *"
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
        Exceptions are ``char *`` parameters, which
        are C-style strings.

        2. All ``void``, ``struct``, ``union``, ``enum`` double (``**``) pointers are
        return values that are created internally by the respective function.
        """
        func_name, parm_idx = parm.parent.name, parm.parm_index
        if (func_name, parm_idx) in (
            ("hipDeviceGetName", 0),
            ("hipIpcGetMemHandle", 0),
            ("hipMemGetAddressRange", 0),
            ("hipDeviceGetUuid", 0),
            ("hipDeviceGetPCIBusId", 0),
            ("hipDrvGetErrorName", 1),
            ("hipDrvGetErrorString", 1),
        ):
            return ParmIntent.OUT
        if (func_name, parm_idx) in (
            ("hipPointerGetAttribute", 0),
            ("hipExtStreamGetCUMask", 2),
        ):
            return ParmIntent.INOUT
        if (func_name, parm_idx) in (("hipExtStreamCreateWithCUMask", 2),):
            return ParmIntent.IN

        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr", "ptr", "dev_ptr", "data", "dptr"]:
                return ParmIntent.OUT
        if parm.is_pointer_to_enum(degree=1):
            return ParmIntent.OUT
        if parm.is_pointer_to_record(degree=2) or (
            parm.is_pointer_to_basic_type(degree=1)
            and not parm.is_pointer_to_char(degree=1)
        ):
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            func_name, parm_idx = node.parent.name, node.parm_index
            if (func_name, parm_idx) in (
                ("hipDrvPointerGetAttributes", 1),
                ("hipMemRangeGetAttributes", 2),
                ("hipMemPoolGetAccess", 0),
                ("hipModuleLoadDataEx", 3),
                ("hipExtStreamCreateWithCUMask", 2),
                ("hipExtStreamGetCUMask", 2),
            ):
                return 1
            if (
                (
                    node.is_pointer_to_basic_type(degree=1)
                    and not node.is_pointer_to_char(degree=1)
                )
                or node.is_pointer_to_enum(degree=1)
                or node.is_pointer_to_record(degree=1)
                or node.is_pointer_to_record(degree=2)
            ):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans HIP doxygen documentation strings.

        * Removes '@}', '@{', and dash sequences of more than three dashes.
        * Removes other strings associated with groups.
        """
        result = re.sub(r"@{|@}|----+", "", raw_comment)
        result = result.replace(
            "This section describes the event management functions of HIP runtime API.",
            "",
        )
        return result

    @staticmethod
    def renamer(name: str):
        """Handle macros that rename functions and types.

        Handles the following ROCm 6.0.0 macros:

        ```
        #define hipGetDeviceProperties hipGetDevicePropertiesR0600
        #define hipDeviceProp_t hipDeviceProp_tR0600
        #define hipChooseDevice hipChooseDeviceR0600
        ```
        """
        return name.replace("R0600", "")


# HIPRTC


class hiprtc:

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return False # NOTE: node.file is None for macros
        elif node.file is None:
            print(f"node.file is None: {node.cursor.kind}")
            return False
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
                    [
                        TypeCategory.ARRAY,
                        TypeCategory.POINTER,
                        TypeCategory.BASIC,
                    ],
                    [
                        TypeCategory.ARRAY,
                        TypeCategory.POINTER,
                        TypeCategory.VOID,
                    ],
                ):
                    return 2
                return 1
            elif (
                len(node.name) == 1
                and node.name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ):
                return 2
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans hipBLAS doxygen documentation strings.

        Removes the ******************************************************************
        """
        return raw_comment.replace(
            "******************************************************************",
            "",
        )


class hipsolver:

    @staticmethod
    def node_filter(node: Node):
        if node.name in ("__int16_t", "__uint16_t"):
            return True
        if not isinstance(node, MacroDefinition):
            if node.name[0:9] in ("hipsolver,HIPSOLVER"):
                # if "Batched" in node.name:
                #    return False
                return True
        elif node.name in (
            "hipsolverVersionMajor",
            "hipsolverVersionMinor",
            "hipsolverVersionMinor",
            "hipsolverVersionPatch",
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
        return hipblas.ptr_rank(node)

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        return raw_comment


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
            return "char *"
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
        if node.is_pointer_to_constantarray_of_basic_type(degree=2):
            return ParmIntent.OUT
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if node.is_pointer_to_basic_type(degree=1):
            if node.name == "output_data":
                return ParmIntent.INOUT
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
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
        func_name = node.parent.name
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if func_name == "hipsparseCreate":
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

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans hipSPARSE doxygen documentation strings.

        Removes the doxygen @{ group start parts from the comments.
        """
        parts = []
        for tokens, _, __ in pyp.cppStyleComment.scanString(raw_comment):
            stripped = (
                tokens[0].replace(" ", "").replace("\n", "").replace("!", "")
            )
            if stripped != "/**@{*/":
                parts.append(tokens[0])
                if "@}" in stripped or "@{" in stripped:
                    print(tokens[0])

        return "\n".join(parts)


# ROCTX


class roctx:

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name in (
                "ROCTX_VERSION_MAJOR",
                "ROCTX_VERSION_MINOR",
            )
        elif node.name.startswith("roctx"):
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
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        In roctx, all pointers are `const char *`, i.e. char sequences.
        """
        return 1

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans roctx doxygen documentation strings."""
        return raw_comment


class hipfile:
    """Code generation controls for hipFILE (Accelerated I/O Storage).

    The hipFILE C API uses the `hipFile` prefix for all functions/types and
    `HIPFILE_` for macros. Identifiers outside those prefixes are external
    types pulled in via `#include` (hipError_t, off_t, sockaddr, ...) and
    must not be re-emitted by the bindings.
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name in (
                "HIPFILE_VERSION_MAJOR",
                "HIPFILE_VERSION_MINOR",
                "HIPFILE_VERSION_PATCH",
                "HIPFILE_BASE_ERR",
            )
        if node.name.startswith("hipFile") or node.name.startswith("HIPFILE_"):
            return True
        return False

    @staticmethod
    def macro_type(node: MacroDefinition):
        return "int"

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Classify pointer parameter intent for hipFILE APIs.

        Default to IN; specific OUT parameters are recognized by parameter
        name conventions (handles like 'fhPtr', 'driver_ptr', sizes/offsets
        that are typically input).
        """
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """All pointers in hipFILE are single-level (handles, buffers, paths)."""
        return 1

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans hipFILE doxygen documentation strings."""
        return raw_comment


class comgr:
    """Code generation controls for AMD COMGR (`amd_comgr/amd_comgr.h`).

    Ported from the original `recipes/hip_python/comgr/generate_comgr.py`
    when comgr was merged into the per-library hip recipe.
    The `ptr_complicated_type_handler` here is a logic stub that needs
    the project's util_types prefix; `pkg_compiler.generate_amd_comgr()`
    wraps it with that prefix and falls back to `default_ptr_handler`
    for everything else.
    """

    @staticmethod
    def node_filter(node: Node):
        return (
            # use global_name because of anonymous funptrs
            node.global_name("_").startswith("amd_comgr")
            or node.name.startswith("AMD_COMGR_INTERFACE_VERSION")
            or node.name == "code_object_info_s"
        )

    @staticmethod
    def node_init(node: Node):
        if isinstance(node, Function):
            if not node.is_enum and node.name.startswith("amd_comgr"):
                # amd_comgr routines without status return — force them
                # to not throw exceptions and to always return
                # AMD_COMGR_STATUS_SUCCESS as the first return value.
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
                node.prepend_python_return_value(
                    "amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS",
                    "amd_comgr_status_s",
                    "Always returns `~.amd_comgr_status_s.AMD_COMGR_STATUS_SUCCESS`.",
                )

    @staticmethod
    def ptr_rank(node):
        # `Typed.is_pointer_to_char(degree=-1)` — only char pointers are
        # treated as char sequences (rank 1). Everything else stays scalar.
        if hasattr(node, "is_pointer_to_char") and node.is_pointer_to_char(degree=-1):
            return 1
        return 0

    @staticmethod
    def ptr_parm_intent(parm: Parm):
        func_name, parm_index = parm.parent.name, parm.parm_index
        if (func_name, parm_index) in (
            ("amd_comgr_action_info_set_option_list", 1),
        ):
            return ParmIntent.IN
        if func_name in (
            "amd_comgr_get_isa_count",
            "amd_comgr_get_version",
            "amd_comgr_create_data_set",
            "amd_comgr_create_action_info",
        ):
            return ParmIntent.OUT
        if (func_name, parm_index) in (
            ("comgr.amd_comgr_create_data", 1),
            ("amd_comgr_status_string", 1),
            ("amd_comgr_action_info_get_option_list_count", 1),
            ("amd_comgr_create_data", 1),
            ("amd_comgr_get_data_kind", 1),
            ("amd_comgr_action_data_count", 2),
            ("amd_comgr_action_data_get_data", 3),
            ("amd_comgr_get_isa_name", 1),
            ("amd_comgr_get_isa_metadata", 1),
            ("amd_comgr_get_data_metadata", 1),
            ("amd_comgr_get_metadata_kind", 1),
            ("amd_comgr_get_metadata_map_size", 1),
            ("amd_comgr_metadata_lookup", 2),
            ("amd_comgr_get_metadata_list_size", 1),
            ("amd_comgr_index_list_metadata", 2),
            ("amd_comgr_create_symbolizer_info", 2),
            ("amd_comgr_create_disassembly_info", 4),
        ):
            return ParmIntent.OUT
        # INOUT (uncovered):
        # amd_comgr_get_data, amd_comgr_get_data_name, amd_comgr_get_isa_name,
        # amd_comgr_get_metadata_string, amd_comgr_iterate_map_metadata
        return ParmIntent.INOUT

    @staticmethod
    def is_listofbytes_pointer(node: Node) -> bool:
        """Whether a pointer node should be exposed as ListOfBytes.

        Used by `pkg_compiler.generate_amd_comgr()` to wrap the special-case
        `(amd_comgr_action_info_set_option_list, 1)` mapping with the
        project's util_types prefix.
        """
        if isinstance(node, Parm):
            return (node.parent.name, node.parm_index) == (
                "amd_comgr_action_info_set_option_list", 1,
            )
        return False

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans amd_comgr doxygen documentation strings."""
        return raw_comment


class llvm_c:
    """Code generation controls shared across all `llvm-c/*.h` modules.

    Ported from the closures in `recipes/hip_python/llvm/generate_llvm.py`
    (`create_generator()` body, lines 65-135) when the LLVM recipe was
    merged into the per-library hip recipe.

    `is_listofpointer_param` and `is_ndbuffer_return` are static
    predicates; `pkg_compiler.write_llvm_modules()` wraps them with the
    project's util_types prefix to build the actual
    `ptr_complicated_type_handler` closure (mirrors the comgr pattern).
    """

    @staticmethod
    def ptr_rank(node: Node):
        """Mirrors generate_llvm.py:65-76 — ParamTypes/Params/Dest are
        arrays of pointers (rank 1); char* sequences are rank 1; everything
        else is rank 0 (scalar)."""
        if (node.parent.cursor.spelling, node.cursor.spelling) in (
            ("LLVMFunctionType", "ParamTypes"),
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
        ):
            return 1
        if node.is_pointer_to_char(degree=-1):
            return 1
        return 0

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Mirrors generate_llvm.py:78-115 — INOUT for the array-out
        params, OUT for `Out*`-prefixed names + a per-(fn, parm) lookup,
        IN otherwise. Order matters."""
        fn_name: str = node.parent.cursor.spelling
        parm_name: str = node.cursor.spelling
        if (fn_name, parm_name) in (
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
            ("LLVMTargetMachineEmitToMemoryBuffer", "OutMemBuf"),
            ("LLVMDisasmInstruction", "OutString"),
        ):
            return ParmIntent.INOUT
        if (
            fn_name in ("LLVMGetVersion",)
            or parm_name in (
                "OutEE",
                "OutError",
                "OutFn",
                "OutInterp",
                "OutJIT",
                "OutM",
                "OutMemBuf",
                "OutMessage",
                "OutMod",
                "OutModule",
            )
            or (fn_name, parm_name) in (
                ("LLVMGetValueName2", "Length"),
                ("LLVMGetTargetFromTriple", "T"),
                ("LLVMGetTargetFromTriple", "ErrorMessage"),
            )
        ):
            return ParmIntent.OUT
        return ParmIntent.IN

    @staticmethod
    def is_listofpointer_param(node: Node) -> bool:
        """Whether a parameter pointer should be exposed as ListOfPointer.
        Used by `pkg_compiler.write_llvm_modules()` to wrap the special-case
        mapping with the project's util_types prefix."""
        return (
            node.parent.cursor.spelling,
            node.cursor.spelling,
        ) in (
            ("LLVMFunctionType", "ParamTypes"),
            ("LLVMGetParams", "Params"),
            ("LLVMGetParamTypes", "Dest"),
            ("LLVMRunFunction", "Args"),
            ("LLVMGetBufferStart"),
        )

    @staticmethod
    def is_ndbuffer_return(node: Node) -> bool:
        """Whether a function return pointer should be exposed as NDBuffer.
        Currently only `LLVMGetBufferStart`."""
        return node.cursor.spelling in ("LLVMGetBufferStart",)

    @staticmethod
    def location_filter(header_relpath: str):
        """Returns a node_filter closure that keeps declarations whose
        `render_location()` contains `header_relpath`. Mirrors
        `create_node_filter()` at generate_llvm.py:222-232."""
        def _filter(node: Node):
            if isinstance(node, MacroDefinition):
                return False
            return header_relpath in node.render_location()
        return _filter


class llvm_config:
    """Code generation controls for `llvm/Config/llvm-config.h`.

    Ported from the closures at generate_llvm.py:171-220 — selective
    macro extraction with type classification.
    """

    # Target-/platform-specific macros we don't want to bake into bindings.
    _NATIVE_MACROS = (
        "LLVM_NATIVE_ARCH",
        "LLVM_NATIVE_ASMPARSER",
        "LLVM_NATIVE_ASMPRINTER",
        "LLVM_NATIVE_DISASSEMBLER",
        "LLVM_NATIVE_TARGET",
        "LLVM_NATIVE_TARGETINFO",
        "LLVM_NATIVE_TARGETMC",
    )
    _STR_MACROS = (
        "LLVM_DEFAULT_TARGET_TRIPLE",
        "LLVM_HOST_TRIPLE",
        "LLVM_VERSION_STRING",
    )
    _INT_MACROS = (
        "LLVM_VERSION_MAJOR",
        "LLVM_VERSION_MINOR",
        "LLVM_VERSION_PATCH",
    )
    _BOOL_MACROS = (
        "LLVM_ENABLE_THREADS",
        "LLVM_HAS_ATOMICS",
        "LLVM_ON_UNIX",
        "LLVM_USE_INTEL_JITEVENTS",
        "LLVM_USE_OPROFILE",
        "LLVM_USE_PERF",
        "LLVM_FORCE_ENABLE_STATS",
        "LLVM_ENABLE_ZLIB",
        "LLVM_ENABLE_ZSTD",
        "HAVE_SYSEXITS_H",
        "LLVM_UNREACHABLE_OPTIMIZE",
        "LLVM_ENABLE_DIA_SDK",
    )

    @staticmethod
    def macro_type(node: MacroDefinition):
        name = node.cursor.spelling
        if name in llvm_config._NATIVE_MACROS:
            return None  # skip — target/platform specific
        if name in llvm_config._STR_MACROS:
            return "const char *"
        if name in llvm_config._INT_MACROS:
            return "int"
        if name in llvm_config._BOOL_MACROS:
            return "bint"
        if name in ("LLVM_ENABLE_PLUGINS",):  # existence means True
            return True
        return None

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return llvm_config.macro_type(node) is not None
        return False
