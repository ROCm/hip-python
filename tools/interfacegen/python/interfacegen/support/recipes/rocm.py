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
from interfacegen.support.recipes import generic
from interfacegen.support.recipes.control import (
    DEFAULT_PTR_PARM_INTENT,
    DEFAULT_PTR_RANK,
    ParmIntent,
    fallback,
)
from interfacegen.tree import (
    Field,
    Function,
    MacroDefinition,
    Node,
    Parm,
    Record,
)

TypeCategory = TypeHandler.TypeCategory


# ---------------------------------------------------------------------------
# Per-library @fallback chains (intent + rank), most-specific first.
#
# Composition derived from §(3b) of the design plan
# (`share/design/POINTER_ARGUMENTS.md`). Library-specific overrides
# (`(funcname, parm_idx)` maps and HIP/RCCL/etc. type heuristics) live
# inside each class's `ptr_parm_intent` / `ptr_rank` body and run BEFORE
# any of these — the wrapped function returns ``None`` to defer to the
# chain.
#
# `array_with_length_param` and `status_return_out_pointer` are listed
# but currently no-op (relational rules deferred per §(2)). Their
# placement here is aspirational so the chain order is ready when they
# land.
# ---------------------------------------------------------------------------
_RUNTIME_INTENT_CHAIN = (
    generic.double_indirection_out.ptr_parm_intent,
    generic.opaque_typedef_is_handle.ptr_parm_intent,
    generic.status_return_out_pointer.ptr_parm_intent,
    generic.string_z.ptr_parm_intent,
    generic.conservative.ptr_parm_intent,
    DEFAULT_PTR_PARM_INTENT,
)
_RUNTIME_RANK_CHAIN = (
    generic.double_indirection_out.ptr_rank,
    generic.opaque_typedef_is_handle.ptr_rank,
    generic.string_z.ptr_rank,
    generic.conservative.ptr_rank,
    DEFAULT_PTR_RANK,
)
_NUMERICAL_INTENT_CHAIN = (
    generic.double_indirection_out.ptr_parm_intent,
    generic.opaque_typedef_is_handle.ptr_parm_intent,
    generic.array_with_length_param.ptr_parm_intent,
    generic.status_return_out_pointer.ptr_parm_intent,
    generic.conservative.ptr_parm_intent,
    DEFAULT_PTR_PARM_INTENT,
)
_NUMERICAL_RANK_CHAIN = (
    generic.double_indirection_out.ptr_rank,
    generic.opaque_typedef_is_handle.ptr_rank,
    generic.array_with_length_param.ptr_rank,
    generic.conservative.ptr_rank,
    DEFAULT_PTR_RANK,
)
_INPLACE_NUMERICAL_INTENT_CHAIN = (
    generic.double_indirection_out.ptr_parm_intent,
    generic.opaque_typedef_is_handle.ptr_parm_intent,
    generic.pointer_as_reference.ptr_parm_intent,
    generic.array_with_length_param.ptr_parm_intent,
    generic.status_return_out_pointer.ptr_parm_intent,
    generic.conservative.ptr_parm_intent,
    DEFAULT_PTR_PARM_INTENT,
)
_INPLACE_NUMERICAL_RANK_CHAIN = _NUMERICAL_RANK_CHAIN

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
    @fallback(*_RUNTIME_INTENT_CHAIN)
    def ptr_parm_intent(parm: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.

        Library-specific overrides only — modifier-based deductions and
        the unclassifiable-pointer fallback live in the chain wired by
        ``@fallback`` (see ``_RUNTIME_INTENT_CHAIN``).
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

        # HIP-specific naming: certain void** parms with these names are OUT
        # (handle creation slots).
        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr", "ptr", "dev_ptr", "data", "dptr"]:
                return ParmIntent.OUT
        # HIP runtime convention: pointer-to-enum is always a scalar OUT
        # (generally a status/attribute return through pointer).
        if parm.is_pointer_to_enum(degree=1):
            return ParmIntent.OUT
        # HIP runtime convention: scalar-via-pointer OUT for non-string
        # basic-type pointers. Subsumed by `status_return_out_pointer`
        # once that relational rule lands.
        if parm.is_pointer_to_basic_type(degree=1) and not parm.is_pointer_to_char(
            degree=1
        ):
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    @fallback(*_RUNTIME_RANK_CHAIN)
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
        return None  # defer to chain

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
    @fallback(*_RUNTIME_INTENT_CHAIN)
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
        return None  # defer to chain

    @staticmethod
    @fallback(*_RUNTIME_RANK_CHAIN)
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
        return None  # defer to chain


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
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
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
        return None  # defer to chain

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans hipBLAS doxygen documentation strings.

        Removes the ******************************************************************
        """
        return raw_comment.replace(
            "******************************************************************",
            "",
        )


class hipblaslt:
    """Controls for hipBLASLt — extension of hipBLAS for modern matmul.

    `hipblaslt.h` includes `<hipblas/hipblas.h>`, so the AST contains
    every hipblas symbol too. The strict-prefix `node_filter` keeps
    only `hipblasLt*` / `HIPBLASLT_*` symbols; parent hipblas types
    are referenced via `from rocm.bindings.cyhipblas cimport *` in
    the Cython prolog (mirrors how hipblas excludes hip symbols and
    relies on `from rocm.bindings.cyhip cimport *`).
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPBLASLT_")
        return node.name.startswith("hipblasLt") or node.name.startswith("HIPBLASLT_")

    # Reuse the hipblas LAPACK-style heuristics — same `handle`,
    # `alpha`/`beta`, single-letter matrix-name conventions apply.
    ptr_parm_intent = hipblas.ptr_parm_intent
    ptr_rank = hipblas.ptr_rank
    raw_comment_cleaner = hipblas.raw_comment_cleaner


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
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention. Reuses
        ``hipblas.ptr_rank`` (already wired through the numerical chain).
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
    @fallback(*_INPLACE_NUMERICAL_INTENT_CHAIN)
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
        return None  # defer to chain

    @staticmethod
    @fallback(*_INPLACE_NUMERICAL_RANK_CHAIN)
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
        return None  # defer to chain


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
    @fallback(*_NUMERICAL_INTENT_CHAIN)
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
        return None  # defer to chain

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
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
        return None  # defer to chain


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
    @fallback(*_INPLACE_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if node.name == "workSize":
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    @fallback(*_INPLACE_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=(1, 2)):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        return None  # defer to chain


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
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        func_name = node.parent.name
        if node.is_pointer_to_record(degree=2):
            return ParmIntent.OUT
        if func_name == "hipsparseCreate":
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
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
        return None  # defer to chain

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


class hiptensor:
    """Controls for hipTensor — high-performance tensor primitives.

    `hiptensor.h` is a C-API header (extern "C" guarded). It pulls in
    `hiptensor_utility.hpp` which is C-clean despite the .hpp
    extension. The strict-prefix `node_filter` keeps only `hiptensor*`
    / `HIPTENSOR_*` symbols; HIP types (hipDataType etc.) are
    referenced via `from rocm.bindings.cyhip cimport *`.
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPTENSOR_")
        return node.name.startswith("hiptensor") or node.name.startswith("HIPTENSOR_")

    # Reuse generic numerical heuristics; tensor APIs follow handle/
    # descriptor + opaque-pointer conventions analogous to hipBLASLt.
    ptr_parm_intent = hipblas.ptr_parm_intent
    ptr_rank = hipblas.ptr_rank


class hipdnn:
    """Controls for hipDNN backend — graph-style DNN primitives.

    The header `hipdnn_backend.h` is a clean C ABI: depends only on
    `<hip/hip_runtime_api.h>` and a tree of small per-attribute
    enum/typedef sub-headers. Strict-prefix filter keeps only
    `hipdnn*` / `HIPDNN_*` symbols.
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPDNN_")
        return node.name.startswith("hipdnn") or node.name.startswith("HIPDNN_")

    # Reuse generic numerical heuristics — hipdnn uses opaque
    # descriptor handles + per-attribute getter/setter functions
    # (similar shape to hiptensor).
    ptr_parm_intent = hipblas.ptr_parm_intent
    ptr_rank = hipblas.ptr_rank


class hipsparselt:
    """Controls for hipSPARSELt — extension of hipSPARSE for
    structured-sparsity matmul.

    `hipsparselt.h` includes `<hipsparse/hipsparse.h>`, so the AST
    contains every hipsparse symbol too. The strict-prefix
    `node_filter` keeps only `hipsparseLt*` / `HIPSPARSELT_*`
    symbols; parent types are referenced via
    `from rocm.bindings.cyhipsparse cimport *` in the Cython prolog.
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPSPARSELT_")
        return node.name.startswith("hipsparseLt") or node.name.startswith("HIPSPARSELT_")

    # Reuse the hipsparse heuristics — opaque-handle and pointer-rank
    # conventions are identical for the sparse extension family.
    macro_type = hipsparse.macro_type
    ptr_parm_intent = hipsparse.ptr_parm_intent
    ptr_rank = hipsparse.ptr_rank
    raw_comment_cleaner = hipsparse.raw_comment_cleaner


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


# Naming sets derived from a survey of amdsmi.h's 559 documented `@param`
# tags (60% [in], 14% [out], 25% [in,out]). The verb-prefix and parameter-
# name rules below cover ~90% of the API; per-function carve-outs (the rccl
# pattern at lines 522-535 above) handle the remaining edge cases.
_AMDSMI_INOUT_PARM_NAMES = frozenset({
    # Documented as @param[in,out]: input as buffer-capacity, output as
    # actual count. Always paired with an array out-param sibling.
    "count", "num_pages", "len", "size", "num_afids",
    "processor_count", "sensor_count",
    # Caller-allocated array; callee fills counter values per element.
    # `amdsmi_get_utilization_count(..., utilization_counters[], ...)`.
    "utilization_counters",
    # Documented `[in,out]` everywhere it appears in amdsmi.h.
    "timestamp",
})
_AMDSMI_BUFFER_PARM_NAMES = frozenset({
    # @param[out] documented as "Pointer to string" / "array of"
    "name", "uuid", "bdf", "data",
    "processor_handles", "socket_handles", "sensor_inds", "sensor_types",
    "afids",
})


class amdsmi:
    """Code generation controls for AMD SMI (`amd_smi/amdsmi.h`).

    All public symbols use the `amdsmi_` (functions/types) or `AMDSMI_`
    (macros/enum values) prefix. Functions uniformly return
    `amdsmi_status_t`; the dominant call pattern is
    `amdsmi_status_t f(in_args..., out_param*)`.

    Pointer-intent and rank rules are derived from the header's
    `@param[in/out/in,out]` doxygen tags (559 documented params total).
    Edge cases are expected to need per-`(fname, pname)` carve-outs after
    the first codegen run, mirroring how rccl handles `ncclCommInitAll`.
    """

    # Filter policy: ADMIT every `amdsmi_*` / `AMDSMI_*` symbol declared
    # in amdsmi.h, plus the explicit in-header non-prefix symbols below
    # (`_EXTRA_*`). Symbols that hit a remaining codegen-tool gap go in
    # the `_IGNORE_*` exclusion sets. Both ignorelists are empty today —
    # all 281 amdsmi_* functions and 77 amdsmi_* types compile cleanly.
    #
    # The previous shape was a small `_WHITELIST_*` admit-list that grew
    # with each codegen-tool fix. Inversion to ignorelist landed once:
    #   - `Field.cython_repr` learned to move array suffixes after the
    #     field name (test_codegen_gap_void_typedef_array_field.py),
    #   - `node_filter` started walking to the top-most parent so a
    #     named-nested `struct bdf_` inherits its enclosing
    #     `amdsmi_bdf_t`'s admission
    #     (test_codegen_gap_named_nested_struct_hoist.py),
    #   - `tree.Node.is_cursor_anonymous` and treefactory honored
    #     libclang's `cursor.is_anonymous()` so the synthetic
    #     `"struct (anonymous at /path:N:M)"` spelling no longer leaked
    #     into emitted Cython identifiers.
    #
    # Bit-fields are tolerated by Cython inside `cdef extern from`
    # (the `:N` width is silently dropped). The `T arr[]` OUT param gap
    # is sidestepped by the doxygen-driven intent rule in
    # `ptr_parm_intent` — the only such param in amdsmi.h is documented
    # `@param[in,out]` and routes through the working INOUT branch.

    # Symbols to NOT emit even though they pass the prefix filter.
    # Add (function_name, type_name, macro_name) entries here as new
    # codegen-tool gaps surface in future amdsmi.h revisions.
    _IGNORE_FUNCTIONS = frozenset()
    _IGNORE_TYPES = frozenset()

    # Top-level types that are declared INSIDE amdsmi.h but lack the
    # `amdsmi_` prefix. They're referenced by amdsmi-prefixed functions /
    # records so the binding is incomplete without them. (`<stdint.h>` types
    # — uint{8,16,32,64}_t, int{32,64}_t — don't need to be listed here:
    # the codegen already handles them via `from libc.stdint cimport *`.)
    _EXTRA_TYPES = frozenset({
        "amd_metrics_table_header_t",   # used by amdsmi_gpu_metrics_t and
                                        # amdsmi_get_gpu_metrics_header_info
        "processor_type_t",             # used by amdsmi_get_processor_type
    })

    # Useful non-`AMDSMI_`-prefixed integer macros declared in amdsmi.h.
    # `__AMDSMI_H__` (the include guard) is intentionally excluded.
    _EXTRA_MACROS = frozenset({
        "CENTRIGRADE_TO_MILLI_CENTIGRADE",
        "MAX_NUMBER_OF_AFIDS_PER_RECORD",
    })

    # String-valued AMDSMI_ macros — emitted as `char *` constants rather
    # than int. Identified by the printf-format/version-string nature of
    # their literal value in amdsmi.h.
    _STRING_MACROS = frozenset({
        "AMDSMI_TIME_FORMAT",
        "AMDSMI_DATE_FORMAT",
        "AMDSMI_LIB_VERSION_STRING",
    })
    # Function-like macros (parameters in their definition). Cython emits
    # them as `__Pyx_PyInt_From_int(MACRO)` which fails to compile because
    # `MACRO` is a token-paste expression, not an evaluable int. Filter
    # them out at node_filter time.
    _SKIPPED_MACROS = frozenset({
        "AMDSMI_LIB_VERSION_CREATE_STRING",  # (MAJOR, MINOR, RELEASE)
        "AMDSMI_LIB_VERSION_EXPAND_PARTS",   # (MAJOR_STR, MINOR_STR, …)
        "AMDSMI_EVENT_MASK_FROM_INDEX",      # (i)
    })

    @staticmethod
    def _topmost_name(node):
        """Name of the top-most non-Root ancestor (or the node itself).

        For a top-level Function/Record/Typedef this returns the node's
        own name; for a nested type (e.g. ``struct bdf_`` inside
        ``typedef union { struct bdf_ {...} bdf; } amdsmi_bdf_t``) it walks
        up the ``parent`` chain to the outermost type and returns its name.

        Lets a named-nested struct inherit its enclosing type's whitelist
        verdict, so admitting ``amdsmi_bdf_t`` also admits its inner
        ``bdf_`` struct without listing it explicitly.
        """
        from interfacegen import tree
        curr = node
        while curr.parent is not None and not isinstance(curr.parent, tree.Root):
            curr = curr.parent
        return curr.name or ""

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            if node.name in amdsmi._SKIPPED_MACROS:
                return False
            if node.name in amdsmi._EXTRA_MACROS:
                return True
            return node.name.startswith("AMDSMI_")
        # Use the top-most ancestor's name so named-nested types
        # (e.g. `struct bdf_` inside `amdsmi_bdf_t`) inherit their
        # enclosing type's admission.
        name = amdsmi._topmost_name(node)
        if name in amdsmi._IGNORE_FUNCTIONS or name in amdsmi._IGNORE_TYPES:
            return False
        if name in amdsmi._EXTRA_TYPES:
            return True
        return name.startswith("amdsmi_") or name.startswith("AMDSMI_")

    @staticmethod
    def macro_type(node: MacroDefinition):
        if node.name in amdsmi._STRING_MACROS:
            return "char *"
        # Remaining 65-ish macros are integer (sizes, counts, status codes,
        # bitmasks): AMDSMI_MAX_STRING_LENGTH=256, AMDSMI_GPU_UUID_SIZE=38,
        # AMDSMI_STATUS_SUCCESS=0, AMDSMI_PWR_PROF_PRST_INVALID=0xFFFF...
        # The two `_EXTRA_MACROS` names also fall through to "int".
        return "int"

    # Doxygen `@param[in|out|in,out] <name>` regex. amdsmi.h tags 294 of
    # 302 pointer parms; the recipe's verb-based heuristic below covers the
    # ~8 undocumented stragglers.
    _DOXY_PARAM_TAG_RE = re.compile(
        r"@param\[(in|out|in,\s*out)\]\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    _DOXY_TAG_TO_INTENT = {
        "in": ParmIntent.IN,
        "out": ParmIntent.OUT,
        "in,out": ParmIntent.INOUT,
    }

    @staticmethod
    def _doxygen_intent(parm: Parm):
        """Parse `@param[in|out|in,out] <pname>` from the parent function's
        raw doxygen comment. Returns the documented intent or ``None`` if
        the parm isn't tagged.

        amdsmi.h is unusually well annotated — an audit found that trusting
        the docs here resolves all 88 mismatches that the previous
        verb-based heuristic produced (utilization_counters classified as
        OUT instead of INOUT, socket_handles classified as OUT instead of
        INOUT, and so on).
        """
        raw = parm.parent.raw_comment if parm.parent is not None else None
        if not raw:
            return None
        pname = parm.name or ""
        for tag, name_in_doc in amdsmi._DOXY_PARAM_TAG_RE.findall(raw):
            if name_in_doc == pname:
                return amdsmi._DOXY_TAG_TO_INTENT[tag.replace(" ", "")]
        return None

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Classify pointer parameter intent for amdsmi APIs.

        Priority of rules (most specific first):
          1. Doxygen `@param[in|out|in,out]` tag on the parent function —
             trusted as the source of truth. Covers 294 of 302 pointer
             parms in amdsmi.h.
          2. Opaque handle parameters — names ending in `_handle` are
             IN (typedef'd void* values, NOT pointer-to-output-buffer).
          3. INOUT name set — paired count/len params docs as @param[in,out].
          4. amdsmi_set_*  → IN (~44 functions).
          5. amdsmi_get_*  → OUT (~160 functions).
          6. amdsmi_init / amdsmi_shut_down / amdsmi_status_string → IN.
          7. Default → IN.
        """
        doxy = amdsmi._doxygen_intent(node)
        if doxy is not None:
            return doxy
        fname = node.parent.name
        pname = node.name or ""
        if pname.endswith("_handle"):
            return ParmIntent.IN
        if pname in _AMDSMI_INOUT_PARM_NAMES:
            return ParmIntent.INOUT
        if fname.startswith("amdsmi_set_"):
            return ParmIntent.IN
        if fname.startswith("amdsmi_get_"):
            return ParmIntent.OUT
        if fname in ("amdsmi_init", "amdsmi_shut_down", "amdsmi_status_string"):
            return ParmIntent.IN
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Underlying rank (0=scalar, 1=array) for an amdsmi pointer.

        Rules:
          1. Opaque handles (`amdsmi_*_handle` typedefs of `void*`) are
             scalars even when passed as `handle*` for OUT — detected via
             `is_pointer_to_void` because libclang's canonical type
             traversal sees through the typedef.
          2. Documented array/buffer parameter names → rank 1.
          3. Default → rank 0 (single-struct OUT or single-scalar OUT
             dominates: `info`, `config`, `enabled`, `count`, etc.).
        """
        if not isinstance(node, Parm):
            return 1
        # Handles canonicalize to void* — pointer-to-handle is void**.
        if node.is_pointer_to_void(degree=1) or node.is_pointer_to_void(degree=2):
            return 0
        if (node.name or "") in _AMDSMI_BUFFER_PARM_NAMES:
            return 1
        if node.is_pointer_to_record(degree=1):
            return 0
        if node.is_pointer_to_basic_type(degree=1):
            return 0
        return 1

    @staticmethod
    def raw_comment_cleaner(raw_comment: str):
        """Cleans amdsmi doxygen documentation strings."""
        return raw_comment


class hsa:
    """Controls for HSA — Heterogeneous System Architecture runtime.

    The codegen binds `hsa_ext_amd.h`, which transitively includes
    the core `hsa.h`, `hsa_ext_image.h`, and
    `hsa_ven_amd_pc_sampling.h`. The strict-prefix `node_filter`
    accepts every `hsa_*` / `HSA_*` symbol — that captures all four
    families in one go: core HSA, AMD extensions (`hsa_amd_*`),
    image extensions (`hsa_ext_image_*`), and AMD vendor PC-sampling
    (`hsa_ven_amd_*`).

    Note: hsakmt is intentionally not bound — ROCm only ships
    `libhsakmt.a` (static), which conflicts with hip-python's
    dlopen-based runtime model.
    """

    @staticmethod
    def node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HSA_")
        return node.name.startswith("hsa_") or node.name.startswith("HSA_")

    # HSA's API uses opaque `*_t` handles + numerical-rank pointer
    # conventions that match hipBLAS heuristics — reuse them rather
    # than carrying a parallel set.
    ptr_parm_intent = hipblas.ptr_parm_intent
    ptr_rank = hipblas.ptr_rank


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
    @fallback(*_RUNTIME_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Classify pointer parameter intent for hipFILE APIs.

        No library-specific overrides yet — defers fully to the chain
        (``double_indirection_out`` for handle creation, ``string_z`` for
        path strings, ``conservative`` for pointer-to-const, etc.).
        """
        return None  # defer to chain

    @staticmethod
    @fallback(*_RUNTIME_RANK_CHAIN)
    def ptr_rank(node: Node):
        """All pointers in hipFILE are single-level (handles, buffers, paths)."""
        return None  # defer to chain

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
