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
# Useless-macro filters (Category A: visibility/deprecation attribute
# macros; Category B: header guards). The strict-prefix node_filters used
# by the prefix-admit recipes (hsa, amdsmi, hipdnn_backend, hipblaslt,
# hipsparselt, hiptensor) would otherwise drag these into the public
# Python namespace as nonsense `int` constants.
#
# A. Visibility / linkage / deprecation attribute macros expand to
#    `__attribute__(...)` decorators with no Python-level meaning
#    (HSA_API_EXPORT, AMDSMI_DEPRECATED, HIPDNN_EXPORT, …).
# B. Header guards (`#define <FILE>_H` / `_H_`) carry no value either.
#    The body sanity check (empty or `1`) keeps the false-positive rate
#    near zero.
# ---------------------------------------------------------------------------

_ATTRIBUTE_MACRO_SUFFIXES = (
    "_EXPORT",
    "_NO_EXPORT",
    "_EXPORT_DECORATOR",
    "_API_EXPORT",
    "_API_IMPORT",
    "_API",
    "_CALL",
    "_DEPRECATED",
)

_HEADER_GUARD_SUFFIXES = ("_H", "_H_")


def _macro_body_tokens(node) -> "list[str]":
    """Return the token spellings that make up a macro's body
    (everything after the macro name, excluding the closing newline).
    Returns an empty list if tokens can't be enumerated (e.g. for
    nodes whose cursor was disposed) — the caller treats that as an
    empty body.
    """
    cursor = getattr(node, "cursor", None)
    if cursor is None:
        return []
    try:
        tokens = list(cursor.get_tokens())
    except Exception:
        return []
    # First token is the macro name itself; drop it.
    return [t.spelling for t in tokens[1:]]


def _macro_body_is_empty_or_one(node) -> bool:
    """True iff a macro definition's body is empty (`#define FOO_H`)
    or the literal token `1` (`#define FOO_H 1`). Other bodies are
    treated as carrying a meaningful value and kept.
    """
    tokens = _macro_body_tokens(node)
    if not tokens:
        return True  # `#define FOO_H` (no body)
    return len(tokens) == 1 and tokens[0] == "1"  # `#define FOO_H 1`


def _is_attribute_macro(node) -> bool:
    """True if `node` is a MacroDefinition whose name matches a
    visibility / linkage / deprecation attribute pattern."""
    if not isinstance(node, MacroDefinition):
        return False
    name = node.name or ""
    return any(name.endswith(s) for s in _ATTRIBUTE_MACRO_SUFFIXES)


def _is_header_guard_macro(node) -> bool:
    """True if `node` is a MacroDefinition whose name matches the
    `*_H` / `*_H_` header-guard convention AND whose body is the
    typical empty / `1` shape."""
    if not isinstance(node, MacroDefinition):
        return False
    name = node.name or ""
    if not any(name.endswith(s) for s in _HEADER_GUARD_SUFFIXES):
        return False
    return _macro_body_is_empty_or_one(node)


def _is_useless_macro(node) -> bool:
    """True for either an attribute macro or a header guard."""
    return _is_attribute_macro(node) or _is_header_guard_macro(node)


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
    generic.documented_param_intent.ptr_parm_intent,
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
    generic.documented_param_intent.ptr_parm_intent,
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
    generic.documented_param_intent.ptr_parm_intent,
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

    # ---------------------------------------------------------------------
    # Hardcoded ptr_parm_intent overrides for the HIP runtime API.
    #
    # These run BEFORE `_RUNTIME_INTENT_CHAIN` (and so before the new
    # `generic.documented_param_intent` rule) — wired by the
    # `@fallback(*_RUNTIME_INTENT_CHAIN)` decorator on `ptr_parm_intent`.
    # The hardcoded entries are needed because the upstream HIP doxygen
    # tags are wrong (or under-expressive) for the following families:
    #
    # 1. **`hipMemcpy*` / `hipMemset*` `hipArray_t` destinations** are
    #    caller-allocated output buffers that must stay in the args.
    #    `[out]` denotes output *direction* only, which is correct here —
    #    the issue is the *allocation* axis. The `void*`-alias destinations
    #    (`hipDeviceptr_t`, plain `void*`) are handled generically:
    #    `opaque_typedef_is_handle` defers for `void*`-canonical typedefs,
    #    so they rank as rank-1 buffers and stay caller-allocated with no
    #    override. Only the `*A*` variants whose destination is a
    #    `hipArray_t` (`struct hipArray*`, a genuine rank-0 `record*`
    #    handle) need help: a plain rank-0 OUT would be promoted to a
    #    callee-allocated return by the rank-0 inference and dropped. We
    #    pin those to OUT here and override their rank to 1 in `ptr_rank`
    #    (`_HIPMEMCPY_RECORD_DST_NAMES`, gated on
    #    `is_pointer_to_record(degree=1)`), leaving the generic rank-0
    #    fallback unchanged.
    #
    # 2. **Opaque-handle creators** (`hipStreamCreate`, `hipEventCreate`,
    #    `hipMalloc*`, `hipModuleLoad*`, `hipModuleGet*`, `hipMemPool*`,
    #    `hipGraph*`, `hipImport*`, `hipGetSymbol*`, `hipCtxCreate`,
    #    `hipDevicePrimaryCtxRetain`, `hipExternalMemoryGetMappedBuffer`)
    #    are tagged `@param[in, out]` for what is in fact a callee-
    #    allocated OUT pointer of a scalar opaque struct
    #    (`ParmIntent.OUT_CALLEE_ALLOCATED`) — the caller does not
    #    pre-populate, the function produces a fresh handle and the
    #    prior slot contents are discarded. Trusting the `[in, out]` tag
    #    drags the OUT pointer back into the args instead of into the
    #    return tuple.
    #
    # 3. **Caller-provided INPUT pointers tagged `@param[out]`**
    #    (Family 4 in UPSTREAM_BUGS): `hipHostRegister`'s `void* hostPtr`
    #    is a caller-allocated buffer the function only registers, and
    #    `hipMemcpyToSymbol*`'s `const void* symbol` is a read-only input
    #    (the `const` self-contradicts the `[out]` tag). Both are
    #    semantically IN. Trusting the `[out]` tag drops `hostPtr` from
    #    the args entirely and forces `symbol` to NULL — see the explicit
    #    `(func, parm_idx) -> IN` entries below.
    #
    # An upstream bug report has been filed against ROCm/HIP for the
    # mistagged doxygen intent annotations in `hip_runtime_api.h`.
    # Until those tags are fixed in the public header, the codegen
    # has to special-case these families here.
    # ---------------------------------------------------------------------

    # `hipMemcpy*` / `hipMemset*` destinations that are `hipArray_t`
    # (`struct hipArray*`, a genuine rank-0 record* handle). These are
    # caller-allocated output buffers and must stay in the python args.
    # void*-alias destinations (`hipDeviceptr_t`, plain `void*`) need no
    # entry: `opaque_typedef_is_handle` defers for `void*`-canonical
    # typedefs, so they rank as rank-1 buffers and stay caller-allocated
    # via the generic rule. Only the `record*` destinations are rank-0; the
    # rank-0 callee-allocation inference would otherwise drop them, so we
    # pin them to OUT and override their rank to 1 (see `ptr_parm_intent` /
    # `ptr_rank` and UPSTREAM_BUGS Family 1).
    #
    # The destination parameter is named `dstArray` on the `*A*` driver
    # variants and `dst` on `hipMemcpy2DToArray`/`*Async`; both are listed.
    # The `is_pointer_to_record(degree=1)` gate at the use sites keeps this
    # from matching the like-named `void*` destinations (e.g.
    # `hipMemcpyAtoH`'s `void* dst`) or the `record*` *source* arrays
    # (`srcArray`).
    _HIPMEMCPY_RECORD_DST_NAMES = frozenset((
        "dst",
        "dstArray",
    ))

    # Opaque-handle creators — function name + parm 0 -> OUT.
    # Sourced by enumerating the pre-regen committed `hip.pyi` for
    # functions whose old high-level signature dropped parm 0 entirely
    # (i.e. it was always returned, never an input).
    _HIP_HANDLE_CREATOR_OUT_PARM0 = frozenset((
        # streams + events
        "hipStreamCreate",
        "hipStreamCreateWithFlags",
        "hipStreamCreateWithPriority",
        "hipExtStreamCreateWithCUMask",
        "hipEventCreate",
        "hipEventCreateWithFlags",
        # memory allocators (return ptr/devPtr)
        "hipMalloc",
        "hipExtMallocWithFlags",
        "hipMallocHost",
        "hipMemAllocHost",
        "hipHostMalloc",
        "hipHostAlloc",
        "hipMallocManaged",
        "hipMallocAsync",
        "hipMallocFromPoolAsync",
        "hipMallocArray",
        "hipMalloc3DArray",
        "hipMemAlloc",
        # mempool handle creators / importers
        "hipMemPoolCreate",
        "hipMemPoolImportFromShareableHandle",
        "hipMemPoolImportPointer",
        # module loaders
        "hipModuleLoad",
        "hipModuleLoadData",
        "hipModuleLoadDataEx",
        "hipModuleLoadFatBinary",
        # symbol / function / global lookups
        "hipModuleGetFunction",
        "hipModuleGetGlobal",
        "hipModuleGetTexRef",
        "hipGetSymbolAddress",
        "hipGetSymbolSize",
        # graph creators + node adders (parm 0 is pGraphNode/pGraph OUT)
        "hipGraphCreate",
        "hipGraphClone",
        "hipGraphInstantiate",
        "hipGraphInstantiateWithFlags",
        "hipGraphInstantiateWithParams",
        "hipGraphAddNode",
        "hipGraphAddKernelNode",
        "hipGraphAddMemcpyNode",
        "hipGraphAddMemcpyNode1D",
        "hipGraphAddMemcpyNodeFromSymbol",
        "hipGraphAddMemcpyNodeToSymbol",
        "hipGraphAddMemsetNode",
        "hipGraphAddHostNode",
        "hipGraphAddChildGraphNode",
        "hipGraphAddEmptyNode",
        "hipGraphAddEventRecordNode",
        "hipGraphAddEventWaitNode",
        "hipGraphAddMemAllocNode",
        "hipGraphAddMemFreeNode",
        "hipGraphAddBatchMemOpNode",
        "hipGraphAddExternalSemaphoresWaitNode",
        "hipGraphAddExternalSemaphoresSignalNode",
        # external resource importers
        "hipImportExternalMemory",
        "hipImportExternalSemaphore",
        "hipExternalMemoryGetMappedBuffer",
        # contexts
        "hipCtxCreate",
        "hipDevicePrimaryCtxRetain",
        # texture / surface objects
        "hipCreateTextureObject",
        "hipCreateSurfaceObject",
        "hipUserObjectCreate",
    ))
    # Pitch-style allocators write parm 0 (dev ptr) AND parm 1 (pitch).
    _HIP_HANDLE_CREATOR_OUT_PARM01 = frozenset((
        "hipMallocPitch",
        "hipMemAllocPitch",
    ))

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

        # `hipMemcpy*` / `hipMemset*` destinations that are `hipArray_t`
        # (a genuine rank-0 `record*` handle) are caller-allocated output
        # buffers and must stay in the args. Pin them to OUT — the honest
        # direction for a destination — and rely on the rank-1 override in
        # `ptr_rank` to keep the unchanged rank-0 callee-allocation
        # inference from dropping them into the return tuple. The
        # like-named `void*` destinations (`hipDeviceptr_t`, plain `void*`)
        # are rank-1 buffers via the generic rule and need no override; the
        # `is_pointer_to_record(degree=1)` gate excludes them (and the
        # `record*` source arrays). See UPSTREAM_BUGS Family 1.
        if (func_name.startswith("hipMemcpy") or func_name.startswith("hipMemset")) \
                and parm.name in hip._HIPMEMCPY_RECORD_DST_NAMES \
                and parm.is_pointer_to_record(degree=1):
            return ParmIntent.OUT

        # Opaque-handle creators: parm 0 (or parm 0+1 for pitch
        # allocators) is callee-allocated OUT — the callee produces a
        # fresh handle/pointer and the caller's slot is discarded.
        # Doxygen tags these `@param[in, out]` which is wrong — see
        # comment block above.
        if func_name in hip._HIP_HANDLE_CREATOR_OUT_PARM0 and parm_idx == 0:
            return ParmIntent.OUT_CALLEE_ALLOCATED
        if func_name in hip._HIP_HANDLE_CREATOR_OUT_PARM01 and parm_idx in (0, 1):
            return ParmIntent.OUT_CALLEE_ALLOCATED

        # Callee-produced OUT slots the callee writes a fresh value into: rank-0
        # struct handles, and `const char**` slots the callee points at its own
        # internal storage. These are returns. Stated explicitly now that the
        # rank-0 consumer fallback is gone (`const char**` is skipped by both
        # `string_z` and `double_indirection_out`, so it cannot be deferred).
        if (func_name, parm_idx) in (
            ("hipIpcGetMemHandle", 0),   # hipIpcMemHandle_t* — rank-0 struct
            ("hipDeviceGetUuid", 0),     # hipUUID*          — rank-0 struct
            ("hipDrvGetErrorName", 1),   # const char**      — internal string
            ("hipDrvGetErrorString", 1),
        ):
            return ParmIntent.OUT_CALLEE_ALLOCATED
        # Caller-sized char buffers (rank 1): caller-allocated OUT, kept in the
        # argument list (the HIP Cython generator turns `name`/`pciBusId` into
        # returns via `_CSTR_OUT_BUFFERS`; see POINTER_ARGUMENTS.md §4.3).
        #
        # NOTE: `hipMemGetAddressRange`'s `pbase` is deliberately NOT listed
        # here. It is a `hipDeviceptr_t*` (canonically `void**`): a callee-
        # produced pointer returned through a pointer-to-pointer slot.
        # Deferring to the chain lets `documented_param_intent` /
        # `double_indirection_out` classify it as OUT_CALLEE_ALLOCATED (a
        # return), which is correct.
        if (func_name, parm_idx) in (
            ("hipDeviceGetName", 0),
            ("hipDeviceGetPCIBusId", 0),
        ):
            return ParmIntent.OUT
        if (func_name, parm_idx) in (
            ("hipPointerGetAttribute", 0),
            ("hipExtStreamGetCUMask", 2),
        ):
            return ParmIntent.INOUT
        if (func_name, parm_idx) in (
            ("hipExtStreamCreateWithCUMask", 2),
            # Family 4 (see UPSTREAM_BUGS): caller-provided INPUT pointer
            # mistagged `@param[out]` in `hip_runtime_api.h`. Without these
            # overrides `documented_param_intent` trusts the `[out]` tag and
            # the generated binding drops the arg (`hipHostRegister`) or
            # passes NULL (`hipMemcpyToSymbol*`).
            ("hipHostRegister", 0),         # void* hostPtr   (caller-allocated)
            ("hipMemcpyToSymbol", 0),       # const void* symbol
            ("hipMemcpyToSymbolAsync", 0),  # const void* symbol
        ):
            return ParmIntent.IN

        # HIP-specific naming: certain void** parms with these names are
        # callee-allocated OUT (handle creation slots — the callee
        # produces a fresh pointer through the void** slot).
        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr", "ptr", "dev_ptr", "data", "dptr"]:
                return ParmIntent.OUT_CALLEE_ALLOCATED
        # HIP runtime convention: pointer-to-enum is always a scalar OUT
        # (generally a status/attribute return through pointer). The
        # callee writes a fresh value into the scalar slot.
        if parm.is_pointer_to_enum(degree=1):
            return ParmIntent.OUT_CALLEE_ALLOCATED
        # HIP runtime convention: scalar-via-pointer OUT for non-string
        # basic-type pointers (callee writes a fresh scalar). Subsumed by
        # `status_return_out_pointer` once that relational rule lands.
        if parm.is_pointer_to_basic_type(degree=1) and not parm.is_pointer_to_char(
            degree=1
        ):
            return ParmIntent.OUT_CALLEE_ALLOCATED
        return None  # defer to chain

    @staticmethod
    @fallback(*_RUNTIME_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            func_name, parm_idx = node.parent.name, node.parm_index
            # `hipMemcpy*` / `hipMemset*` `hipArray_t` (record*) destinations
            # are caller-allocated output buffers. They rank as scalar (0)
            # by default (record* handle), which would let the rank-0
            # callee-allocation inference drop them from the signature.
            # Override to rank 1 so the (unchanged) inference treats them as
            # caller-allocated buffers, keeping the caller-provided array in
            # the args. Paired with the OUT pin in `ptr_parm_intent`; see
            # UPSTREAM_BUGS Family 1. The `is_pointer_to_record(degree=1)`
            # gate excludes the `void*` destinations and `record*` sources.
            if (
                (func_name.startswith("hipMemcpy") or func_name.startswith("hipMemset"))
                and node.name in hip._HIPMEMCPY_RECORD_DST_NAMES
                and node.is_pointer_to_record(degree=1)
            ):
                return 1
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
        # Callee-produced OUT slots (version scalars, sizes, program/link-state
        # handles, and `lowered_name` — a `const char**` the callee points at
        # its internal mangled-name storage): callee-allocated returns.
        out_callee_parms = (
            ("hiprtcVersion", "major"),
            ("hiprtcVersion", "minor"),
            ("hiprtcCreateProgram", "prog"),
            ("hiprtcGetLoweredName", "lowered_name"),
            ("hiprtcGetProgramLogSize", "logSizeRet"),
            ("hiprtcGetCodeSize", "codeSizeRet"),
            ("hiprtcGetBitcodeSize", "bitcode_size"),
            ("hiprtcLinkCreate", "hip_link_state_ptr"),
            ("hiprtcLinkComplete", "size_out"),
        )
        # NOTE: `hiprtcLinkComplete`'s `bin_out` (`void**`) is deliberately
        # NOT listed. It is a callee-produced pointer returned through a
        # pointer-to-pointer slot; deferring to the chain lets
        # `double_indirection_out` classify it as OUT_CALLEE_ALLOCATED.
        inout_parms = (  # these buffers must be allocated by user
            ("hiprtcGetCode", "code"),
            ("hiprtcGetProgramLog", "log"),
            ("hiprtcGetBitcode", "bitcode"),
        )
        if (parm.parent.name, parm.name) in out_callee_parms:
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
            # `T* handle` creator (e.g. hipblasCreate) — callee produces a
            # fresh opaque handle through the void** slot.
            return ParmIntent.OUT_CALLEE_ALLOCATED
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

    # `HIPBLASLT_*` macros whose bodies are NOT integer constants, so the
    # default `int` macro_type would emit `__Pyx_PyLong_From_int(MACRO)`,
    # which fails to compile. They are only referenced in comments in the
    # public header, so dropping them from export is sufficient (no shim).
    #   * VERSION_TWEAK — bare git-hash token (from hipblaslt-version.h).
    #   * *_INVALID     — `static_cast<T>(v)` (C++ only, no C fallback).
    #   * *_VEC_EXT     — `static_assert(false, "…deprecated…")` traps for
    #                     removed enum values.
    _CODEGEN_BLOCKLIST = frozenset({
        "HIPBLASLT_VERSION_TWEAK",                       # bare git-hash token
        "HIPBLASLT_DATATYPE_INVALID",                    # static_cast<...>
        "HIPBLASLT_COMPUTE_TYPE_INVALID",                # static_cast<...>
        "HIPBLASLT_OPERATION_INVALID",                   # static_cast<...>
        "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT", # static_assert(false, ...)
        "HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT", # static_assert(false, ...)
    })

    @staticmethod
    def node_filter(node: Node):
        if _is_useless_macro(node):
            return False
        if isinstance(node, MacroDefinition):
            if node.name in hipblaslt._CODEGEN_BLOCKLIST:
                return False
            return node.name.startswith("HIPBLASLT_")
        return node.name.startswith("hipblasLt") or node.name.startswith("HIPBLASLT_")

    # Scalar OUT pointers (`T* out` — a single value the callee writes)
    # that read most naturally as Python return values. hipBLASLt reuses
    # hipblas's LAPACK-oriented `ptr_rank`, which leaves a plain `int*` /
    # `size_t*` at the default rank 1, so a documented `@param[out]` scalar
    # only refines to plain `OUT` (a `ListOf*` caller argument) and
    # `hipblasLtGetVersion`'s untagged `version` isn't even seen as OUT.
    # Pinning these to OUT_CALLEE_ALLOCATED + rank 0 makes the wrapper
    # return the value, mirroring amdsmi's `_MISTAGGED_OUT` and hiprtc's
    # `out_callee_parms`. Maps (funcname, parm_name).
    #
    # Deliberately excluded (stay caller-allocated):
    #   * `hipblasLtMatmulAlgoGetHeuristic`'s `heuristicResultsArray[]` — a
    #     caller-sized record array (only its sibling scalar count is here).
    #   * the `*GetAttribute` `buf` slots — caller-sized attribute buffers.
    _SCALAR_OUT_PARMS = frozenset((
        # (funcname, parm_name)
        ("hipblasLtGetVersion", "version"),               # @param tag missing upstream
        ("hipblasLtGetSmCountTarget", "smCountTarget"),
        ("hipblasLtCheckNumericsDrain", "first_nan_call_id"),
        ("hipblasLtMatrixLayoutGetAttribute", "sizeWritten"),
        ("hipblasLtMatmulDescGetAttribute", "sizeWritten"),
        ("hipblasLtMatmulPreferenceGetAttribute", "sizeWritten"),
        ("hipblasLtMatrixTransformDescGetAttribute", "sizeWritten"),
        ("hipblasLtMatmulAlgoGetHeuristic", "returnAlgoCount"),
    ))

    @staticmethod
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Intent classifier for hipBLASLt.

        The `_SCALAR_OUT_PARMS` override runs first (before the doxygen
        rule) so that both the untagged `hipblasLtGetVersion` `version`
        and the documented-but-caller-buffer-shaped `sizeWritten` /
        `returnAlgoCount` scalars become callee-allocated returns. The
        tail delegates to hipblas's body — preserving the `void** handle`
        creator heuristic — then to the shared numerical chain.
        """
        parent = node.parent
        if parent is not None and (parent.name, node.name) in hipblaslt._SCALAR_OUT_PARMS:
            return ParmIntent.OUT_CALLEE_ALLOCATED
        return hipblas.ptr_parm_intent.__wrapped__(node)

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Pointer-rank classifier for hipBLASLt.

        Forces rank 0 for the `_SCALAR_OUT_PARMS` scalars so they render
        as single returned values rather than rank-1 `ListOf*` buffers.
        Everything else delegates to hipblas's LAPACK-letter / `handle`
        rank body.
        """
        if isinstance(node, Parm):
            parent = node.parent
            if parent is not None and (parent.name, node.name) in hipblaslt._SCALAR_OUT_PARMS:
                return 0
        return hipblas.ptr_rank.__wrapped__(node)

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
            # hipsolverCreate-style handle creator — callee-allocated.
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
            # ``pnccl*`` is RCCL's profiling/instrumentation thin layer
            # (PMPI-style mirror of every ``nccl*`` symbol). It's not part
            # of the public binding surface — drop it.
            if node.name.startswith("nccl"):
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

    # Hardcoded overrides for RCCL functions whose upstream doxygen
    # `@param[out]` tag is wrong: the parm is actually a caller-
    # allocated device buffer that the function writes into. Trusting
    # the tag would push the buffer into the python return tuple,
    # silently dropping it from the function signature so the user
    # has no way to pass their own pre-allocated buffer.
    _MISTAGGED_INOUT = frozenset((
        # (funcname, parm_name)
        ("ncclReduce",  "recvbuff"),
        ("pncclReduce", "recvbuff"),
    ))

    @staticmethod
    @fallback(*_INPLACE_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        parent = node.parent
        if parent is not None and (parent.name, node.name) in rccl._MISTAGGED_INOUT:
            return ParmIntent.INOUT
        if node.is_pointer_to_record(degree=2):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "comm"),
                ("pncclCommInitAll", "comm"),
            ):
                return ParmIntent.INOUT
            # `ncclComm_t** comm` creators — defer to the chained
            # `double_indirection_out`, which classifies non-const `T**`
            # as OUT_CALLEE_ALLOCATED (callee produces a fresh handle).
            return None
        if node.is_pointer_to_record(degree=1):
            if node.parent.name == "ncclGetUniqueId":
                # `ncclUniqueId* uniqueId` — rank-0 callee-produced handle
                # (degree 1, so not caught by `double_indirection_out`).
                return ParmIntent.OUT_CALLEE_ALLOCATED
        if node.is_pointer_to_basic_type(degree=1):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "devlist"),
                ("pncclCommInitAll", "devlist"),
            ):
                return ParmIntent.IN
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
            return ParmIntent.OUT  # caller-sized buffer, not a handle
        if node.is_pointer_to_record(degree=2):
            # `T**` handle creator — defer to the chained
            # `double_indirection_out` (OUT_CALLEE_ALLOCATED).
            return None
        if node.is_pointer_to_basic_type(degree=1):
            if node.name == "output_data":
                return ParmIntent.INOUT
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
        # `odata` in `hipfftExec*` is INOUT, NOT OUT — the caller
        # allocates the (device) output buffer; the function fills it.
        # Doxygen tags it `@param[out] odata` in
        # `/opt/rocm/include/hipfft/hipfft.h`, which the new
        # `documented_param_intent` rule trusts and would drop the
        # `odata` arg from the python signature. Override here so it
        # stays in args. Same upstream-doxygen issue as hipMemcpy /
        # hipStreamCreate (see the upstream bug report filed against
        # ROCm/HIP for the mistagged intent annotations).
        if node.parent is not None and node.parent.name.startswith(
            "hipfftExec"
        ) and node.name == "odata":
            return ParmIntent.INOUT
        if node.is_pointer_to_record(degree=2):
            # `hipfftHandle* plan` creator — defer to the chained
            # `double_indirection_out` (OUT_CALLEE_ALLOCATED).
            return None
        if node.name == "workSize":
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
        if node.is_pointer_to_record(degree=2):
            # opaque descriptor/handle creator (`T**`) — callee-allocated
            # return. Stated explicitly (not deferred) because the const
            # descriptor creators (`hipsparseCreateConst*`) are `const T**`,
            # which `double_indirection_out` skips (const guard).
            return ParmIntent.OUT_CALLEE_ALLOCATED
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            # `hipsparseHandle_t* handle` creator (void** slot) — callee
            # produces a fresh opaque handle, like hipblasCreate /
            # hipsolverCreate. Paired with the rank-0 override in `ptr_rank`
            # so it renders as a single returned handle rather than a
            # rank-1 `ListOfPointer` buffer.
            return ParmIntent.OUT_CALLEE_ALLOCATED
        return None  # defer to chain

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.name == "handle":
                # `hipsparseHandle_t* handle` (void**) is a single opaque
                # handle slot, not a rank-1 buffer — mirrors hipblas.
                return 0
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
        if _is_useless_macro(node):
            return False
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPTENSOR_")
        return node.name.startswith("hiptensor") or node.name.startswith("HIPTENSOR_")

    # Hardcoded overrides for hiptensor functions whose upstream doxygen
    # `@param[out]` tag is wrong: the parm is actually INOUT — the
    # caller must pre-allocate the buffer (typically a hipMalloc'd
    # device tensor or a sized attribute buffer) and the function
    # writes into it. Trusting the doxygen tag would push these into
    # the python return tuple, which is wrong because the caller
    # still needs to control allocation/lifetime/sizing. Filed
    # upstream against ROCm/rocm-libraries (hipTensor) as
    # /tmp/hiptensor_doxygen_param_intent_mistags.md.
    _MISTAGGED_INOUT = frozenset((
        # (funcname, parm_name)
        ("hiptensorOperationDescriptorGetAttribute", "buf"),
        ("hiptensorPlanGetAttribute",                "buf"),
        ("hiptensorContract",                        "D"),
        ("hiptensorContract",                        "workspace"),
        ("hiptensorContractTrinary",                 "E"),
        ("hiptensorContractTrinary",                 "workspace"),
        ("hiptensorElementwiseBinaryExecute",        "D"),
        ("hiptensorElementwiseTrinaryExecute",       "D"),
        ("hiptensorReduce",                          "D"),
        ("hiptensorReduce",                          "workspace"),
        # Bonus: hiptensorDestroy(handle) is also doxygen-mistagged
        # (`@param[out] handle` on the destructor). Treat as IN.
        # IN is the doxygen-rule default for non-pointer-to-pointer
        # handles, so we don't need an explicit override here — the
        # by-value `hiptensorHandle_t handle` parm doesn't even hit
        # ptr_parm_intent.
    ))

    @staticmethod
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Intent classifier for hipTensor.

        Override for the upstream-doxygen-mistagged caller-allocated
        buffers (see ``_MISTAGGED_INOUT``). Runs BEFORE the doxygen
        rule because the doxygen tag is the very thing that's wrong.
        Otherwise delegates to hipblas (handle pointer-to-void), then
        the chain.
        """
        parent = node.parent
        if parent is not None and (parent.name, node.name) in hiptensor._MISTAGGED_INOUT:
            return ParmIntent.INOUT
        # Delegate the tail to hipblas's body — preserves the
        # `pointer-to-void degree=2 named handle → OUT` heuristic.
        return hipblas.ptr_parm_intent.__wrapped__(node)

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Pointer-rank classifier for hipTensor.

        Override for the one currently-known scalar OUT param whose
        generic chain misclassifies as a list.

        ``hiptensorEstimateWorkspaceSize(uint64_t* workspaceSizeEstimate)``
        is a single-element OUT (the function writes one ``uint64_t``).
        Without this override the generic ``array_with_length_param``
        rule would flag it as rank=1 and the Python wrapper would
        return a ``ListOfUnsignedLong`` instead of a plain ``int``.
        """
        if isinstance(node, Parm) and node.name == "workspaceSizeEstimate":
            return 0
        # Delegate the tail to hipblas's body for the LAPACK-letter /
        # alpha/beta handling.
        return hipblas.ptr_rank.__wrapped__(node)


class hipdnn_backend:
    """Controls for hipDNN backend — graph-style DNN primitives.

    The header `hipdnn_backend.h` is a clean C ABI: depends only on
    `<hip/hip_runtime_api.h>` and a tree of small per-attribute
    enum/typedef sub-headers. Strict-prefix filter keeps only
    `hipdnn*` / `HIPDNN_*` symbols.
    """

    # Empty-body include-guard-style marker, not a value-carrying
    # constant: `#define HIPDNN_CALLBACK_TYPES_DEFINED` after a
    # corresponding `#ifndef`. The default macro_type would treat it
    # as int and emit `__Pyx_PyLong_From_int(HIPDNN_CALLBACK_TYPES_DEFINED)`
    # which expands to `__Pyx_PyLong_From_int()` ("too few arguments").
    # Same shape as HSA's HSA_LARGE_MODEL — see the hsa class.
    _CODEGEN_BLOCKLIST = frozenset((
        "HIPDNN_CALLBACK_TYPES_DEFINED",
    ))

    @staticmethod
    def node_filter(node: Node):
        if _is_useless_macro(node):
            return False
        if node.name in hipdnn_backend._CODEGEN_BLOCKLIST:
            return False
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
        if _is_useless_macro(node):
            return False
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HIPSPARSELT_")
        return node.name.startswith("hipsparseLt") or node.name.startswith("HIPSPARSELT_")

    # Reuse the hipsparse heuristics — opaque-handle and pointer-rank
    # conventions are identical for the sparse extension family.
    macro_type = hipsparse.macro_type
    raw_comment_cleaner = hipsparse.raw_comment_cleaner

    # Scalar OUT pointers (`T* out` — a single value the callee writes) that
    # read most naturally as Python return values. hipSPARSELt documents these
    # as `@param[out]`, but ROCm's split doxygen style (`@param[out]` on one
    # line, ` *  version` on the next) defeats `documented_param_intent`'s tag
    # regex, so `hipsparseLtGetVersion`/`hipsparseLtGetProperty` otherwise stay
    # caller-allocated `int*` arguments. Pinning them to OUT_CALLEE_ALLOCATED +
    # rank 0 makes the wrapper return the value, mirroring hipBLASLt's
    # `_SCALAR_OUT_PARMS`. Maps (funcname, parm_name).
    #
    # Deliberately excluded (stays caller-allocated): `hipsparseLtGetGitRevision`'s
    # `rev` — a caller-sized `char*` string buffer, not a scalar.
    _SCALAR_OUT_PARMS = frozenset((
        # (funcname, parm_name)
        ("hipsparseLtGetVersion", "version"),
        ("hipsparseLtGetProperty", "value"),
    ))

    @staticmethod
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Intent classifier for hipSPARSELt.

        The `_SCALAR_OUT_PARMS` override runs first (before the doxygen
        rule) so the version/property getters become callee-allocated
        returns even though their split-style `@param[out]` tags aren't
        machine-parseable. The tail delegates to hipsparse's body.
        """
        parent = node.parent
        if parent is not None and (parent.name, node.name) in hipsparselt._SCALAR_OUT_PARMS:
            return ParmIntent.OUT_CALLEE_ALLOCATED
        return hipsparse.ptr_parm_intent.__wrapped__(node)

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Pointer-rank classifier for hipSPARSELt.

        Forces rank 0 for the `_SCALAR_OUT_PARMS` scalars so they render
        as single returned values rather than rank-1 `ListOf*` buffers.
        Everything else delegates to hipsparse's rank body.
        """
        if isinstance(node, Parm):
            parent = node.parent
            if parent is not None and (parent.name, node.name) in hipsparselt._SCALAR_OUT_PARMS:
                return 0
        return hipsparse.ptr_rank.__wrapped__(node)


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
    # @param[out] "user-provided buffer ... must contain at least
    # max_processes entries ... Must be allocated by user" — a caller-sized
    # record array driven by the in,out `max_processes` count, NOT a single
    # callee-allocated record. Without this it would be a degree-1
    # pointer-to-record => rank 0 => returned single struct.
    "list",
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

    # Top-level types that lack the `amdsmi_` prefix but are referenced by
    # amdsmi-prefixed functions / records, so the binding is incomplete
    # without them. (`<stdint.h>` types — uint{8,16,32,64}_t, int{32,64}_t —
    # don't need to be listed here: the codegen already handles them via
    # `from libc.stdint cimport *`.)
    _EXTRA_TYPES = frozenset({
        "amd_metrics_table_header_t",   # used by amdsmi_gpu_metrics_t and
                                        # amdsmi_get_gpu_metrics_header_info
        "processor_type_t",             # used by amdsmi_get_processor_type
        # `struct timespec` (<time.h>, transitively included by amdsmi.h) is
        # a BY-VALUE field of amdsmi_fabric_telemetry_dataset_t.timestamp.
        # Admitting it lets the codegen emit its `cdef struct timespec:`
        # layout AND a `.fromPtr` wrapper (mirrors the in-header by-value
        # record amdsmi_cper_timestamp_t). It is a tagged struct (not a
        # typedef), so it renders as `cdef struct timespec` -> valid C
        # `struct timespec`; the real per-platform layout comes from the
        # `cdef extern from "amd_smi/amdsmi.h"` block, so no ABI is baked in.
        "timespec",
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
        if _is_useless_macro(node):
            return False
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

    # Hardcoded overrides for amdsmi functions whose upstream doxygen
    # `@param[in,out]` tag is semantically wrong. The amdsmi header
    # uses `[in,out]` for two distinct cases:
    #
    #  - Genuine INOUT: the two-call count-then-fill pattern, where
    #    the caller pre-populates a count value and the function
    #    consumes/updates it (e.g. `socket_count` in
    #    `amdsmi_get_socket_handles`). These stay routed through the
    #    doxygen rule.
    #
    #  - Pure OUT mistagged as `[in,out]`: a fixed-size output buffer
    #    or pointer-to-scalar that the caller does NOT pre-populate;
    #    the function just writes (e.g. `version` in
    #    `amdsmi_get_lib_version`). These need an override so the
    #    parm is moved to the return tuple instead of being dragged
    #    into the python args.
    #
    # A third upstream-doxygen failure mode also lands here: a param
    # documented by its *type* spelling rather than its identifier, so
    # the doxygen rule cannot match it by name and the verb/handle
    # fallback misclassifies it. `amdsmi_get_node_handle`'s OUT
    # `node_handle` is tagged `@param[out] amdsmi_node_handle*` (the
    # type, not `node_handle`), so without an override the
    # `_handle -> IN` fallback would bind a *returned* handle as a
    # caller input. Forcing it to OUT lets the rank-0 callee-allocated
    # fallback return it.
    #
    # An upstream bug report (same overloaded-tag pattern across
    # many amdsmi `amdsmi_get_*` getters) has been filed against
    # ROCm/amdsmi. This entry list is incremental — extend as
    # additional ones surface.
    _MISTAGGED_OUT = frozenset((
        ("amdsmi_get_lib_version", 0),  # `version` is pure OUT
        ("amdsmi_get_node_handle", 1),  # `node_handle` is a returned OUT
                                        # handle (doc'd by type spelling)
    ))

    # Caller-allocated array buffers that upstream doxygen mistags
    # `@param[out]` (instead of the `[in,out]` its siblings use), which
    # would otherwise refine to OUT_CALLEE_ALLOCATED for a void** shape
    # and emit a single `&ptr` slot — corrupting memory for what is
    # actually a caller-sized array the callee fills. Forcing INOUT keeps
    # them on the caller-allocated path (a `ListOfPointer` argument),
    # matching the identically-typed `amdsmi_get_processor_handles`.
    _FORCE_INOUT = frozenset((
        ("amdsmi_get_processor_handles_by_type", 2),  # processor_handles
    ))

    @staticmethod
    def ptr_parm_intent(node: Parm):
        """Classify pointer parameter intent for amdsmi APIs.

        Priority of rules (most specific first):
          0. Hardcoded overrides for upstream-doxygen-mistagged parms
             (`_MISTAGGED_OUT` -> OUT, `_FORCE_INOUT` -> INOUT). Run
             BEFORE the doxygen rule because the doxygen tag is the very
             thing that's wrong.
          1. Doxygen `@param[in|out|in,out]` tag on the parent function —
             trusted as the source of truth. Covers 294 of 302 pointer
             parms in amdsmi.h. Delegated to
             `generic.documented_param_intent` (the same helper now
             slotted at the head of all `_*_INTENT_CHAIN` tuples for
             libraries that don't have a custom dispatch).
          2. Opaque handle parameters — names ending in `_handle` are
             IN (typedef'd void* values, NOT pointer-to-output-buffer).
          3. INOUT name set — paired count/len params docs as @param[in,out].
          4. amdsmi_set_*  → IN (~44 functions).
          5. amdsmi_get_*  → OUT (~160 functions).
          6. amdsmi_init / amdsmi_shut_down / amdsmi_status_string → IN.
          7. Default → IN.
        """
        fname = node.parent.name
        if (fname, node.parm_index) in amdsmi._MISTAGGED_OUT:
            # Both entries are callee-produced (version scalar, node_handle):
            # rank-0 slots the callee writes -> callee-allocated returns.
            return ParmIntent.OUT_CALLEE_ALLOCATED
        if (fname, node.parm_index) in amdsmi._FORCE_INOUT:
            return ParmIntent.INOUT
        doxy = generic.documented_param_intent.ptr_parm_intent(node)
        if doxy is not None:
            return doxy
        pname = node.name or ""
        if pname.endswith("_handle"):
            return ParmIntent.IN
        if pname in _AMDSMI_INOUT_PARM_NAMES:
            return ParmIntent.INOUT
        if fname.startswith("amdsmi_set_"):
            return ParmIntent.IN
        if fname.startswith("amdsmi_get_"):
            # Verb catch-all for the handful of undocumented get_* params
            # (the ~294 documented ones resolved above via doxygen). Shape
            # decides allocation: rank-0 scalars are callee-produced returns,
            # caller-sized buffers stay caller-allocated OUT.
            if generic.is_callee_allocated_out_shape(node):
                return ParmIntent.OUT_CALLEE_ALLOCATED
            return ParmIntent.OUT
        if fname in ("amdsmi_init", "amdsmi_shut_down", "amdsmi_status_string"):
            return ParmIntent.IN
        return ParmIntent.IN

    @staticmethod
    def ptr_rank(node: Node):
        """Underlying rank (0=scalar, 1=array) for an amdsmi pointer.

        Rules:
          1. Documented array/buffer parameter names → rank 1. This is
             checked FIRST, before the void**->scalar rule below, so that
             void** *handle arrays* (`socket_handles`, `processor_handles`)
             are treated as rank-1 arrays (and pick up the `ListOfPointer`
             wrapper) rather than collapsing to a single scalar slot.
          2. Single opaque handles (`amdsmi_*_handle` typedefs of `void*`)
             are scalars even when passed as `handle*` for OUT — detected
             via `is_pointer_to_void` because libclang's canonical type
             traversal sees through the typedef. Only single-handle slots
             reach here (e.g. `node_handle`); the handle *arrays* are
             captured by rule 1.
          3. A single pointer-to-record/enum/basic-type slot is a scalar
             (rank 0). Enums are included here so that a pure `@param[out]`
             `enum_t*` (e.g. `amdsmi_get_processor_type`'s `processor_type`)
             becomes a returned `OUT_CALLEE_ALLOCATED` scalar rather than a
             caller-supplied argument; without this it would hit the default
             below and be treated as a rank-1 array. Enum *arrays* are
             captured by rule 1's buffer-name set (e.g. `sensor_types`).
          4. Default → rank 1 (anything still unclassified is treated as an
             array buffer).
        """
        if not isinstance(node, Parm):
            return 1
        # Array/buffer params (incl. void** handle arrays). MUST precede the
        # void**->scalar rule, which is meant only for single handle slots.
        if (node.name or "") in _AMDSMI_BUFFER_PARM_NAMES:
            return 1
        # A single opaque handle canonicalizes to void* — pointer-to-handle
        # is void**; treat such a single slot as a scalar.
        if node.is_pointer_to_void(degree=1) or node.is_pointer_to_void(degree=2):
            return 0
        if node.is_pointer_to_record(degree=1):
            return 0
        if node.is_pointer_to_enum(degree=1):
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

    # Foreign-prefix types declared inside the parsed HSA headers
    # that nonetheless need to surface in the binding because they
    # appear in admitted function signatures. The `BrigModuleHeader`
    # opaque struct (and its `BrigModule_t` pointer typedef) is
    # referenced by `hsa_ext_program_add_module` /
    # `hsa_ext_program_iterate_modules` parameters — without the
    # ctypedef, Cython can't resolve the type identifier in the
    # rendered function signature.
    _EXTRA_TYPES = frozenset((
        "BrigModuleHeader",
        "BrigModule_t",
    ))

    # Targeted exclusions where the codegen can't currently produce a
    # well-formed binding. Keep this list as small as possible and
    # link each entry to its tracking issue.
    #
    # `hsa_amd_memory_copy_op_s` (and its `_t` typedef + the only
    # consumer `hsa_amd_memory_async_batch_copy`) — the C struct uses
    # C11 *anonymous members with no field name* (e.g.
    # `union { void* src; void** src_list; };` and an anonymous union of
    # anonymous structs, all unnamed). The nested-record *admission* gap
    # that used to drop referenced nested types is now fixed
    # (CythonModuleGenerator.walk_filtered_nodes top-most-ancestor rule),
    # so these no longer dangle — BUT the codegen still drops
    # no-field-name anonymous members entirely: the parent struct loses
    # `src`/`dst`/`size`/... and the anon-union-of-anon-structs emits as
    # an empty `cdef union ...: pass`. That is a separate, still-open gap
    # (C11 anonymous-member name injection is not modeled), so the
    # emitted binding would be silently wrong. Keep these three omitted
    # until the anonymous-member path is implemented.
    _CODEGEN_BLOCKLIST = frozenset((
        "hsa_amd_memory_copy_op_s",
        "hsa_amd_memory_copy_op_t",
        "hsa_amd_memory_async_batch_copy",
        # Feature-detection markers, not value-carrying constants:
        # `#define HSA_LARGE_MODEL` (set when `__LP64__` is defined),
        # `#define HSA_LITTLE_ENDIAN` (set when `LITTLEENDIAN_CPU` is
        # defined). The default macro_type classifies them as ints,
        # yielding a `__Pyx_PyLong_From_int(HSA_LARGE_MODEL)` call
        # that expands to `__Pyx_PyLong_From_int()` — too few args.
        # The C-side use is `#ifdef HSA_LARGE_MODEL`, never as an
        # rvalue; nothing user-visible to expose.
        "HSA_LARGE_MODEL",
        "HSA_LITTLE_ENDIAN",
    ))

    @staticmethod
    def node_filter(node: Node):
        if _is_useless_macro(node):
            return False
        if node.name in hsa._CODEGEN_BLOCKLIST:
            return False
        if node.name in hsa._EXTRA_TYPES:
            return True
        if isinstance(node, MacroDefinition):
            return node.name.startswith("HSA_")
        return node.name.startswith("hsa_") or node.name.startswith("HSA_")

    # Hardcoded overrides for HSA functions whose upstream doxygen
    # `@param[out]` tag is wrong: the parm is actually a caller-
    # allocated buffer (host or device memory the user manages).
    # Trusting the tag would push the parm into the return tuple,
    # silently dropping it from the function signature so the user
    # has no way to pass their own pre-allocated buffer.
    _MISTAGGED_INOUT = frozenset((
        # (funcname, parm_name)
        ("hsa_memory_copy", "dst"),
    ))

    @staticmethod
    @fallback(*_NUMERICAL_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Intent classifier for HSA — overrides the upstream-doxygen-
        mistagged caller-allocated buffers, otherwise defers to
        hipblas (handle pointer-to-void degree=2 → OUT) and the chain.
        """
        parent = node.parent
        if parent is not None and (parent.name, node.name) in hsa._MISTAGGED_INOUT:
            return ParmIntent.INOUT
        return hipblas.ptr_parm_intent.__wrapped__(node)

    @staticmethod
    @fallback(*_NUMERICAL_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Scalar-oriented pointer ranks for HSA.

        HSA parameters do not follow hipBLAS's LAPACK naming
        conventions, so borrowing ``hipblas.ptr_rank`` left every
        unmatched pointer at the chain default (rank 1), leaking scalar
        OUT handles/values as rank-1 buffers. Classify the genuine
        single-slot shapes as rank 0 (mirrors ``hip`` / ``amdsmi``):
        opaque handles (canonicalize to ``void*`` / ``void**``),
        pointer-to-enum, pointer-to-record, and non-string
        pointer-to-basic-type. ``void**`` stays rank 0 here (an HSA
        handle slot) rather than taking the generic byte-buffer rank-1
        path. Everything else defers to the chain (documented arrays /
        buffers stay rank 1).
        """
        if isinstance(node, Parm):
            # Opaque handles canonicalize to void*; pointer-to-handle is
            # void** — both are single-slot, not byte buffers.
            if node.is_pointer_to_void(degree=1) or node.is_pointer_to_void(
                degree=2
            ):
                return 0
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


class hipfile:
    """Code generation controls for hipFILE (Accelerated I/O Storage).

    The hipFILE C API uses the `hipFile` prefix for all functions/types and
    `HIPFILE_` for macros. This recipe admits only that `hipFile*` / `HIPFILE_*`
    library surface. Identifiers outside those prefixes are external types
    pulled in via `#include` (hipError_t, off_t, sockaddr, timespec, ...):
    HIP types are cimported, and the plain POSIX structs hipFILE references by
    pointer (`sockaddr`, `timespec`) are hand-declared as opaque, pointer-only
    structs by the systems-wheel generator itself — see
    `generators_systems.generate_hipfile`, which keeps that platform-dependent
    provisioning localized so a future Windows build variant can diverge
    without touching this shared recipe.
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

    # Async byte-count OUT slots: documented ``@param[out]`` but CALLER-
    # allocated — the stream writes them after the call returns, so they must
    # persist as pointer arguments (a rank-0 ``PointerToLong``), not be
    # synthesized as scalar returns. Pinning plain OUT here (ahead of
    # ``documented_param_intent`` in the chain) keeps them caller-allocated;
    # ``ptr_rank`` leaves them at their honest rank 0.
    _ASYNC_CALLER_ALLOCATED_OUT = (
        ("hipFileReadAsync", "bytes_read_p"),
        ("hipFileWriteAsync", "bytes_written_p"),
    )

    @staticmethod
    @fallback(*_RUNTIME_INTENT_CHAIN)
    def ptr_parm_intent(node: Parm):
        """Classify pointer parameter intent for hipFILE APIs.

        Only one library-specific override: the async byte-count OUT slots
        (see ``_ASYNC_CALLER_ALLOCATED_OUT``) are pinned to plain OUT so they
        stay caller-allocated pointer arguments. Everything else defers to the
        chain (``documented_param_intent``, ``double_indirection_out`` for
        handle creation, ``string_z`` for path strings, ``conservative`` for
        pointer-to-const, etc.).
        """
        if (node.parent.name, node.name) in hipfile._ASYNC_CALLER_ALLOCATED_OUT:
            return ParmIntent.OUT
        return None  # defer to chain

    @staticmethod
    @fallback(*_RUNTIME_RANK_CHAIN)
    def ptr_rank(node: Node):
        """Classify single-slot scalar pointers as rank 0 for hipFILE.

        A non-``char`` ``pointer-to-basic-type`` (degree 1) is a single
        value passed by reference. Rank 0 marks it as a single slot rather
        than a rank-1 ``ListOf*`` buffer. Direction and allocation are
        applied downstream: a callee-produced rank-0 OUT
        (``OUT_CALLEE_ALLOCATED``) becomes a scalar return, while a caller-
        allocated rank-0 OUT/INOUT stays a ``PointerTo*`` argument. This
        fixes ``hipFileGetVersion``'s ``unsigned int * major/minor/patch``
        (``@param[out]``), and pairs with the ``OUT_CALLEE_ALLOCATED``
        overrides on ``hipFileGetParameterSizeT`` / ``hipFileGetParameterBool``
        ``value`` (see ``generators_systems._hipfile_node_init``).

        The rule is intentionally shape-based (not gated on intent):
        ``ptr_rank`` must not consult ``is_out_ptr`` / the intent chain,
        because several intent rules themselves query rank
        (``is_indirection`` -> ``ptr_rank``), which would recurse. Direction
        (OUT vs INOUT) is applied downstream, so ``@param[in,out]`` slots
        such as ``hipFileBatchIOGetStatus``'s ``nr`` stay caller-allocated
        rather than becoming scalar returns.

        ``char *`` is excluded so string params stay rank-1 sequences
        (``generic.string_z``); void buffers, handles, and
        pointer-to-record (including the ``hipFileIOEvents_t *`` event
        arrays) are not basic types and keep the chain default.

        The async ``bytes_read_p`` / ``bytes_written_p`` slots keep their
        honest rank 0 here; ``ptr_parm_intent`` pins them to plain OUT so
        they stay caller-allocated ``PointerToLong`` arguments (the stream
        writes them after the call returns) instead of scalar returns.
        """
        if isinstance(node, Parm):
            if node.is_pointer_to_basic_type(degree=1) and not node.is_pointer_to_char(
                degree=1
            ):
                return 0
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
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
        params, OUT_CALLEE_ALLOCATED for `Out*`-prefixed names + a
        per-(fn, parm) lookup, IN otherwise. Order matters.

        The `Out*` / per-(fn, parm) block covers callee-produced outputs:
        opaque handles (`OutEE`, `OutMemBuf`, `T`, ...), scalar outputs
        (`LLVMGetVersion` Major/Minor/Patch, `LLVMGetValueName2` Length),
        and callee-allocated strings (`OutError`, `OutMessage`,
        `ErrorMessage`). They are pinned to `OUT_CALLEE_ALLOCATED` so the
        binding turns them into Python return values regardless of rank.
        The explicit callee-allocation hint is load-bearing for the
        `char **` string params: their rank-1 classification (a NUL-
        terminated char sequence) would otherwise fail the rank-0
        structural fallback in `is_out_callee_allocated_ptr` and leave
        them as positional arguments."""
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
            return ParmIntent.OUT_CALLEE_ALLOCATED
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
