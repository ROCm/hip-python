# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

"""Stubs, signature matching, and implementation support for HIP device library functions.

Stubs, signature matching, and implementation support for HIP device library functions.

Note:
    In parts based on the following Numba CUDA files:
    `numba/cuda/libdevicedecl.py` and `numba/cuda/libdeviceimpl.py`.
"""

import threading
import logging

from llvmlite import ir

import rocm.clang.cindex as ci

from rocm.amd_comgr import amd_comgr as comgr

from hip import HIP_VERSION_TUPLE

from numba.core import cgutils, types

import numba.core.typing.templates as typing_templates
import numba.core.imputils as imputils

from numba.hip.amdgputarget import *
from numba.hip import stubs as numba_hip_stubs

from .hipsource import *
from . import typemaps
from . import comgrutils

_lock = threading.Lock()

_log = logging.getLogger(__name__)

DEVICE_FUN_PREFIX = "NUMBA_HIP_"


class HIPDeviceLib:
    """Generate Python stubs and bitcode per GPU architecture.

    Implemented as singleton per GPU architecture.

    Note:
        Bitcode is only generated if the respective property
        is queried first. Can take a few seconds (~2 s on EPYC Rome).
    """

    __INSTANCES = {}

    _HIPRTC_RUNTIME_SOURCE: HIPSource = None

    def __new__(cls, amdgpu_arch: str = None):
        """Creates/returns the singleton per AMD GPU architecture."""
        with _lock:
            if not cls._HIPRTC_RUNTIME_SOURCE:
                cls._HIPRTC_RUNTIME_SOURCE = (
                    HIPDeviceLib._create_hiprtc_runtime_source()
                )
            if amdgpu_arch not in cls.__INSTANCES:
                cls.__INSTANCES[amdgpu_arch] = object.__new__(cls)
        return cls.__INSTANCES[amdgpu_arch]

    @staticmethod
    def _create_hiprtc_runtime_source() -> HIPSource:
        """Create HIPSource from HIPRTC runtime header file.

        Create an object that allows to parse the HIP C++ entities
        of the HIPRTC runtime header.

        Per coordinate struct ``threadIdx``, ``blockIdx``, ``blockDim``, and ``gridDim``;
        and per direction ``x``, ``y``, and ``z``; this routine further adds a
        wrapper function that accesses members of the struct. The wrapper
        functions have the name ``get_<struct>_<direction>``.
        """

        def cursor_filter_(cursor: ci.Cursor):
            """Filter what cursors to consider when parsing device functions."""
            if "threadIdx" in cursor.spelling:
                _log.warn(
                    f"process cursor '{cursor.spelling}' of kind '{cursor.kind.name}'"
                )
            if cursor.kind == ci.CursorKind.FUNCTION_DECL:
                for parm_type_kind_layers in HIPDeviceFunction(
                    cursor
                ).parm_type_kind_layers(canonical=True):
                    if parm_type_kind_layers[-1] == ci.TypeKind.RECORD:
                        return False  # TODO activate later on
                    if parm_type_kind_layers == [
                        ci.TypeKind.POINTER,
                        ci.TypeKind.CHAR_S,
                    ]:
                        return False  # TODO activate later on for C chars
                return not cursor.spelling.startswith(
                    "operator"
                )  # TODO activate later on
            return False

        hiprtc_runtime_source = HIPSource(
            source=comgr.ext.HIPRTC_RUNTIME_HEADER + HIPDeviceLib._create_extensions(),
            filter=cursor_filter_,
            append_cflags=["-D__HIPCC_RTC__"],
        )
        hiprtc_runtime_source.check_for_duplicates(log_errors=True)
        return hiprtc_runtime_source

    @staticmethod
    def _create_extensions():
        """Create extensions such as getters for threadIdx and blockIdx
        that we can easily identify.
        """
        extensions = ""
        # NOTE for some reason, extern "C" cannot be specified.
        # The clang parser seems to ignore these definitions if you do so.
        for kind in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
            for dim in "xyz":
                extensions += textwrap.dedent(
                    f"""\
                unsigned __attribute__((device)) GET_{kind}_{dim}() {{
                    return {kind}.{dim};
                }}
                """
                )
        for dim in "xyz":
            extensions += textwrap.dedent(
                f"""\
            unsigned __attribute__((device)) GET_global_id_{dim}() {{
                return threadIdx.{dim} + blockIdx.{dim} * blockDim.{dim};
            }}

            unsigned __attribute__((device)) GET_gridsize_{dim}() {{
                return blockDim.{dim}*gridDim.{dim};
            }}
            """
            )
        # NOTE all lower case "warpsize" in "GET_warpsize" is by purpose;
        #      we follow Numba CUDA here.
        extensions += textwrap.dedent(
            """
            int __attribute__((device)) GET_warpsize() {{
                return warpSize;
            }}
            """
        )
        return extensions

    def __init__(self, amdgpu_arch: str = None):
        """Constructor.

        Initialize a HIPDeviceLib instance for the given AMD GPU architecture.

        Args:
            amdgpu_arch (`str`):
                An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
                Can also have target features appended that are separated via ":".
                These are stripped away where not needed.

        Note:
            Argument ``amdgpu_arch`` can be `None` but in this case no bitcode can be generated.
            Any attempt will result in an error message.
        """
        self._amdgpu_arch: str = None
        self._set_amdgpu_arch(amdgpu_arch)
        self._bitcode = None  # lazily

    @property
    def amdgpu_arch(self):
        return self._amdgpu_arch

    def _set_amdgpu_arch(self, arch: str):
        if arch != None and arch.split(":")[0] not in AMDGPUTargetMachine.ISA_INFOS:
            supported_archs = ", ".join(
                (f"{a}" for a in AMDGPUTargetMachine.ISA_INFOS.keys())
            )
            raise ValueError(
                f"{self._amdgpu_arch} must be `None` or one of: {supported_archs} (features may be appended after ':')"
            )
        self._amdgpu_arch = arch

    @staticmethod
    def create_stubs_decls_impls():
        """_summary_"""

        def function_renamer_splitter_(name: str):
            """Splits atomics and coordinate getters, strips leading "_".

            Examples:

            * Converts ``"safeAtomicAdd"`` to ``["atomic","add","safe"]``.
            * Converts ``"unsafeAtomicAdd_system"`` to ``["atomic","add","system","unsafe"]``.
            * Converts ``"__syncthreads"`` to ``["syncthreads"]``.
            * Converts ``"GET_threadIdx_x"`` to ``["threadIdx","x"]``.
            * Converts ``"GET_warpsize"`` to ``["warpsize"]``.

            Returns:
                list: Parts of the name, which describe a nesting hierarchy.
            """
            p_atomic = re.compile(r"(safe|unsafe)?[Aa]tomic([A-Z][A-Za-z]+)_?(\w+)?")
            name = name.lstrip("_")
            if "atomic" in name.lower():
                name = p_atomic.sub(repl=r"atomic.\2.\3.\1", string=name).rstrip(".")
                name = name.lower()
                return [
                    part for part in name.split(".") if part
                ]  # remove "", some match groups are optional
            for kind in (
                "threadIdx",
                "blockIdx",
                "blockDim",
                "gridDim",
                "global_id",
                "gridsize",
            ):
                for dim in "xyz":
                    if name == f"GET_{kind}_{dim}":
                        return [kind, dim]
            if name == f"GET_warpsize":
                return ["warpsize"]
            return [name]

        typing_registry: typing_templates.Registry = typing_templates.Registry()
        impl_registry: imputils.Registry = imputils.Registry()

        def process_stub_(stub, parent, device_fun_variants, name_parts):
            """Registers function signatures and call generators for every stub.
            Args:
                stub (object): Subclass of `numba.hip.stubs.Stub`.
                parent (object): Top-most parent type of the stub type.
                device_fun_variants (list):
                   List of device functions that represent overloaded
                   variants of the function represented by the stub.
                name_parts (list): Parts of renamed and splitted name.
            """
            global DEVICE_FUN_PREFIX
            nonlocal typing_registry
            nonlocal impl_registry

            setattr(stub, "_signatures_", [])
            setattr(stub, "_call_generators_", [])
            for device_fun in device_fun_variants:
                _log.debug(
                    f"attempting to create Numba signature for function '{device_fun.displayname}: {device_fun.result_type().spelling}'"
                )
                (
                    success,
                    result_type_numba,
                    in_parm_types_numba,
                    parm_types_numba,
                    parm_is_ptr,
                ) = HIPDeviceLib.create_signature(device_fun)
                if not success:
                    _log.debug(
                        f"stub '{'.'.join(name_parts)}' failed to create Numba signature for function '{device_fun.displayname}'"
                    )  # TODO warn -> debug
                else:
                    _log.debug(
                        f"stub '{'.'.join(name_parts)}': created Numba signature for function '{device_fun.displayname}: {device_fun.result_type().spelling}'"
                    )  # TODO warn -> debug
                    stub._signatures_.append(
                        typing_templates.signature(
                            result_type_numba, *in_parm_types_numba
                        )
                    )
                    # register call generator
                    wrapper_name = DEVICE_FUN_PREFIX + device_fun.mangled_name
                    if isinstance(result_type_numba, types.Tuple):
                        stub._call_generators_.append(
                            HIPDeviceLib.register_call_generator_for_function_with_ptr_parms(
                                impl_registry=impl_registry,
                                func_name=wrapper_name,
                                key=stub,
                                result_type_numba=result_type_numba,
                                parm_is_out_ptr=parm_is_ptr,  # TODO not precise; not taking records into account; ptr parm intent needs to be prescribed
                                parm_types_numba=parm_types_numba,
                            )
                        )
                    else:
                        stub._call_generators_.append(
                            HIPDeviceLib.register_call_generator_for_function_without_ptr_parms(
                                impl_registry=impl_registry,
                                func_name=wrapper_name,
                                key=stub,
                                result_type_numba=result_type_numba,
                                parm_types_numba=in_parm_types_numba,
                            )
                        )
            # register signatures
            if len(stub._signatures_):
                typename = DEVICE_FUN_PREFIX + "_".join(
                    name_parts
                )  # just a unique name TODO check if current is fine
                typing_registry.register(
                    typing_templates.make_concrete_template(
                        typename, stub, stub._signatures_
                    )
                )

        stubs = HIPDeviceLib._HIPRTC_RUNTIME_SOURCE.create_stubs(
            stub_base_class=numba_hip_stubs.Stub,
            function_renamer_splitter=function_renamer_splitter_,
            stub_processor=process_stub_,
        )
        return (stubs, typing_registry, impl_registry)

    @staticmethod  # TODO externalize to typemaps -> typemapping
    def create_signature(device_fun: HIPDeviceFunction):
        """Creates a `numba.core.types` signature from the device function.

        Similar to Numba CUDA, pointer arguments are treated as additional return values
        that are appended to the function's return value.

        Returns:
            `tuple`:
                A 5-tuple consisting of (1) a boolean indicating if clang could be mapped
                successfully to Numba types, (2) the result type, (3) a list of the input parameter types,
                (4) a list of all parameter types, and (5) a mask, a list of bools, that indicates per
                parameter if it has a pointer type or not. All entries of the first three result tuple entries
                are expressed via `numba.core.types` types.
        TODO:
            * Distinguish between void* and pointers to basic type.
              parm_is_ptr[i] -> out_parm[i] is too simple.
        """
        parm_is_ptr = [
            cparser.clang_type_kind(parm_type) == ci.TypeKind.POINTER
            for parm_type in device_fun.parm_types(canonical=True)
        ]
        parm_types_numba = []
        for i, parm_type in enumerate(device_fun.parm_types(canonical=True)):
            if parm_is_ptr[i]:
                parm_types_numba.append(
                    typemaps.map_clang_to_numba_core_type(parm_type.get_pointee())
                )
            else:
                parm_types_numba.append(
                    typemaps.map_clang_to_numba_core_type(parm_type)
                )
        result_types_numba = [
            parm_type_numba
            for i, parm_type_numba in enumerate(parm_types_numba)
            if parm_is_ptr[i]
        ]
        in_parm_types_numba = [
            parm_type_numba
            for i, parm_type_numba in enumerate(parm_types_numba)
            if not parm_is_ptr[i]
        ]
        if device_fun.result_type_kind(canonical=True) != ci.TypeKind.VOID:
            result_types_numba.insert(
                0,
                typemaps.map_clang_to_numba_core_type(
                    device_fun.result_type(canonical=True)
                ),
            )

        # If a in/out parameter value couldn't be mapped or has
        # been mapped to types.void, we do not generate a signature.
        # Note that typemaps.map_clang_to_numba_core_type returns None
        # if a mapping failed.
        successful_mapping = True
        for numba_type in in_parm_types_numba + result_types_numba:
            if numba_type in (None, types.void):
                successful_mapping = False

        if successful_mapping:
            if len(result_types_numba) > 1:
                result_type_numba = types.Tuple(result_types_numba)
            elif len(result_types_numba):
                result_type_numba = result_types_numba[0]
            else:
                result_type_numba = types.void
        else:
            result_type_numba = None

        return (
            successful_mapping,
            result_type_numba,
            in_parm_types_numba,
            parm_types_numba,
            parm_is_ptr,
        )

    @staticmethod
    def register_call_generator_for_function_without_ptr_parms(
        impl_registry: imputils.Registry,
        func_name: str,
        key: object,
        result_type_numba,
        parm_types_numba,
    ):
        """Registers a function call generator for a function WITHOUT pointer parameters.

        Registers a function call generator for a function WITHOUT pointer parameters
        and associates it with the stub object 'key'.

        This generator generates an LLVM function call statement when provided with the arguments
        'context' (LLVM context), 'builder' (LLVM IR buider), 'sig' (signature of the function),
        and 'args' (actual arguments passed to the function call).
        If not done so already, the function call generator further inserts a
        function declaration into the builder's module via `numba.core.cgutils` ("codegen utils").

        Returns:
            `tuple`: A tuple of size 2 that contains (1) the call generator and (2) the argument types.
        """

        def core(context, builder, sig, args):
            nonlocal result_type_numba
            nonlocal parm_types_numba
            lmod = builder.module
            fretty = context.get_value_type(result_type_numba)
            fargtys = [context.get_value_type(arg.ty) for arg in parm_types_numba]
            fnty = ir.FunctionType(fretty, fargtys)
            fn = cgutils.get_or_insert_function(lmod, fnty, func_name)
            return builder.call(fn, args)

        # NOTE: 'impl_registry.lower(...)' is expanded: 'HIPDeviceLib.impl_registry.functions.append((core, key, *numba_parm_types))' and returns 'core'
        return (impl_registry.lower(key, *parm_types_numba)(core), parm_types_numba)

    @staticmethod
    def register_call_generator_for_function_with_ptr_parms(
        impl_registry: imputils.Registry,
        func_name: str,
        key: object,
        result_type_numba,
        parm_types_numba,
        parm_is_out_ptr,
    ):
        """Registers a function call generator for a function WITH output pointer parameters.

        Registers a function call generator for a function WITH output pointer parameters
        and associates it with the stub object 'key'.

        This generator generates an LLVM function call statement when provided with the arguments
        'context' (LLVM context), 'builder' (LLVM IR buider), 'sig' (signature of the function),
        and 'args' (actual arguments passed to the function call).
        If not done so already, the function call generator further inserts a
        function declaration into the builder's module via `numba.core.cgutils` ("codegen utils").

        Example:

        void foo(double * bar)

        double bar

        TODO:
            * Distinguish between pointers to records and basic types, chars.
            * Provide a way to prescribe pointer intent (in, inout, out).

        Returns:
            `tuple`: A tuple of size 2 that contains (1) the call generator and (2) the argument types.
        """

        def core(context, builder, sig, args):
            nonlocal result_type_numba
            nonlocal parm_types_numba
            nonlocal parm_is_out_ptr
            lmod = builder.module

            fargtys = []
            for i, parm_type in enumerate(parm_types_numba):
                ty = context.get_value_type(parm_type)
                if parm_is_out_ptr[i]:
                    ty = ty.as_pointer()
                fargtys.append(ty)

            fretty = context.get_value_type(result_type_numba)

            fnty = ir.FunctionType(fretty, fargtys)
            fn = cgutils.get_or_insert_function(lmod, fnty, func_name)

            # For returned values that are returned through a pointer, we need to
            # allocate variables on the stack and pass a pointer to them.
            actual_args = []
            virtual_args = []
            arg_idx = 0
            for i, parm_type in enumerate(parm_types_numba):
                if parm_is_out_ptr[i]:
                    # Allocate space for return value and add to args
                    tmp_arg = cgutils.alloca_once(
                        builder, context.get_value_type(parm_type.ty)
                    )
                    actual_args.append(tmp_arg)
                    virtual_args.append(tmp_arg)
                else:
                    actual_args.append(args[arg_idx])
                    arg_idx += 1

            ret = builder.call(fn, actual_args)

            # Following the call, we need to assemble the returned values into a
            # tuple for returning back to the caller.
            tuple_args = []
            if result_type_numba != types.void:
                tuple_args.append(ret)
            for parm_type in virtual_args:
                tuple_args.append(builder.load(parm_type))

            if isinstance(result_type_numba, types.UniTuple):
                return cgutils.pack_array(builder, tuple_args)
            else:
                return cgutils.pack_struct(builder, tuple_args)

        # Now register the implementation for the signature without pointer
        # parameters, i.e., where the pointer parameters have been transformed
        # to result parameters.
        in_parm_types_numba = [
            parm_type_numba
            for i, parm_type_numba in enumerate(parm_types_numba)
            if not parm_is_out_ptr[i]
        ]
        # NOTE: 'impl_registry.lower(...)' is expanded: 'HIPDeviceLib.impl_registry.functions.append((core, key, *in_parm_types_numba))' and returns 'core'
        return (
            impl_registry.lower(key, *in_parm_types_numba)(core),
            in_parm_types_numba,
        )

    @property
    def llvm_bc(self):
        """Returns the bitcode-version of the HIPRTC device lib"""
        if self.amdgpu_arch == None:
            raise ValueError("cannot generate bitcode for AMDGPU architecture 'None'")
        if self._bitcode == None:
            self._bitcode = self._create_hiprtc_runtime_bitcode()
        return self._bitcode

    def _create_hiprtc_runtime_bitcode(self):
        """Create bitcode from the HIPRTC runtime header.

        The routine performs the following steps:

        1. Per device function (typically with inline attribute) in the HIPRTC runtime header file,
        this routine generates a wrapper functions that calls the original function.
        This "un-inlines" all functions that have inline and forceinline attributes.
        2. Finally both files (1) HIPRTC runtime header and (2) device function wrappers
        are combined and translated to bitcode via an HIP-to-LLVM-BC AMD COMGR compilation action.

        The name of all device wrapper functions is chosen as ``prefix`` + mangled name of the wrapped function.
        The prefix is prepended to prevent conflicts with the types/functions declared/defined by the HIPRTC runtime header.
        """
        global DEVICE_FUN_PREFIX
        wrappers = HIPDeviceLib._HIPRTC_RUNTIME_SOURCE.render_device_function_wrappers(
            prefix=DEVICE_FUN_PREFIX
        )

        hipdevicelib_src = self._HIPRTC_RUNTIME_SOURCE.source + wrappers

        # print(hipdevicelib_source)
        amdgpu_arch = self.amdgpu_arch.split(":")[
            0
        ]  # TODO check how to pass the features in
        (bcbuf, logbuf, diagnosticbuf) = comgrutils.compile_hip_source_to_llvm(
            amdgpu_arch=amdgpu_arch,
            extra_opts=" -D__HIPCC_RTC__",
            hip_version_tuple=HIP_VERSION_TUPLE,
            comgr_logging=False,
            source=hipdevicelib_src,
            to_llvm_ir=True,
        )  # TODO logbuf, diagnosticbuf not accessible currently due to error check method in rocm.amd_comgr.amd_comgr.ext
        return bcbuf
