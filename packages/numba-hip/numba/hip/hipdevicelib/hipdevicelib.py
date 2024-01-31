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

import math

from llvmlite import ir

import rocm.clang.cindex as ci

from numba.core import cgutils, types
from numba.core.typing.templates import make_concrete_template, signature

import numba.core.typing.templates as typing_templates
import numba.core.imputils as imputils

from rocm.amd_comgr import amd_comgr as comgr

from numba.hip.amdgputarget import *

from hip import HIP_VERSION_TUPLE

from .hipsource import *

from . import typemaps
from . import comgrutils

_lock = threading.Lock()

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

    typing_registry: typing_templates.Registry = typing_templates.Registry()
    impl_registry: imputils.Registry = imputils.Registry()

    def __new__(cls, amdgpu_arch: str = None):
        """Creates/returns the singleton per AMD GPU architecture."""
        with _lock:
            if not cls._HIPRTC_RUNTIME_SOURCE:
                cls._HIPRTC_RUNTIME_SOURCE = HIPDeviceLib._create_hiprtc_runtime_source()
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
            if cursor.kind == ci.CursorKind.FUNCTION_DECL:
                for parm_type_kind_layers in HIPDeviceFunction(
                    cursor
                ).parm_type_kind_layers(canonical=True):
                    if parm_type_kind_layers[-1] == ci.TypeKind.RECORD:
                        return False  # TODO activate later on
                    if parm_type_kind_layers == [ci.TypeKind.POINTER,ci.TypeKind.CHAR_S]:
                        return False # TODO activate later on for C chars
                return not cursor.spelling.startswith(
                    "operator"
                )  # TODO activate later on
            return False

        coordinates = "\n"
        for kind in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
            for dim in "xyz":
                coordinates += textwrap.dedent(
                    f"""\
                extern "C" std::uint32_t __attribute__((device)) get_{kind}_{dim}() {{
                    return {kind}.{dim};
                }}
                """
                )

        hiprtc_runtime_source = HIPSource(
            source=comgr.ext.HIPRTC_RUNTIME_HEADER + coordinates,
            filter=cursor_filter_,
            append_cflags=["-D__HIPCC_RTC__"], # TODO outsorcing
        )
        hiprtc_runtime_source.check_for_duplicates(log_errors=True)
        return hiprtc_runtime_source

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

    @property
    def bitcode(self):
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
        hipdevicelib_src = comgr.ext.HIPRTC_RUNTIME_HEADER + wrappers + coordinates
        wrappers = HIPDeviceLib._HIPRTC_RUNTIME_SOURCE.render_device_function_wrappers(
            prefix=DEVICE_FUN_PREFIX
        )

        hipdevicelib_src = comgr.ext.HIPRTC_RUNTIME_HEADER + wrappers

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

    def _create_stubs_decls_impls(self, stub_base_class=object):
        """_summary_"""

        def function_renamer_splitter_(name: str):
            """Splits atomics, strips leading "_".

            Examples:

            * Converts ``"safeAtomicAdd"`` to ``["atomic","add","safe"]``.
            * Converts ``"unsafeAtomicAdd_system"`` to ``["atomic","add","system","unsafe"]``.
            * Converts ``"__syncthreads"`` to ``["syncthreads"]``.

            Returns:
                list: Parts of the name, which describe a nesting hierarchy.
            """
            p_atomic = re.compile(r"(safe|unsafe)?[Aa]tomic([A-Z][A-Za-z]+)_?(\w+)?")
            name = name.lstrip("_")
            if "atomic" in name.lower():
                name = p_atomic.sub(repl=r"atomic.\2.\3.\1", string=name).rstrip(".")
                name = name.lower()
                return name.split(".")
            for kind in ("threadIdx", "blockIdx", "blockDim", "gridDim"):
                for dim in "xyz":
                    if name == "get_{kind}_{dim}":
                        return [kind, dim]
            return [name]

        def stub_processor_(stub, parent, device_fun_variants, name_parts):
            """Registers function signatures and call generators for every stub.
            Args:
                stub (object): A type.
                parent (object): Top-most parent type of the stub type.
                device_fun_variants (list):
                   List of device functions that represent overloaded
                   variants of the function represented by the stub.
                name_parts (list): Parts of renamed and splitted name.
            """
            nonlocal self
            signatures = []
            for device_fun in device_fun_variants:
                (
                    result_type_numba,
                    in_parm_types_numba,
                    parm_types_numba,
                    parm_is_ptr,
                ) = HIPDeviceLib.create_signature(device_fun)
                signatures.append(signature(result_type_numba, *in_parm_types_numba))
                wrapper_name = PREFIX + device_fun.mangled_name
                if len(result_type_numba) > 1:
                    HIPDeviceLib.register_call_generator_for_function_with_ptr_parms(
                        func_name=wrapper_name,
                        key=stub,
                        parm_is_out_ptr=parm_is_ptr,  # TODO not precise; not taking records into account; ptr parm intent needs to be prescribed
                        parm_types_numba=parm_types_numba,
                    )
                else:
                    HIPDeviceLib.register_call_generator_for_function_without_ptr_parms(
                        func_name=wrapper_name,
                        key=stub,
                        result_type_numba=result_type_numba,
                        parm_types_numba=in_parm_types_numba,
                    )
            # register signatures
            HIPDeviceLib.typing_registry.register(
                make_concrete_template(PREFIX + "_".join(name_parts), stub, signatures)
            )

        return HIPDeviceLib._HIPRTC_RUNTIME_SOURCE.create_stubs(
            stub_base_class=stub_base_class,
            function_renamer_splitter=function_renamer_splitter_,
            stub_processor=stub_processor_,
        )

    @staticmethod
    def create_signature(device_fun: HIPDeviceFunction):
        """Creates a `numba.core.types` signature from the device function.

        Similar to Numba CUDA, pointer arguments are treated as additional return values
        that are appended to the function's return value.

        Returns:
            `tuple`:
                A 4-tuple consisting of (1) the result type, (2) a list of the input parameter types,
                (3) a list of all parameter types, and (4) a mask, a list of bools, that indicates per
                parameter if it has a pointer type or not. All entries of the first three result tuple entries
                are expressed via `numba.core.types` types.
        """
        parm_types_numba = [
            typemaps.map_clang_to_numba_core_type(parm_type)
            for parm_type in device_fun.parm_types(canonical=True)
        ]
        parm_is_ptr = [
            parm_type.kind == ci.TypeKind.POINTER
            for parm_type in device_fun.parm_types(canonical=True)
        ]
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
                    device_fun.result_type_kind(canonical=True)
                ),
            )

        if len(result_types_numba) > 1:
            result_type_numba = types.Tuple(result_types_numba)
        else:
            result_type_numba = result_types_numba[0]

        return (
            result_type_numba,
            in_parm_types_numba,
            parm_types_numba,
            parm_is_ptr,
        )

    @staticmethod
    def register_call_generator_for_function_without_ptr_parms(
        func_name: str, key, result_type_numba, parm_types_numba
    ):
        """Registers a function call generator for a function WITHOUT pointer parameters.

        Registers a function call generator for a function WITHOUT pointer parameters
        and associates it with the stub object 'key'.

        This generator generates an LLVM function call statement when provided with the arguments
        'context' (LLVM context), 'builder' (LLVM IR buider), 'sig' (signature of the function),
        and 'args' (actual arguments passed to the function call).
        If not done so already, the function call generator further inserts a
        function declaration into the builder's module via `numba.core.cgutils` ("codegen utils").
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

        # Note below is expanded: 'HIPDeviceLib.impl_registry.functions.append((core, key, *numba_parm_types))'
        HIPDeviceLib.impl_registry.lower(key, *parm_types_numba)(core)

    @staticmethod
    def register_call_generator_for_function_with_ptr_parms(
        func_name: str, key, result_type_numba, parm_types_numba, parm_is_out_ptr
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

        TODOs:
            * Distinguish between pointers to records and basic types, chars.
            * Provide a way to prescribe pointer intent (in, inout, out).
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
        HIPDeviceLib.impl_registry.lower(key, *in_parm_types_numba)(core)
