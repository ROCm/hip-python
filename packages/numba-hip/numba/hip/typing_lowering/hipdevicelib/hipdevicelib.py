# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
    Parts derived from the following Numba CUDA files:
    `numba/cuda/libdevicedecl.py`, `numba/cuda/libdeviceimpl.py`.
Note:
    Adds overloaded variants such as variants that take
    uint64 and int64 for math functions.
"""

import threading
import logging

from llvmlite import ir

import rocm.clang.cindex as ci

from rocm.amd_comgr import amd_comgr as comgr

from hip import HIP_VERSION_TUPLE, ROCM_VERSION_TUPLE

from numba.core import cgutils, types

import numba.core.typing.templates as typing_templates
import numba.core.imputils as imputils

from numba.hip.amdgcn import ISA_INFOS
from numba.hip.typing_lowering import stubs as numba_hip_stubs
from numba.hip.util import comgrutils, llvmutils

from .hipsource import *
from . import typemaps


_lock = threading.Lock()

_log = logging.getLogger(__name__)

USER_HIP_EXTENSIONS = ""
USER_HIP_CFLAGS = []

DEVICE_FUN_PREFIX = "NUMBA_HIP_"

_GET = f"{DEVICE_FUN_PREFIX}GET_"


class HIPDeviceLib:
    """Generate Python stubs and bitcode per GPU architecture.

    Implemented as singleton per GPU architecture.

    Note:
        Bitcode is only generated if the respective property
        is queried first. Can take a few seconds (~2 s on EPYC Rome).
    """

    __INSTANCES = {}

    _HIPDEVICELIB_SOURCE: HIPSource = None

    def reload(cls):
        """Rewrite the input HIP source and clear the BC cache."""
        cls._HIPDEVICELIB_SOURCE = HIPDeviceLib._create_hipdevicelib_source()
        cls.__INSTANCES.clear()

    def __new__(cls, amdgpu_arch: str = None):
        """Creates/returns the singleton per AMD GPU architecture."""
        with _lock:
            if not cls._HIPDEVICELIB_SOURCE:
                cls._HIPDEVICELIB_SOURCE = HIPDeviceLib._create_hipdevicelib_source()
            if amdgpu_arch not in cls.__INSTANCES:
                cls.__INSTANCES[amdgpu_arch] = object.__new__(cls)
        return cls.__INSTANCES[amdgpu_arch]

    @staticmethod
    def _create_hipdevicelib_source() -> HIPSource:
        """Create HIPSource from HIPRTC runtime header file.

        Create an object that allows to parse the HIP C++ entities
        of the HIPRTC runtime header.

        Per coordinate struct ``threadIdx``, ``blockIdx``, ``blockDim``, and ``gridDim``;
        and per direction ``x``, ``y``, and ``z``; this routine further adds a
        wrapper function that accesses members of the struct. The wrapper
        functions have the name ``{_GET}<struct>_<direction>``.

        TODO:
            * Support functions that take/return char*.
            * Support functions take/return vector types such as uchar4, int3, ...
            * Support function that take/return __half, which is defined as struct.
        """
        filename = "source.hip"

        def cursor_filter_(cursor: ci.Cursor):
            """Filter what cursors to consider when parsing device functions."""
            if cursor.location.file and cursor.location.file.name.endswith(filename):
                if cursor.kind == ci.CursorKind.FUNCTION_DECL:
                    # print(cursor.displayname)
                    for parm_type_kind_layers in HIPDeviceFunction(
                        cursor
                    ).parm_type_kind_layers(canonical=True):
                        if parm_type_kind_layers[-1] == ci.TypeKind.RECORD:
                            # TODO activate later on for char1...4, int1...4
                            # TODO activate later on for __half, which is a struct
                            return False
                        if parm_type_kind_layers == [
                            ci.TypeKind.POINTER,
                            ci.TypeKind.CHAR_S,
                        ]:
                            return False  # TODO activate later on for C chars
                    if "make_mantissa" in cursor.spelling:
                        return False
                    if cursor.spelling.startswith("__ldg"):
                        return False
                    if cursor.spelling.startswith("operator"):  # TODO activate later on
                        return False
                    return True
            return False

        hiprtc_runtime_source = HIPSource(
            filename=filename,
            source=(
                comgr.ext.HIPRTC_RUNTIME_HEADER
                + HIPDeviceLib._create_extensions()
                + HIPDeviceLib._create_overloads()
                + USER_HIP_EXTENSIONS
            ),
            filter=cursor_filter_,
            append_cflags=["-D__HIPCC_RTC__"] + USER_HIP_CFLAGS,
        )
        hiprtc_runtime_source.check_for_duplicates(log_errors=True)
        return hiprtc_runtime_source

    @staticmethod
    def _create_overloads():
        """Create overloaded variants for certain functions.

        Note:
            We follow ``numba/cuda/cudamath.py`` here.
            The HIPRTC runtime header file already contains
            double-float-overloaded variants for math functions such as
            ``sin`` and ``cos``. This routine adds further overloaded
            variants that take ``long long`` and ``unsigned long long``
            as argument and return a ``double``.
            Function ``pow(*,*)`` already has all required overloads.

            We used the following snippet for generating the list of math functions:

            ```python
            from numba import hip
            import math
            # binary ops
            sorted([key for key in vars(math).keys()
                if key in vars(hip) and not key.startswith("_")
                and key.startswith("is")
                and len(getattr(hip,key)._signatures_[0].args)==1])
            # unary double ops
            sorted([key for key in vars(math).keys()
                if key in vars(hip) and not key.startswith("_")
                and not key.startswith("is")
                and len(getattr(hip,key)._signatures_[0].args)==1])
            ```
        TODO:
            * Overload math functions to accept integer arguments, see
              `numba/cuda/cudamath.py`. Typically, ``double`` versions accept unsigned and signed ``long long``.
            * Overload certain math functions to accept fp16 type too,
              `numba/cuda/cudamath.py`.
        """
        overloads = ""
        # 1) Integer functions with ll suffix
        # popcll: (uint64,) -> uint32
        # ffsll: (uint64,) -> uint32, (int64,) -> uint32
        # brevll: (uint64,) -> uint64
        # clzll: (int64,) -> int32
        overloads += textwrap.dedent(
            """\
            // popc
            unsigned int __attribute__((device)) __popc(unsigned long long _0) {
                return __popcll(_0);
            }
            // ffs
            unsigned int __attribute__((device)) __ffs(unsigned long long _0) {
                return __ffsll(_0);
            }
            unsigned int __attribute__((device)) __ffs(long long _0) {
                return __ffsll(_0);
            }
            // brev
            unsigned long long __attribute__((device)) __brev(unsigned long long _0) {
                return __brevll(_0);
            }
            // clz
            int __attribute__((device)) __clz(unsigned long long _0) {
                return __clzll(_0);
            }
            """
        )
        # 2) Math functions
        # 2.a) Boolean functions
        for fun in ["isfinite", "isinf", "isnan"]:
            constant = (
                "true" if fun == "isfinite" else "false"
            )  # infinity, NaN, finiteness are float concepts
            overloads += textwrap.dedent(
                f"""\
                // {fun}
                bool __attribute__((device)) {fun}(unsigned long long _0) {{
                    return {constant};
                }}
                bool __attribute__((device)) {fun}(long long _0) {{
                    return {constant};
                }}
                """
            )
        # 2.b) Unary functions (real)
        # <fun>(double)->double): <fun>(ull)->double), <fun>(ll)->double)
        # fmt: off
        for fun in [
            'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'ceil', 'cos',
            'cosh', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'floor', "frexp",
            'lgamma', 'log', 'log10', 'log1p', 'log2', "modf", 'sin', 'sinh',
            'sqrt', 'tan', 'tanh', 'trunc',
            'radians', 'degrees', 'gamma',
        ]:
            # fmt: on
            if fun in ("modf","frexp"):
                continue # these are not further overloaded for integers; see numba/cuda/cudamath.py
            overloads += textwrap.dedent(f"""\
                // {fun}
                double __attribute__((device)) {fun}(unsigned long long _0) {{
                    return {fun}(static_cast<double>(_0));
                }}
                double __attribute__((device)) {fun}(long long _0) {{
                    return {fun}(static_cast<double>(_0));
                }}
                """)
        # 2.c) Binary functions (real)
        # <fun>(double,double) -> double: <fun>(ull,ull) -> double, <fun>(ll,ll) -> double
        for fun in [ 'atan2', 'copysign', 'fmod', 'hypot', 'remainder']:
            overloads += textwrap.dedent(f"""\
                // {fun}
                double __attribute__((device)) {fun}(unsigned long long _0, unsigned long long _1) {{
                    return {fun}(static_cast<double>(_0),static_cast<double>(_1));
                }}
                double __attribute__((device)) {fun}(long long _0, long long _1) {{
                    return {fun}(static_cast<double>(_0),static_cast<double>(_1));
                }}
                """)
        # NOTE: function pow(*,*) already has all required overloads.
        fun="ldexp"
        # TODO(HIP/AMD): ROCm 6.2.x & ROCm 6.3.0 AMD COMGR (not hipcc and not HIPRTC) translates the
        #                calls in the body of the below functions into @llvm.ldexp.f32.f32 and
        #                and @llvm.ldexp.f64.f64 for some reason while only
        #                @llvm.ldexp.f*.i32 variants are defined.
        #                This then causes linker errors.
        #                Workarounds such as replacing the function calls with
        #                __builtin_amdgcn_ldexp* and __ocml_ldexp_f* did not change
        #               anything. Therefore, we check the ROCm version and disable
        #               these overloads for the time being.
        if ROCM_VERSION_TUPLE < (6,2,0):
            overloads += textwrap.dedent(f"""\
                // {fun}
                float __attribute__((device)) {fun}(float _0, float _1) {{
                    return {fun}(_0,static_cast<int>(_1));
                }}
                double __attribute__((device)) {fun}(double _0, double _1) {{
                    return {fun}(_0,static_cast<int>(_1));
                }}
                """)
        return overloads

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
                unsigned __attribute__((device)) {_GET}{kind}_{dim}() {{
                    return {kind}.{dim};
                }}
                """
                )
        for dim in "xyz":
            extensions += textwrap.dedent(
                f"""\
            unsigned __attribute__((device)) {_GET}global_id_{dim}() {{
                return threadIdx.{dim} + blockIdx.{dim} * blockDim.{dim};
            }}

            unsigned __attribute__((device)) {_GET}gridsize_{dim}() {{
                return blockDim.{dim}*gridDim.{dim};
            }}
            """
            )
        # NOTE all lower case "warpsize" in "GET_warpsize" is by purpose;
        #      we follow Numba CUDA here.
        extensions += textwrap.dedent(
            f"""
            int __attribute__((device)) {_GET}warpsize() {{
                return warpSize;
            }}
            """
        )
        # radians, degrees, and gamma (unaries)
        extensions += textwrap.dedent(
            """
            // radians
            double __attribute__((device)) radians(double _0) {{
                return _0 * (HIP_PI / 180.0);
            }}
            float __attribute__((device)) radiansf(float _0) {{
                return _0 * (HIP_PI_F / 180.0f);
            }}
            float __attribute__((device)) radians(float _0) {{
                return radiansf(_0);
            }}
            // degrees
            double __attribute__((device)) degrees(double _0) {{
                return _0 * (180.0 / HIP_PI);
            }}
            float __attribute__((device)) degreesf(float _0) {{
                return _0 * (180.0f / HIP_PI_F);
            }}
            float __attribute__((device)) degrees(float _0) {{
                return degreesf(_0);
            }}
            // gamma (Python uses name 'gamma' instead of 'tgamma')
            double __attribute__((device)) gamma(double _0) {{
                return tgamma(_0);
            }}
            float __attribute__((device)) gamma(float _0) {{
                return tgammaf(_0);
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
        Note:
            __init__ might be called multiple times due to the way we implement the
            singleton pattern. Hence, we need to check if attributes set in
            the __init__ routine are already present and if that's
            the case we need to return immediately.
        """
        if hasattr(self, "amdgpu_arch"):
            return
        self._amdgpu_arch: str = None
        self._set_amdgpu_arch(amdgpu_arch)
        self._bitcode = None  # lazily
        self._module = None  # lazily

    @property
    def amdgpu_arch(self):
        return self._amdgpu_arch

    def _set_amdgpu_arch(self, arch: str):
        if arch != None and arch.split(":")[0] not in ISA_INFOS:
            supported_archs = ", ".join((f"{a}" for a in ISA_INFOS.keys()))
            raise ValueError(
                f"{self._amdgpu_arch} must be `None` or one of: {supported_archs} (features may be appended after ':')"
            )
        self._amdgpu_arch = arch

    @staticmethod
    def create_stubs_decls_impls(
        typing_registry: typing_templates.Registry,
        impl_registry: imputils.Registry,
    ):
        """_summary_"""
        p_atomic_op = re.compile(
            r"(safe|unsafe)?[Aa]tomic([A-Z][A-Za-z]+)_?(\w+)?"
        )  # group 1,3 optional

        def function_renamer_splitter_(name: str):
            """Splits atomics and coordinate getters, strips leading "_".

            Examples:

            * Converts ``"atomicAdd"`` to ``["atomic","add"]``.
            * Converts ``"safeAtomicAdd"`` to ``["atomic","safe","add"]``.
            * Converts ``"atomicAdd_system"`` to ``["atomic","system","add"]``.
            * Converts ``"unsafeAtomicAdd_system"`` to ``["atomic","system","safe","add"]``.
            * Converts ``"__syncthreads"`` to ``["syncthreads"]``.
            * Converts ``"{_GET}threadIdx_x"`` to ``["get_threadIdx","x"]``.
            * Converts ``"{_GET}warpsize"`` to ``["get_warpsize"]``.

            Note:
                We must ensure that the stub hierarchy parents do not
                map to functions themselves for the attribute resolution to work
                (see numba/hip/target.py). Therefore, we split the atomics in the
                way shown in the examples, i.e. the operation ("add", "min", ...)
                is always the innermost stub.

            Returns:
                list: Parts of the name, which describe a nesting hierarchy.
            """
            nonlocal p_atomic_op

            # example: (unsafe)Atomic(Add)_(system)
            name = name.lstrip("_")
            if "atomic" in name.lower():
                name = p_atomic_op.sub(repl=r"atomic.\1.\3.\2", string=name)
                name = name.lower()
                return [
                    part for part in name.split(".") if part
                ]  # remove "", some match groups are optional
            elif name == f"{_GET}warpsize":
                return ["get_warpsize"]
            elif name == "lane_id":
                return ["get_lane_id"]

            for kind in (
                "threadIdx",
                "blockIdx",
                "blockDim",
                "gridDim",
                "global_id",
                "gridsize",
            ):
                for dim in "xyz":
                    if name == f"{_GET}{kind}_{dim}":
                        return [f"get_{kind}", dim]
            return [name]

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
                template = typing_templates.make_concrete_template(
                    name=typename, key=stub, signatures=stub._signatures_
                )
                setattr(stub, "_template_", template)
                typing_registry.register(template)

        stubs = HIPDeviceLib._HIPDEVICELIB_SOURCE.create_stubs(
            stub_base_class=numba_hip_stubs.Stub,
            function_renamer_splitter=function_renamer_splitter_,
            stub_processor=process_stub_,
        )
        return stubs

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

        def callgen(context, builder, sig, args):
            nonlocal result_type_numba
            nonlocal parm_types_numba
            lmod = builder.module
            fretty = context.get_value_type(result_type_numba)
            fargtys = [
                context.get_value_type(parm_type) for parm_type in parm_types_numba
            ]
            fnty = ir.FunctionType(fretty, fargtys)
            fn = cgutils.get_or_insert_function(lmod, fnty, func_name)
            return builder.call(fn, args)

        # NOTE: 'impl_registry.lower(...)' is expanded: 'HIPDeviceLib.impl_registry.functions.append((core, key, *numba_parm_types))' and returns 'core'
        return (impl_registry.lower(key, *parm_types_numba)(callgen), parm_types_numba)

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

        def callgen(context, builder, sig, args):
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
            impl_registry.lower(key, *in_parm_types_numba)(callgen),
            in_parm_types_numba,
        )

    @property
    def module(self):
        """Returns the ROCm LLVM module derived from the HIP device lib.

        Returns:
            `rocm.llvm.c.types.LLVMOpaqueModule`:
                The ROCm LLVM module wrapper.
        """
        if self._module == None:
            self._module = llvmutils._get_module(self._bitcode)[0]
        return self._module

    @module.deleter
    def module(self):
        if self._module != None:
            llvmutils._get_module_dispose_all(self._module)
            self._module = None

    def __del__(self):
        """Destroys the ROCm LLVM Python module if allocated."""
        del self.module

    @property
    def bitcode(self):
        """Returns the bitcode-version of the HIP device lib"""
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
        wrappers = HIPDeviceLib._HIPDEVICELIB_SOURCE.render_device_function_wrappers(
            prefix=DEVICE_FUN_PREFIX
        )

        hipdevicelib_src = self._HIPDEVICELIB_SOURCE.source + wrappers

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
            to_llvm_ir=False,
        )  # TODO logbuf, diagnosticbuf not accessible currently due to error check method in rocm.amd_comgr.amd_comgr.ext
        return bcbuf
