# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Per-module generators for the **rocm-bindings-hip** wheel.

Each `generate_*` function returns a configured `CythonModuleGenerator`
for one Cython module (`rocm.bindings.hip`, `rocm.bindings.hiprtc`).
The orchestrator (`binding_generator.py`) wires them up with the rest of
the recipe.

Required globals are passed in as keyword-only arguments — no module-
level state is read from the orchestrator's globals. The `generate_hip`
function additionally returns the HIP version triple it discovered
in the parsed headers so the orchestrator can populate the
`generated_versions.cmake` files.
"""

import ctypes
import textwrap

import interfacegen
from interfacegen.cython import CythonModuleGenerator
from interfacegen.support.recipes import rocm as controls
from interfacegen.tree import MacroDefinition, Node, Parm


def _toclassname(name: str) -> str:
    return name[0].upper() + name[1:]


def _make_header_arg(header_relpath: str, header_content: str = None):
    """Build the header argument for CythonModuleGenerator."""
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def generate_hip(
    *,
    include_dir: str,
    header_relpath: str = "hip/hip_runtime.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    """Build the `rocm.bindings.hip` generator.

    Returns:
        tuple[CythonModuleGenerator, tuple[int, int, int, str]]:
            (generator, (HIP_VERSION_MAJOR, MINOR, PATCH, GITHASH)).
            The version triple is recovered from the parsed
            `hip_runtime.h` macros so the orchestrator can write
            `generated_versions.cmake` from it.
    """

    def hip_ptr_complicated_type_handler(parm: Node):
        if (parm.parent.name, parm.name) == ("hipModuleLaunchKernel", "extra"):
            return (
                f"rocm.bindings._hip_helpers.{_toclassname(parm.parent.name)}_{parm.name}"
            )
        if (parm.parent.name, parm.name) in (
            ("hipMalloc", "ptr"),
            ("hipExtMallocWithFlags", "ptr"),
            ("hipMallocManaged", "dev_ptr"),
            ("hipMallocAsync", "dev_ptr"),
            ("hipMallocFromPoolAsync", "dev_ptr"),
        ):
            if parm.parent.name == "hipExtMallocWithFlags":
                size = "sizeBytes"
            else:
                size = "size"
            parm.parent.python_body_prepend_before_return(
                f"{parm.name}.configure(_force=True,shape=(cpython.long.PyLong_FromUnsignedLong({size}),))"
            )
            return "rocm.bindings.util.types.DeviceArray"

        return default_ptr_handler(parm)

    def hip_node_init(node: Node):
        if isinstance(node, interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("hip"):
                # hip routines without hipError_t return status —
                # we force them to not throw exceptions and to
                # always return hipSuccess as first return value.
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"
                node.prepend_python_return_value(
                    "hipError_t.hipSuccess",
                    "hipError_t",
                    "Always returns `~.hipError_t.hipSuccess`.",
                )
        elif isinstance(node, interfacegen.tree.Parm):
            func_name, parm_idx = node.parent.name, node.parm_index
            if (func_name, parm_idx) in (
                ("hipDeviceGetName", 0),
                ("hipDeviceGetPCIBusId", 0),
            ):
                func = node.parent
                assert isinstance(func, interfacegen.cython.Function)
                len_param: interfacegen.tree.Parm = func.get_parm(1)
                func.python_body_prepend_before_c_interface_call(
                    f"{node.name}.malloc({len_param.name})"
                )

    def renamer(name: str):
        return interfacegen.cython.DEFAULT_RENAMER(controls.hip.renamer(name))

    def macro_type(node: MacroDefinition):
        macro_name = node.name
        if macro_name in controls.hip.void_p_macros:
            return ctypes.c_ulonglong(controls.hip.void_p_macros[macro_name])
        return controls.hip.macro_type(node)

    generator = CythonModuleGenerator(
        "rocm.bindings.hip",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libamdhip64.so",
        modifiers_lazy_loader=" except? hipErrorInitializationError nogil",
        error_return_value_lazy_loader="hipErrorInitializationError",
        # we hijack hipError_t constant hipErrorInitializationError for
        # propagating exceptions; see Cython docs on language_basics
        # error-return-values.
        node_init=hip_node_init,
        renamer=renamer,
        node_filter=controls.hip.node_filter,
        ptr_parm_intent=controls.hip.ptr_parm_intent,
        ptr_rank=controls.hip.ptr_rank,
        ptr_complicated_type_handler=hip_ptr_complicated_type_handler,
        macro_type=macro_type,
        raw_comment_cleaner=controls.hip.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    cimport rocm.bindings._hip_helpers
    """
    )

    hip_version_major = 0
    hip_version_minor = 0
    hip_version_patch = 0
    hip_version_githash = ""
    for node in generator.backend.root.walk():
        if isinstance(node, MacroDefinition):
            last_token = list(node.cursor.get_tokens())[-1].spelling
            if node.name == "HIP_VERSION_MAJOR":
                hip_version_major = int(last_token)
            elif node.name == "HIP_VERSION_MINOR":
                hip_version_minor = int(last_token)
            elif node.name == "HIP_VERSION_PATCH":
                hip_version_patch = int(last_token)
            elif node.name == "HIP_VERSION_GITHASH":
                hip_version_githash = last_token.strip('"')

    return generator, (
        hip_version_major,
        hip_version_minor,
        hip_version_patch,
        hip_version_githash,
    )


def generate_hiprtc(
    *,
    include_dir: str,
    header_relpath: str = "hip/hiprtc.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
    rocm_version_tuple: tuple,
):
    """Build the `rocm.bindings.hiprtc` generator.

    `rocm_version_tuple` is `(major, minor, patch)`; it gates the
    `hipJitOption` cross-import that only exists from ROCm 6.4 onward.
    """

    def hiprtc_ptr_complicated_type_handler(node: Node):
        if isinstance(node, Parm):
            if (node.parent.name, node.name) in (
                ("hiprtcCompileProgram", "options"),
                ("hiprtcCreateProgram", "headers"),
                ("hiprtcCreateProgram", "includeNames"),
            ):
                return "rocm.bindings.util.types.ListOfBytes"
            if (node.parent.name, node.name) == (
                "hiprtcLinkCreate",
                "option_ptr",
            ):
                return "rocm.bindings._hiprtc_helpers.HiprtcLinkCreate_option_ptr"
            if (node.parent.name, node.name) == (
                "hiprtcLinkCreate",
                "option_vals_pptr",
            ):
                return "rocm.bindings.util.types.ListOfPointer"
            if (node.parent.name, node.parm_index) in (
                ("hiprtcLinkComplete", 1),
                ("hiprtcGetCode", 1),
                ("hiprtcGetBitcode", 1),
            ):
                return "rocm.bindings.util.types.NDBuffer"
        return default_ptr_handler(node)

    def hiprtc_node_init(node: Node):
        if isinstance(node, interfacegen.tree.Function):
            if not node.is_enum and node.name.startswith("hiprtc"):
                node.error_return_value_lazy_loader = None
                node.modifiers_lazy_loader = " noexcept nogil"

    generator = CythonModuleGenerator(
        "rocm.bindings.hiprtc",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhiprtc.so",
        # we hijack hiprtcResult constant HIPRTC_ERROR_INTERNAL_ERROR
        # for propagating exceptions.
        modifiers_lazy_loader=" except? HIPRTC_ERROR_INTERNAL_ERROR nogil",
        error_return_value_lazy_loader="HIPRTC_ERROR_INTERNAL_ERROR",
        node_init=hiprtc_node_init,
        node_filter=controls.hiprtc.node_filter,
        ptr_parm_intent=controls.hiprtc.ptr_parm_intent,
        ptr_rank=controls.hiprtc.ptr_rank,
        ptr_complicated_type_handler=hiprtc_ptr_complicated_type_handler,
        cflags=generator_args,
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
        cimport rocm.bindings._hiprtc_helpers
        """
    )

    if rocm_version_tuple[:2] >= (6, 4):
        generator.c_interface_decl_prolog += textwrap.dedent(
            """\
            from rocm.bindings.cyhip cimport hipJitOption
            from rocm.bindings.cyhip cimport hipJitOption as hiprtcJIT_option
            from rocm.bindings.cyhip cimport hipJitInputType
            from rocm.bindings.cyhip cimport hipJitInputType as hiprtcJITInputType
            """
        )

        generator.python_interface_impl_prolog += textwrap.dedent(
            """\
            from rocm.bindings.hip import _hipJitOption__Base
            from rocm.bindings.hip import hipJitOption
            from rocm.bindings.hip import hipJitOption as hiprtcJIT_option
            from rocm.bindings.hip import _hipJitInputType__Base
            from rocm.bindings.hip import hipJitInputType
            from rocm.bindings.hip import hipJitInputType as hiprtcJITInputType
            """
        )

    return generator
