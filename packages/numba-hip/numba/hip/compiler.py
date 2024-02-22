# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, typing, funcdesc, config, compiler, sigutils
from numba.core.compiler import (
    sanitize_compile_result_entries,
    CompilerBase,
    DefaultPassBuilder,
    Flags,
    Option,
    CompileResult,
)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import LoweringPass, PassManager, register_pass
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.typed_passes import IRLegalization, NativeLowering, AnnotateTypes
from warnings import warn

from numba.hip.api import get_current_device

from numba.hip import target
from numba.hip import codegen


def _options_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, dict)
        return x


class HIPFlags(Flags):
    amdgpu_arch = Option(
        type=str,
        default=None,
        doc="AMD GPU architecture.",
    )
    hip_options = Option(  # note plain 'options' must not be used as it overrides a member of the base class 'Flags'
        type=_options_type,
        default=None,
        doc="HIP compilation options",
    )  # TODO add options for particular stages (linking, compiling, etc.)


# TODO read that carefully
# The HIPCompileResult (HCR) has a specially-defined entry point equal to its
# id.  This is because the entry point is used as a key into a dict of
# overloads by the base dispatcher. The id of the HCR is the only small and
# unique property of a CompileResult in the HIP target (cf. the CPU target,
# which uses its entry_point, which is a pointer value).
#
# This does feel a little hackish, and there are two ways in which this could
# be improved:
#
# 1. We could change the core of Numba so that each CompileResult has its own
#    unique ID that can be used as a key - e.g. a count, similar to the way in
#    which types have unique counts.
# 2. At some future time when kernel launch uses a compiled function, the entry
#    point will no longer need to be a synthetic value, but will instead be a
#    pointer to the compiled function as in the CPU target.


class HIPCompileResult(CompileResult):
    @property
    def entry_point(self):
        return id(self)


def hip_compile_result(**entries):
    entries = sanitize_compile_result_entries(entries)
    return HIPCompileResult(**entries)


@register_pass(mutates_CFG=True, analysis_only=False)
class HIPBackend(LoweringPass):

    _name = "hip_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        lowered = state["cr"]
        signature = typing.signature(state.return_type, *state.args)

        state.cr = hip_compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            fndesc=lowered.fndesc,
        )
        return True


@register_pass(mutates_CFG=False, analysis_only=False)
class CreateLibrary(LoweringPass):
    """
    Create a HIPCodeLibrary for the NativeLowering pass to populate. The
    NativeLowering pass will create a code library if none exists, but we need
    to set it up with options from the flags if they are present.
    """

    _name = "create_library"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        codegenerator: codegen.JITHIPCodegen = state.targetctx.codegen()
        name = state.func_id.func_qualname

        options = state.flags.hip_options
        state.library = codegenerator.create_library(name, options=options)
        assert isinstance(state.library, codegen.HIPCodeLibrary)
        # Enable object caching upfront so that the library can be serialized.
        state.library.enable_object_caching()

        return True


class HIPCompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager("hip")

        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = self.define_hip_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_hip_lowering_pipeline(self, state):
        pm = PassManager("cuda_lowering")
        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")

        # lower
        pm.add_pass(CreateLibrary, "create library")
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(HIPBackend, "hip backend")

        pm.finalize()
        return pm


@global_compiler_lock
def compile_hip(
    pyfunc,
    return_type,
    args,
    debug=False,
    lineinfo=False,
    inline=False,
    fastmath=False,
    options=None,
    amdgpu_arch=None,
):
    if amdgpu_arch is None:
        raise ValueError("AMD GPU architecture must be supplied")

    from .descriptor import hip_target

    typingctx = hip_target.typing_context
    targetctx = hip_target.target_context

    flags = HIPFlags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True

    # Both debug and lineinfo turn on debug information in the compiled code,
    # but we keep them separate arguments in case we later want to overload
    # some other behavior on the debug flag. In particular, -opt=3 is not
    # supported with debug enabled, and enabling only lineinfo should not
    # affect the error model.
    if debug or lineinfo:
        flags.debuginfo = True

    if lineinfo:
        flags.dbg_directives_only = True

    if debug:
        flags.error_model = "python"
    else:
        flags.error_model = "numpy"

    if inline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if options:
        flags.hip_options = options
    flags.amdgpu_arch = amdgpu_arch

    # Run compilation pipeline
    from numba.core.target_extension import target_override

    with target_override("hip"):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=HIPCompiler,
        )

    library: codegen.HIPCodeLibrary = cres.library
    library.finalize()

    return cres


@global_compiler_lock
def compile_llvm_ir(
    pyfunc,
    sig,
    debug: bool = False,
    lineinfo: bool = False,
    device: bool = False,
    fastmath: bool = False,
    amdgpu_arch: str = None,
    opt: bool = True,
    to_bc: bool = True,
):
    """Compile a Python function to LLVM IR for a given set of argument types.

    Links in all dependencies and produces a single LLVM IR file.

    Args:
        pyfunc:
            The Python function to compile.
        sig:
            The signature representing the function's input and output types.
        debug (`bool`):
            Whether to include debug info in the generated PTX.
        lineinfo (`bool`):
            Whether to include a line mapping from the generated PTX
            to the source code. Usually this is used with optimized
            code (since debug mode would automatically include this),
            so we want debug info in the LLVM but only the line
            mapping in the final PTX.
        device (`bool`):
            Whether to compile a device function. Defaults to ``False``,
            to compile global kernel functions.
        fastmath (`bool`):
            Whether to enable fast math flags (ftz=1, prec_sqrt=0, prec_div=, and fma=1) TODO HIP enable
        amdgpu_arch (`str`):
            AMD GPU architecture, e.g. ``gfx90a``.
        opt (`bool`):
            Enable optimizations. Defaults to ``True``. TODO HIP enable
        to_bc (`bool`)
            Compile the result to bitcode. Defaults to 'True'.

    :return: (llvm_ir, resty): The LLVM IR code and inferred return type
    :rtype: tuple
    """
    if debug and opt:
        msg = (
            "debug=True with opt=True (the default) "
            "is not supported by HIP. This may result in a crash"
            " - set debug=False or opt=False."
        )
        warn(NumbaInvalidConfigWarning(msg))

    options = {"fastmath": fastmath, "opt": 3 if opt else 0}

    args, return_type = sigutils.normalize_signature(sig)

    # TODO HIP have similar config option
    # cc = cc or config.CUDA_DEFAULT_PTX_CC
    cres = compile_hip(
        pyfunc,
        return_type,
        args,
        debug=debug,
        lineinfo=lineinfo,
        fastmath=fastmath,
        options=options,
        amdgpu_arch=amdgpu_arch,
    )
    resty = cres.signature.return_type

    if resty and not device and resty != types.void:
        raise TypeError("HIP kernel must have void return type.")

    if device:  # device function, __device__
        lib = cres.library
    else:  # kernel, __global__
        tgt: target.HIPTargetContext = cres.target_context
        code = pyfunc.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno

        lib, kernel = tgt.prepare_hip_kernel(
            cres.library,
            cres.fndesc,
            debug,
            lineinfo,
            options,
            filename,
            linenum,
        )
    assert isinstance(lib, codegen.HIPCodeLibrary)
    llvm_ir = lib.get_linked_llvm_ir(amdgpu_arch=amdgpu_arch, to_bc=to_bc)
    return llvm_ir, resty


def compile_llvm_ir_for_current_device(
    pyfunc,
    sig,
    debug=False,
    lineinfo=False,
    device=False,
    fastmath=False,
    opt=True,
    to_bc: bool = True,
):
    """Compile a Python function to AMD GPU LLVM IR for a given set of argument types for
    the current device's compute capabilility. This calls :func:`compile_llvm_ir`
    with an appropriate ``cc`` value for the current device."""
    amdgpu_arch = get_current_device().amdgpu_arch
    return compile_llvm_ir(
        pyfunc,
        sig,
        debug=debug,
        lineinfo=lineinfo,
        device=device,
        fastmath=fastmath,
        amdgpu_arch=amdgpu_arch,
        opt=opt,
        to_bc=to_bc,
    )


# TODO check if this is relevant for HIP as no HSA device functions can be linked, only LLVM IR ones
def declare_device_function(name, restype, argtypes):
    return declare_device_function_template(name, restype, argtypes).key


# TODO check if this is relevant for HIP as no HSA device functions can be linked, only LLVM IR ones
def declare_device_function_template(name, restype, argtypes):
    from .descriptor import hip_target

    typingctx = hip_target.typing_context
    targetctx = hip_target.target_context
    sig = typing.signature(restype, *argtypes)
    extfn = ExternFunction(name, sig)

    class device_function_template(ConcreteTemplate):
        key = extfn
        cases = [sig]

    fndesc = funcdesc.ExternalFunctionDescriptor(
        name=name, restype=restype, argtypes=argtypes
    )
    typingctx.insert_user_function(extfn, device_function_template)
    targetctx.insert_user_function(extfn, fndesc)

    return device_function_template


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig
