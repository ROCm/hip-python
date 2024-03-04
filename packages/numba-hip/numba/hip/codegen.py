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

import os
import re
import textwrap

from llvmlite import ir

from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary

from .hipdrv import devices, driver
from . import amdgcn
from . import hipconfig
from .util import llvmutils, comgrutils
from .typing_lowering import hipdevicelib
from .hipdrv import hiprtc

# TODO replace by AMD COMGR based disasm
# def run_nvdisasm(cubin, flags):

FILE_SEP = "-" * 10 + "(start of next file)" + "-" * 10


def bundle_file_contents(strs):
    filesep = "\n\n" + FILE_SEP + "\n\n"
    return filesep.join(strs)


def unbundle_file_contents(bundled):
    filesep = "\n\n" + FILE_SEP + "\n\n"
    return bundled.split(filesep)


_TYPED_PTR = re.compile(pattern=r"\w+\*+")


def _read_file(filepath: str, mode="r"):
    """Helper routine for reading files in bytes or ASCII mode."""
    with open(filepath, mode) as infile:
        return infile.read()


def _get_amdgpu_arch(amdgpu_arch: str):
    """Helper routine providing default GPU arch if none is specified.

    Uses HIP context's default device if amdgpu_arch is ``None``.
    """
    if amdgpu_arch is None:
        ctx = devices.get_context()
        device: driver.Device = ctx.device
        amdgpu_arch = device.amdgpu_arch
    return amdgpu_arch


class HIPCodeLibrary(serialize.ReduceMixin, CodeLibrary):
    """
    The HIPCodeLibrary generates LLVM IR and AMD GPU code objects
    for multiple different AMD GPU architectures.
    """

    # NOTE: 'ptx' is interpreted as 'll' to ease some porting
    LLVM_IR_EXT = ("ll", "bc", "ptx")

    def __init__(
        self,
        codegen,
        name,
        entry_name=None,
        max_registers=None,
        options=None,
        device: bool = True,
    ):
        """
        codegen:
            Codegen object.
        name:
            Name of the function in the source.
        entry_name:
            Name of the kernel function in the binary, if this is a global
            kernel and not a device function.
        max_registers:
            The maximum register usage to aim for when linking.
        options:
            Dict of options to pass to the compiler/optimizer.
        device (`bool`,optional):
            If the function definition in this module is an
            AMD GPU device function instead of an AMD GPU kernel.
            Defaults to ``True``.
        """
        if max_registers != None:
            raise NotImplementedError(
                "arg 'max_registers' currently not supported due to HIPRTC limitations"
            )

        super().__init__(codegen, name)

        # The llvmlite module for this library.
        # (see: https://github.com/numba/llvmlite/blob/main/llvmlite/ir/module.py)
        self._module: ir.Module = None
        # if this module is an AMD GPU kernel
        self._device: bool = device
        # This list contains entries of the following kind:
        # 1) CodeLibrary objects that will be "linked" into this library. The
        #    modules within them are compiled to LLVM IR along with the
        #    IR from this module - in that sense they are "linked" by LLVM IR
        #    generation time, rather than at link time.
        # 2) LLVM IR/BC or ROCm LLVM Python module types to link with the
        #    generated LLVM IR. These are linked using the Driver API at
        #    link time.
        # 3) Files to link with the generated LLVM IR. These are linked using the
        #    Driver API at link time.
        # NOTE: list maintains insertion order
        self._linking_dependencies = []
        # TODO HIP add TBC HipProgram with user-configurable input as accepted dependency type

        # The raw LLVM IR strs as generated via Numba or
        # added as LLVM IR/BC buffers/files or HIP C++ files to this
        # code library.
        self._raw_source_strs = []
        # Maps GPU arch -> Unlinked AMD GPU LLVM IR snippets (str) per dependency.
        self._unlinked_amdgpu_llvm_strs_cache = {}
        # Maps GPU arch -> Linked AMD GPU LLVM IR (str)
        # A single LLVM file per GPU arch that has been
        # constructed by converting all files to LLVM IR/BC
        # and linking them together into a single LLVM module
        self._linked_amdgpu_llvm_ir_cache = {}
        # Maps GPU arch -> AMD GPU code object
        self._codeobj_cache = {}
        # Maps GPU arch -> linker info output for AMD GPU codeobj
        self._linkerinfo_cache = {}
        # Maps Device numeric ID -> cufunc
        self._hipfunc_cache = {}

        self._max_registers = max_registers
        if options is None:
            options = {}
        self._options = options
        self.init_entry_name(entry_name)

    def init_entry_name(self, entry_name: str):
        """Sets `self._entry_name` and `self._orginal_entry_name` to the passed one."""
        self._entry_name = entry_name
        self._original_entry_name = entry_name

    def change_entry_name(self, new_entry_name: str):
        """Sets `self._entry_name` stores the original one in `self._orginal_entry_name`.

        Requires that `self._entry_name` and `self._original_entry_name` have been set before.
        An assertion fails otherwise.
        """
        assert (
            new_entry_name != None
            and self._entry_name != None
            and self._original_entry_name != None
        )
        self._original_entry_name = self._entry_name
        self._entry_name = new_entry_name

    @staticmethod
    def _walk_linking_dependencies(library, post_order: bool = False):
        """Linearizes the link-time dependency tree via pre- or post-order walk.

        Per default, walks through ``library._linking_dependencies`` in pre-order,
        i.e., a code library is yielded before its
        dependencies. In post-order, this is done the opposite way.

        If a link-time dependency is another code libray, this functions calls
        itself with the dependency as argument while dependencies that are
        LLVM IR/BC files or buffers are yielded directly.

        Note:
            Also yields ``library`` first (pre-order) or last (post-order).

        Args:
            library (`~.HIPCodeLibrary`):
                An instance of `~.HIPCodeLibrary`.
            post_order (`bool`, optional):
                Do the walk in post-order, i.e., all dependencies are yielded
                before ``library``. Defaults to False.
        """
        assert isinstance(library, HIPCodeLibrary)
        if not post_order:
            yield library
        for mod in library._linking_dependencies:
            if isinstance(mod, HIPCodeLibrary):
                yield from HIPCodeLibrary._walk_linking_dependencies(mod)
            elif isinstance(
                mod, (str, tuple)
            ):  # str: filepath, tuple: buffer + buffer len
                yield mod
        if post_order:
            yield library

    def _collect_dependency_llvm_ir(
        self,
        link_time: bool = False,
        amdgpu_arch: str = None,
        force_ir: bool = False,
        remove_duplicates: bool = True,
    ):
        """Collects LLVM IR/BC from all dependencies.

        Note:
            Result may not only contain LLVM IR/BC if a dependency
            is of HIP C++ kind and argument ``link_time==False``.

        Note:
            Simple HIP C++ files that can be compiled via
            HIPRTC and do not require additional includes
            can be specified as dependencies too.
            If compilation of a HIP C++ file is more complicated,
            it is better to use the `numba.hip.hipdrv.hiprtc`
            or `numba.hip.util.comgrutils` APIs directly.

        Note:
            If a dependency is a HIP C++ file and ``link_time==False``,
            it is appended in raw form to the result list.
            If ``link_time==True``, it is first compiled to LLVM IR/BC
            via HIPRTC, which considers the argument ``amdgpu_arch``.

        Args:
            link_time (`bool`,optional):
                If we collect the dependencies at link time and hence
                HIP C++ input files can be compiled to LLVM IR
                and AMD GPU specific modifications can be applied
                to function definitions.
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to ``None``. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.
                Only used if argument ``link_time==True``.
            force_ir (`bool`, optional):
                Converts LLVM BC to human-readable IR (`str`) where required.
                Defaults to ``False``.
            remove_duplicates (`bool`, optional):
                Removes duplicates starting from the end of
                the linearized dependency list.
                Defaults to `True`.
        Returns:
            `list`:
                A string representation (human-readable LLVM IR) of this instance's LLVM module
                and recursively that of all its dependencies.
        """
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)

        # helper routines

        def process_buf_(buf, buf_len: int = -1):
            # note buf_args might be buf and buf_len
            nonlocal amdgpu_arch
            nonlocal force_ir

            if llvmutils.is_human_readable_clang_offload_bundle(buf):
                buf = llvmutils.split_human_readable_clang_offload_bundle(buf)[
                    llvmutils.amdgpu_target_id(amdgpu_arch)
                ]
                buf_len = len(buf)
            if force_ir:
                return llvmutils.to_ir(buf, buf_len).decode("utf-8")
            else:
                try:  # check if the buffer length can be obtained via `len(buf)`
                    len(buf)
                except:  # if not, try to convert to bc buffer
                    if not buf_len or buf_len < 1:
                        raise RuntimeError(
                            f"buffer size cannot be obtained for input {str(buf)}"
                        )
                    return llvmutils.to_bc(buf, buf_len).decode("utf-8")
                else:  # if `len(buf)` works, return `buf`
                    return buf

        def compile_hiprtc_program_(source, name):
            nonlocal amdgpu_arch
            nonlocal process_buf_
            return process_buf_(hiprtc.compile(source, name, amdgpu_arch))

        def add_(llvm_buf, dep_id=None):
            nonlocal unprocessed_result
            nonlocal dependency
            if remove_duplicates:
                unprocessed_result.append(
                    (id(dependency) if dep_id == None else dep_id, llvm_buf)
                )
            else:
                unprocessed_result.append(llvm_buf)

        # pre-order walk
        unprocessed_result = []
        for dependency in HIPCodeLibrary._walk_linking_dependencies(self):
            if isinstance(dependency, HIPCodeLibrary):
                # TODO get unlinked_llvm_ir but with attribute modifications
                if link_time:
                    add_(dependency.get_unlinked_llvm_ir(amdgpu_arch))
                else:
                    add_(str(dependency._module))
            elif isinstance(dependency, str):  # an LLVM IR/BC file
                fileext = os.path.basename(dependency).split(os.path.extsep)[-1]
                if fileext in self.LLVM_IR_EXT:  # 'ptx' is interpreted as 'll'.
                    mode = "rb" if fileext == "bc" else "r"
                    add_(process_buf_(_read_file(dependency, mode)), dep_id=dependency)
                elif link_time:  # assumes HIP C++
                    add_(
                        compile_hiprtc_program_(
                            _read_file(dependency, "r"), name=dependency
                        ),
                        dep_id=dependency,
                    )
                else:  # assumes HIP C++
                    add_(_read_file(dependency, "r"), dep_id=dependency)
            elif isinstance(dependency, tuple):  # an LLVM IR/BC buffer
                if len(dependency) == 2:  # always assume LLVM IR/BC
                    add_(process_buf_(*dependency))
                elif len(dependency) == 3:
                    fileext = dependency[2]
                    if fileext in self.LLVM_IR_EXT:
                        add_(process_buf_(*dependency[:2]))
                    else:  # assumes HIP C++
                        add_(
                            compile_hiprtc_program_(dependency[0], name=dependency),
                        )
                else:
                    raise RuntimeError(
                        "tuple dependency specifications must have length 2 '(buffer, buffer-len or None)' or 3 '(buffer, buffer-len or None, file-ext)'"
                    )
            else:
                raise RuntimeError(
                    f"don't know how to handle dependency specification of type '{type(dependency)}'"
                )

        if remove_duplicates:
            # Example: 'A -> [B, C->B]' linearized to '[A, B, C, B]' => '[A, C, B]'
            result = []
            dep_ids = set()
            for dep_id, llvm_buf in reversed(unprocessed_result):
                if dep_id not in dep_ids:
                    result.append(llvm_buf)
                dep_ids.add(dep_id)
            return list(reversed(result))
        else:
            return unprocessed_result

    def get_raw_source_strs(self):
        """Return raw LLVM IR or HIP C++ sources of this module and its dependencies.

        The first entry contains the LLVM IR for this HIPCodeLibrary's module.
        """
        if self._raw_source_strs:
            return self._raw_source_strs
        return self._collect_dependency_llvm_ir(link_time=False, force_ir=True)

    def get_unlinked_llvm_strs(self, amdgpu_arch):
        """Return unlinked AMD GPU LLVM IR of this module and all its dependencies.

        The first entry contains the LLVM IR for this HIPCodeLibrary's module.

        Returns:
            `list`:
                Contains unlinked AMD GPU LLVM IR of this module and its dependencies.
        """
        unlinked_llvm = self._unlinked_amdgpu_llvm_strs_cache.get(amdgpu_arch, None)
        if unlinked_llvm:
            return unlinked_llvm
        else:
            # NOTE:
            #     We need to set `force_ir` because we
            #     need to apply preprocessing to the generated IR.
            unlinked_llvm = self._collect_dependency_llvm_ir(
                link_time=True, force_ir=True
            )
            self._unlinked_amdgpu_llvm_strs_cache[amdgpu_arch, None] = unlinked_llvm
            return unlinked_llvm

    def get_raw_source_str(self):
        """Bundles the string representation of this instance's LLVM module and that of its dependencies.

        Returns:
            `str`:
                The joined string representation of this instance's LLVM module
                and that of all its dependencies (recursively). Dependencies can
                be other HIPCodeLibrary instances, LLVM IR/BC buffers/files, and
                simple HIP C++ buffers/files

        See:
            `HIPCodeLibrary.get_raw_source_strs`
        """
        return bundle_file_contents(self.get_raw_source_strs)

    # @abstractmethod (5/6), added arch amdgpu_arch
    def get_llvm_str(self, amdgpu_arch: str = None, linked: bool = False):
        """Get linked/unlinked LLVM representation of this HIPCodeLibrary.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.
            linked (`bool`, optional`):
                Return the string representation of the fully linked LLVM IR, where
                all dependencies including the Numba HIP device library have been linked in.
                This file can be quite large (10k+ lines of code).
                Otherwise, bundles the string representation of this instance's LLVM module
                and that of its dependencies.
                Defaults to ``True``.
        Returns:
            `str`:
                The joined string representation of this instance's LLVM module and that of all its dependencies (recursively).
        See:
            `HIPCodeLibrary.get_raw_source_strs`
        """
        if linked:
            return llvmutils.to_ir(self.get_linked_llvm_ir(amdgpu_arch))
        else:
            return bundle_file_contents(self.get_unlinked_llvm_strs(amdgpu_arch))

    # @abstractmethod (5/6), added arch amdgpu_arch
    def get_asm_str(self, amdgpu_arch: str):
        """(Currently not implemented.)

        Return a disassembly of the AMD code object.
        Requires that this functionality is added to
        ROCm AMD COMGR.

        amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
            Defaults to None. If ``None`` is specified, the architecture of the first device
            in the current HIP context is used instead.
        """
        raise NotImplementedError()

    @property
    def linking_libraries(self):
        """Recursively create a list of link-time dependencies.

        Libraries we link to may link to other libraries, so we recursively
        traverse the linking libraries property to build up a list of all
        linked libraries.
        """
        return list(
            mod
            for mod in HIPCodeLibrary._walk_linking_dependencies(self)
            if isinstance(mod, HIPCodeLibrary)
        )

    @property
    def modules(self):
        """Get this instance's llvmlite module and recursively that of all its dependencies
        Returns:
            `list`:
                A list of LLVM IR modules, recursively created from this instance's
                ``_module`` member and the `HIPCodeLibrary` instances in ``self._linking_libraries``.
        """
        return list(
            dependency._module
            for dependency in HIPCodeLibrary._walk_linking_dependencies(self)
            if isinstance(dependency, HIPCodeLibrary)
        )

    def _apply_llvm_amdgpu_modifications(self, amdgpu_arch: str = None):
        """Applies modifications for LLVM AMD GPU device functions.

        Modifies visibility, calling convention and function attributes.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.
                This argument is required if device-code only HIP C++ files are encountered
                that need to compiled to LLVM IR first via HIPRTC.

        Note:
            Directly and persistently modifies member ``self._module``.
        """
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)

        if self._device:
            fun_linkage = comgrutils.llvm_amdgpu_device_fun_visibility
            fun_call_conv = ""
            fun_attributes = comgrutils.get_llvm_device_fun_attributes(
                amdgpu_arch, only_kv=True, raw=True
            )
        else:
            fun_linkage = comgrutils.llvm_amdgpu_kernel_visibility
            fun_call_conv = comgrutils.llvm_amdgpu_kernel_calling_convention
            fun_attributes = comgrutils.get_llvm_kernel_attributes(
                amdgpu_arch, only_kv=True, raw=True
            )
        self._module.data_layout = amdgcn.AMDGPUTargetMachine(amdgpu_arch).data_layout
        for fn in self._module.functions:
            assert isinstance(fn, ir.Function)
            if not fn.is_declaration:
                if fn.name == self._original_entry_name:
                    # NOTE setting f.name here has no effect, as the name might be cached
                    #      Hence, we overwrite it directly in LLVM IR at the
                    #      get_unliked_llvm_ir step.
                    # NOTE: We assume there is only one definition in the
                    # use `fn.linkage` field to specify visibility
                    fn.linkage = fun_linkage
                    fn.calling_convention = fun_call_conv
                    # Abuse attributes to specify address significance
                    # set.add(fn.attributes, "local_unnamed_addr") # TODO HIP disabled for now, causes error
                    for attrib in fun_attributes:
                        # We bypass the known-attribute check performed by ir.FunctionAttributes
                        # by calling the `add` method of the super class `set`
                        # (`ir.FunctionAttributes`->`ir.FunctionAttributes`->`set`)
                        set.add(fn.attributes, attrib)

    def _postprocess_llvm_ir(self, llvm_str: str):
        """Postprocess Numba and third-party LLVM assembly.

        Note:
            Numba might be using an llvmlite package
            that is based on an older LLVM release, which means, e.g., that
            Numba-generated LLVM assembly contains typed pointers such as ``i8*``,
            ``double**``, ... This preprocessing routine converts these to opaque
            pointers (``ptr``), which is the way more recent LLVM releases model pointers.
            More details: https://llvm.org/docs/OpaquePointers.html#version-support

            Further replaces ``sext ptr to null to i<bits>`` with
            ``ptrtoint ptr null to i<bits>`` as ``sext`` only accepts
            integer types now.
            https://llvm.org/docs/LangRef.html#sext-to-instruction
            More details: https://llvm.org/docs/LangRef.html#i-ptrtoint
        """
        global _TYPED_PTR
        if self._entry_name != None:
            assert self._original_entry_name != None
            llvm_str = llvm_str.replace(self._original_entry_name, self._entry_name)
        llvm_str = _TYPED_PTR.sub(string=llvm_str, repl="ptr")
        return llvm_str.replace("sext ptr null to i", "ptrtoint ptr null to i")

    def get_unlinked_llvm_ir(
        self,
        amdgpu_arch: str = None,
    ):
        """Returns the unlinked AMDGPU LLVM IR of this HIPCodeLibrary's module in human-readable format.

        Note:
            Applies architecture specific modifications to the function
            definitions in the `llvmlite.ir.Module` member `self._module`.

        Note:
            Must not be confused with `get_unlinked_llvm_strs` which
            returns LLVM IR for this module and all its dependencies.
        Note:
            Call `self._apply_llvm_amdgpu_modifications` modifies ``self._module``.
            ``self._preprocess_llvm_ir(llvm_ir)`` applies modifications
            that can only/most easily be applied to the LLVM IR in the text representation.
        """
        self._apply_llvm_amdgpu_modifications(amdgpu_arch)
        llvm_ir = self._postprocess_llvm_ir(str(self._module))
        return llvm_ir

    def _postprocess_unlinked_llvm_strs(self, llvm_strs):
        """Preprocess Numba *and* third-party LLVM assembly.

        See:
            _postprocess_llvm_ir
        """
        if config.DUMP_LLVM:
            print(
                (
                    "AMD GPU LLVM for pyfunc '%s' (unlinked inputs, unpreprocessed)"
                    % self._entry_name
                ).center(80, "-")
            )
            print(bundle_file_contents(llvm_strs))
            print("=" * 80)

        for i in range(len(llvm_strs)):
            llvm_strs[i] = self._postprocess_llvm_ir(llvm_strs[i])

        if config.DUMP_LLVM:
            print(
                (
                    "AMD GPU LLVM for pyfunc '%s' (unlinked inputs, preprocessed)"
                    % self._entry_name
                ).center(80, "-")
            )
            print(bundle_file_contents(llvm_strs))
            print("=" * 80)

    def get_linked_llvm_ir(
        self,
        amdgpu_arch: str = None,
        to_bc: bool = True,
    ):
        """Returns/Creates single module from linking in all link-time dependencies.

        Note:
            Always compiles to an AMD GPU device function, never to an AMD GPU kernel.
            The result may be transformed to a kernel in `numba.hip.compiler`.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.
                This argument is required if device-code only HIP C++ files are encountered
                that need to compiled to LLVM IR first via HIPRTC.
            to_bc (`bool`, optional):
                If the result should be LLVM bitcode instead of human-readable LLVM IR.
                Defaults to ``True``.
            remove_unused_helpers (`bool`, optional)
                Remove unused helper functions. Can reduce generated IR by a factor
                of approx. 5x but has small performance impact at codegen time.
                Defaults to ``True``.
        Returns:
            `bytes`:
                The result of the linking as LLVM bitcode or human-readable LLVM IR depending on argument ``to_bc``.
        """
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)
        linked_llvm = self._linked_amdgpu_llvm_ir_cache.get(amdgpu_arch, None)
        if linked_llvm:
            return linked_llvm

        # 1. link all dependencies except the hip device lib
        # - add self._module + dependencies
        # - applies AMD GPU specific modifications to all function definitions.
        linker_inputs = self.get_unlinked_llvm_strs(
            amdgpu_arch=amdgpu_arch,
        )
        self._postprocess_unlinked_llvm_strs(linker_inputs)
        linked_llvm = llvmutils.link_modules(linker_inputs, to_bc=True)
        # 2 identify numba hip helper functions
        used_hipdevicelib_declares = llvmutils.get_function_names(
            linked_llvm, defines=False
        )
        # 2. apply mid-end optimizations if requested
        if hipconfig.ENABLE_MIDEND_OPT and self._options.get("opt", False):

            linked_llvm = amdgcn.AMDGPUTargetMachine(amdgpu_arch).optimize_module(
                linked_llvm
            )
            if config.DUMP_LLVM:
                print(
                    (
                        "AMD GPU LLVM for pyfunc '%s' (mid-end optimizations)"
                        % self._entry_name
                    ).center(80, "-")
                )
                print(llvmutils.to_ir(linked_llvm).decode("utf-8"))
                print("=" * 80)

        # 3. now link the hip device lib
        linked_llvm = llvmutils.link_modules(
            [linked_llvm, hipdevicelib.get_llvm_bc(amdgpu_arch)], to_bc
        )

        # 4. remove unused hip device lib function definitions
        def delete_matcher_(name: str):
            nonlocal used_hipdevicelib_declares
            if hipdevicelib.DEVICE_FUN_PREFIX in name:
                return not name in used_hipdevicelib_declares
            return False

        linked_llvm = llvmutils.delete_functions(
            linked_llvm, matcher=delete_matcher_, defines=True
        )
        # 4. link a last time to get rid of unused third-order function declarations
        linked_llvm = llvmutils.link_modules([linked_llvm], to_bc)
        # lastly add the HIP device lib
        self._linked_amdgpu_llvm_ir_cache[amdgpu_arch] = linked_llvm
        return linked_llvm

    def get_codeobj(self, amdgpu_arch=None):
        """Returns/compiles a code object for the specified AMD GPU architecture.

        Performs the following steps:

        1. If there is already a code object in the cache for 'amdgpu_arch', the function returns it.
        2. If there there no code object, the driver's linker is used
           to build it.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.

        Note:
            We use HIPRTC as linker here instead of ROCm LLVM link module APIs.
        Returns:
            The code object buffer.
        """
        if self._device:
            raise NotImplementedError(
                textwrap.dedent(
                    """\
                    Compiling device functions to AMD GPU code is currently not supported.
                    Device functions can currently only be compiled to LLVM IR.
                    """
                )
            )
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)

        codeobj = self._codeobj_cache.get(amdgpu_arch, None)
        if codeobj:
            return codeobj

        linker = driver.Linker.new(
            max_registers=self._max_registers, amdgpu_arch=amdgpu_arch
        )
        linker.add_llvm_ir(self.get_linked_llvm_ir(amdgpu_arch, to_bc=True))
        codeobj = linker.complete()
        # for inspecting the code object
        # import rocm.amd_comgr.amd_comgr as comgr
        # import pprint
        # pprint.pprint(list(comgr.ext.parse_code_symbols(codeobj,len(codeobj)).keys()))
        self._codeobj_cache[amdgpu_arch] = codeobj
        self._linkerinfo_cache[amdgpu_arch] = linker.info_log
        return codeobj

    def get_hipfunc(self):
        if self._device:
            msg = (
                "Missing entry_name - are you trying to get the hipfunc "
                "for a device function?"
            )
            raise RuntimeError(msg)

        ctx: driver.Context = devices.get_context()
        device: driver.Device = ctx.device

        hipfunc = self._hipfunc_cache.get(device.id, None)
        if hipfunc:
            return hipfunc

        codeobj = self.get_codeobj(amdgpu_arch=device.amdgpu_arch)
        module = ctx.create_module_image(codeobj)

        # Load
        hipfunc = module.get_function(self._entry_name)

        # Populate caches
        self._hipfunc_cache[device.id] = hipfunc

        return hipfunc

    def get_linkerinfo(self, cc):
        try:
            return self._linkerinfo_cache[cc]
        except KeyError:
            raise KeyError(f"No linkerinfo for CC {cc}")

    # @abstractmethod (1/6)
    def add_ir_module(self, mod):
        """Set the Numba-generated llvmlite IR module.

        Note:
            This routine can only be used once. Otherwise,
            an exception is raised.
        """
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError("HIPCodeLibrary only supports one module")
        self._module = mod

    # @abstractmethod (2/6)
    def add_linking_library(self, library):
        """Add another `~.HIPCodeLibrary` library as link-time dependency.

        Args:
            library (`~.HIPCodeLibrary`):
                Another `~.HIPCodeLibrary` to add as link-time dependency.
                Must be finalized, otherwise an exception is raised.
        Note:
            Libraries can only be added if this instance's linking has not
            been finalized yet. We don't want to allow linking more libraries
            in after finalization because our linked libraries are modified by
            the finalization, and we won't be able to finalize again after
            adding new ones.
        """
        assert isinstance(library, HIPCodeLibrary)
        # print(f"{self.name} add library: {library._module.name}")
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        self._linking_dependencies.append(library)

    def add_linking_ir(self, mod, mod_len: int = -1):
        """Add LLVM IR/BC buffers or ROCm LLVM Python module types as link-time dependency.

        Args:
            mod (UTF-8 `str`, or implementor of the Python buffer protocol such as `bytes`, or `rocm.llvm.c.types.LLVMOpaqueModule`):
                Either a buffer that contains LLVM IR or LLVM BC or an instance of `rocm.llvm.c.types.LLVMOpaqueModule`.
            mod_len (`int`, optional):
                Length of the LLVM IR/BC buffer. Must be supplied if it cannot
                be obtained via ``len(mod)``. Not used at all if ``mod`` is an instance of
                `rocm.llvm.c.types.LLVMOpaqueModule`.
        Note:
            Only append an IR linking dependency once.
            This is an API that must be used explicitly
            by users. The users must ensure correct order
            of inputs.
        """
        self._linking_dependencies.append((mod, mod_len))

    def add_linking_file(self, filepath: str):
        """Add files in formats such as HIP C++ or LLVM IR/BC as link-time dependency.

        Note:
            Simple HIP C++ files that can be compiled via HIPRTC and do not
            require additional includes or compiler arguments can be
            specified as dependencies too.

            If compilation of a HIP C++ file is more complicated,
            it is better to use the `numba.hip.hipdrv.hiprtc`
            or `numba.hip.util.comgrutils` APIs directly to
            create an LLVM IR/BC buffer that can then be specified
            as dependency to this `HIPCodeLibrary` via `HIPCodeLibrary.add_linking_ir`
            or `HIPCodeLibrary.add_linking_dependency`.

        Note:
            Only append an IR linking dependency once.
            This is an API that must be used explicitly
            by users. The users must ensure correct order
            of inputs.
        """
        self._linking_dependencies.append(filepath)

    def add_linking_dependency(self, dependency):
        """Adds linking dependency in one of the supported formats.

        Args:
            dependency (`object`):
                1. HIPCodeLibrary objects that will be "linked" into this library. The
                   modules within them are compiled to LLVM IR along with the
                   IR from this module - in that sense they are "linked" by LLVM IR
                   generation time, rather than at link time.
                   See `add_linking_library` for more details.
                2. LLVM IR/BC or ROCm LLVM Python module types to link with the
                   generated LLVM IR. These are linked using the Driver API at
                   link time. See `add_linking_ir` for more details.
                3. Files to link with the generated LLVM IR. These are linked using the
                   Driver API at link time. See `HIPCodeLibrary.add_linking_file` for more details.
        """
        if isinstance(dependency, HIPCodeLibrary):
            self._raise_if_finalized()
        elif isinstance(dependency, str):  # this is a filepath
            pass
        elif isinstance(dependency, tuple):  # this is a
            if len(dependency) not in (2, 3):
                raise TypeError("expected tuple of length 2 or 3")
        else:
            raise TypeError(f"unexpected input of type {type(dependency)}")
        self._linking_dependencies.append(dependency)

    # @abstractmethod (4/6)
    def get_function(self, name):
        """Retrieves an LLVM function from this libraries' llvmlite module."""
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f"Function {name} not found")

    # @abstractmethod (3/6)
    def finalize(self):
        # Unlike the CPUCodeLibrary, we don't invoke the binding layer here -
        # we only adjust the linkage of functions. Global kernels (with
        # external linkage) have their linkage untouched. Device functions are
        # set linkonce_odr to prevent them appearing in the AMD GPU code object.

        self._raise_if_finalized()

        # Note in-place modification of the linkage of functions in linked
        # libraries. This presently causes no issues as only device functions
        # are shared across code libraries, so they would always need their
        # linkage set to linkonce_odr. If in a future scenario some code
        # libraries require linkonce_odr linkage of functions in linked
        # modules, and another code library requires another linkage, each code
        # library will need to take its own private copy of its linked modules.
        #
        # See also discussion on PR #890:
        # https://github.com/numba/numba/pull/890
        # for dependency in HIPCodeLibrary._walk_linking_dependencies(self):
        #     if isinstance(dependency, HIPCodeLibrary):
        #         for fn in dependency._module.functions:
        #             if not fn.is_declaration:
        #                 fn.linkage = "linkonce_odr"  # TODO check if this is required
        #                 fn.unnamed_addr = True

        # TODO original Numba CUDA code; kept (a while) for reference
        # for library in self._linking_libraries:
        #    for mod in library.modules:
        #        for fn in mod.functions:
        #            if not fn.is_declaration:
        #                fn.linkage = "linkonce_odr"
        #
        self._finalized = True

    def _reduce_states(self):
        """
        Reduce the instance for serialization. We retain the LLVM IR and AMD GPU code objects,
        but loaded functions are discarded. They are recreated when needed
        after deserialization.

        Note:
            LLVM buffers and LLVM input files are
        """
        non_llvm_linking_files = [
            dependency
            for dependency in HIPCodeLibrary._walk_linking_dependencies(self)
            if (
                isinstance(dependency, str)
                and os.path.basename(dependency).split(os.path.extsep)[-1]
                not in self.LLVM_IR_EXT
            )
            or (
                isinstance(dependency, tuple)
                and len(dependency) == 3
                and dependency[2] not in self.LLVM_IR_EXT
            )
        ]
        if any(
            non_llvm_linking_files
        ):  # TODO HIP understand why files are not supported
            msg = "Cannot pickle HIPCodeLibrary with linking files and buffers"
            raise RuntimeError(msg)
        if not self._finalized:
            raise RuntimeError("Cannot pickle unfinalized HIPCodeLibrary")
        return dict(
            codegen=None,
            name=self.name,
            entry_name=self._entry_name,
            original_entry_name=self._original_entry_name,
            raw_source_strs=self._raw_source_strs,
            unlinked_llvm_strs_cache=self._unlinked_amdgpu_llvm_strs_cache,
            linked_llvm_ir_cache=self._linked_amdgpu_llvm_ir_cache,
            codeobj_cache=self._codeobj_cache,
            linkerinfo_cache=self._linkerinfo_cache,
            max_registers=self._max_registers,
            options=self._options,
        )

    @classmethod
    def _rebuild(
        cls,
        codegen,
        name,
        entry_name,
        original_entry_name,
        raw_source_strs,
        unlinked_llvm_strs_cache,
        linked_llvm_ir_cache,
        codeobj_cache,
        linkerinfo_cache,
        max_registers,
        options,
    ):
        """
        Rebuild an instance from the a cached reduced state.
        """
        instance = cls(codegen, name, entry_name=entry_name)

        instance._original_entry_name = original_entry_name
        instance._raw_source_strs = raw_source_strs
        instance._unlinked_amdgpu_llvm_strs_cache = unlinked_llvm_strs_cache
        instance._linked_amdgpu_llvm_ir_cache = linked_llvm_ir_cache
        instance._codeobj_cache = codeobj_cache
        instance._linkerinfo_cache = linkerinfo_cache

        instance._max_registers = max_registers
        instance._options = options

        instance._finalized = True

        return instance


class JITHIPCodegen(Codegen):
    """
    This codegen implementation for HIP only generates optimized LLVM IR.
    Generation of AMD GPU code objects is done separately (see numba.hip.compiler).

    Note:
        Calls like `inst.create_library(name,<kwargs>)` on an instance
        of this object will result in `JITHIPCodegen._library_class(inst, name, <kwargs>)`,
        i.e., the creation of a `HIPCodeLibrary` object.
    """

    _library_class = HIPCodeLibrary

    def __init__(self, module_name):
        pass

    def _create_empty_module(self, name):
        ir_module = ir.Module(name)
        ir_module.triple = amdgcn.TRIPLE
        ir_module.data_layout = amdgcn.DATA_LAYOUT
        return ir_module

    def _add_module(self, module):
        pass

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.

        Required to compute `numba.core.caching.Cache` index key.
        """
        ctx = devices.get_context()
        # cc = ctx.device.compute_capability
        device: driver.Device = ctx.device
        return (driver.get_version(), device.amdgpu_arch)
