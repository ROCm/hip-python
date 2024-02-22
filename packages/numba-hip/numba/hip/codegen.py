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

from llvmlite import ir

from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .hipdrv import devices, driver

from .amdgcn import TRIPLE as HIP_TRIPLE
from .amdgcn import DATA_LAYOUT
from numba.hip.util import llvmutils
from numba.hip.typing_lowering import hipdevicelib
from numba.hip.hipdrv import hiprtc

# TODO replace by AMD COMGR based disasm
# def run_nvdisasm(cubin, flags):


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

    def __init__(
        self, codegen, name, entry_name=None, max_registers=None, options=None
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
        """
        if max_registers != None:
            raise NotImplementedError(
                "arg 'max_registers' currently not supported due to HIPRTC limitations"
            )

        super().__init__(codegen, name)

        # The llvmlite module for this library.
        self._module = None
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

        # Cache the LLVM IR string representation of this
        # HIPCodeLibrary's and its dependencies' `_module`` member
        # as well as of all LLVM IR/BC buffers.
        # TODO LLVM IR is technically already target dependent.
        self._llvm_strs = None

        # Maps GPU arch -> Linked AMD GPU LLVM IR (str)
        # A single LLVM file per GPU arch that has been
        # constructed by converting all files to LLVM IR/BC
        # and linking them together into a single LLVM module
        self._linked_llvm_cache = {}
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
        self._entry_name = entry_name

    @property
    def llvm_strs(self):
        """Get this instance's LLVM module as string and recursively that of all its dependencies.

        Note:
            Link-time file dependencies are not considered.

        Returns:
            `list`:
                A string representation (human-readable LLVM IR) of this instance's LLVM module
                and recursively that of all its dependencies.
        """
        return self._get_llvm_strs()

    def _get_llvm_strs(self, compile_hip_files: bool = False, amdgpu_arch: str = None):
        """Get this instance's LLVM module as string and recursively that of all its dependencies.

        Note:
            Link-time file dependencies are not considered.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.

        Returns:
            `list`:
                A string representation (human-readable LLVM IR) of this instance's LLVM module
                and recursively that of all its dependencies.
        """
        if self._llvm_strs is None:
            self._llvm_strs = []
            for dependency in HIPCodeLibrary._walk_linking_dependencies(self):
                if isinstance(dependency, HIPCodeLibrary):
                    self._llvm_strs.append(str(dependency._module))
                elif isinstance(dependency, str):  # an LLVM IR/BC file
                    fileext = os.path.basename(dependency).split(os.path.extsep)[-1]
                    if fileext in ("ll", "bc"):
                        mode = "rb" if fileext == "bc" else "r"
                        self._llvm_strs.append(
                            llvmutils.to_ir(_read_file(dependency, mode)).decode(
                                "utf-8"
                            )
                        )
                    elif compile_hip_files:  # assumes HIP C++
                        prog = hiprtc.HIPRTC().create_program(
                            _read_file(dependency, "r"), name=dependency
                        )
                        compile_opts = (
                            f"--offload-arch={_get_amdgpu_arch(amdgpu_arch)} -fgpu-rdc"
                        )
                        hiprtc.HIPRTC().compile_program(prog, compile_opts)
                elif isinstance(dependency, tuple):  # an LLVM IR/BC buffer
                    self._llvm_strs.append(llvmutils.to_ir(*dependency).decode("utf-8"))
        return self._llvm_strs

    def _join_strs(self, strs):
        return "\n\n".join(strs)

    # @abstractmethod (5/6)
    def get_llvm_str(self):
        """Joins the string representation of this instance's LLVM module and that of all its dependencies (recursively)

        Note:
            Link-time file dependencies are not considered.

        Returns:
            `str`:
                The joined string representation of this instance's LLVM module and that of all its dependencies (recursively).
        """
        return "\n\n".join(self.llvm_strs)

    # @abstractmethod (5/6)
    def get_asm_str(self):
        """Same as get_llvm_str()

        Note:
            Currently, simply returns `self.get_llvm_str()`.
        """
        return self.get_llvm_str()

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

    def get_linked_llvm_ir(self, amdgpu_arch: str = None, to_bc: bool = True):
        """Returns/Creates single module from linking in all link-time dependencies.

        Args:
            amdgpu_arch (`str`, optional): AMD GPU architecture string such as `gfx90a`.
                Defaults to None. If ``None`` is specified, the architecture of the first device
                in the current HIP context is used instead.
                This argument is required if device-code only HIP C++ files are encountered
                that need to compiled to LLVM IR first via HIPRTC.
            to_bc (`bool`, optional):
                If the result should be LLVM bitcode instead of human-readable LLVM IR.
                Defaults to `True`.

        Returns:
            `bytes`:
                The result of the linking as LLVM bitcode or human-readable LLVM IR depending on argument ``to_bc``.
        """
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)
        linked_llvm = self._linked_llvm_cache.get(amdgpu_arch, None)
        if linked_llvm:
            return linked_llvm

        # add module + dependencies
        linker_inputs = self._get_llvm_strs(compile_hip_files=True, amdgpu_arch=amdgpu_arch)
        # lastly add the HIP device lib
        linker_inputs.append(hipdevicelib.get_llvm_bc(amdgpu_arch))
        # link and cache result
        linked_llvm = llvmutils.link_modules(linker_inputs,to_bc)
        self._linked_llvm_cache[amdgpu_arch] = linked_llvm
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

        Returns:
            The code object buffer.
        """
        amdgpu_arch = _get_amdgpu_arch(amdgpu_arch)

        codeobj = self._codeobj_cache.get(amdgpu_arch, None)
        if codeobj:
            return codeobj

        linker = driver.Linker.new(
            max_registers=self._max_registers, amdgpu_arch=amdgpu_arch
        )

        # self._module is also a linking dependency.
        for dependency in HIPCodeLibrary._walk_linking_dependencies(self):
            if isinstance(dependency, HIPCodeLibrary):
                linker.add_llvm_ir(str(dependency._module))
            elif isinstance(dependency, str):  # this is a filepath
                linker.add_file_guess_ext(dependency)
            elif isinstance(dependency, tuple):  # this is a
                linker.add_llvm_ir(llvmutils.to_bc(*dependency))
        # lastly link the HIP device lib
        linker.add_llvm_ir(hipdevicelib.get_llvm_bc(amdgpu_arch))

        codeobj = linker.complete()
        self._codeobj_cache[amdgpu_arch] = codeobj
        self._linkerinfo_cache[amdgpu_arch] = linker.info_log
        return codeobj

    def get_hipfunc(self):
        if self._entry_name is None:
            msg = (
                "Missing entry_name - are you trying to get the cufunc "
                "for a device function?"
            )
            raise RuntimeError(msg)

        ctx = devices.get_context()
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
        """
        self._linking_dependencies.append((mod, mod_len))

    def add_linking_file(self, filepath: str):
        """Add files in formats such as HIP C++ or LLVM IR/BC as link-time dependency."""
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
                   Driver API at link time. See `add_linking_file` for more details.
        """
        if isinstance(dependency, HIPCodeLibrary):
            self._raise_if_finalized()
        elif isinstance(dependency, str):  # this is a filepath
            pass
        elif isinstance(dependency, tuple):  # this is a
            if not len(dependency) == 2:
                raise TypeError("expected tuple of length 2")
        else:
            raise TypeError("expected tuple of length 2")
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
            if isinstance(dependency, str)
            and os.path.basename(dependency).split(os.path.extsep)[-1]
            not in ("ll", "bc")
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
            llvm_strs=self.llvm_strs,
            linked_llvm_cache=self._linked_llvm_cache,
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
        llvm_strs,
        linked_llvm_cache,
        codeobj_cache,
        linkerinfo_cache,
        max_registers,
        options,
    ):
        """
        Rebuild an instance from the a cached reduced state.
        """
        instance = cls(codegen, name, entry_name=entry_name)

        instance._llvm_strs = llvm_strs
        instance._linked_llvm_cache = linked_llvm_cache
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
        ir_module.triple = HIP_TRIPLE
        ir_module.data_layout = DATA_LAYOUT
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
