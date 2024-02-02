from llvmlite import ir

from numba.core import serialize
from numba.core.codegen import Codegen, CodeLibrary
from .hipdrv import devices, driver

from .amdgputargetmachine import TRIPLE as HIP_TRIPLE
from .amdgputargetmachine import DATA_LAYOUT

# TODO replace by AMD COMGR based disasm
# def run_nvdisasm(cubin, flags):


class HIPCodeLibrary(serialize.ReduceMixin, CodeLibrary):
    """
    The HIPCodeLibrary generates LLVM IR for multiple different
    compute capabilities.
    """

    def __init__(self, codegen, name, entry_name=None, max_registers=None):
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
        """
        if max_registers != None:
            raise NotImplementedError(
                "arg 'max_registers' currently not supported due to HIPRTC limitations"
            )

        super().__init__(codegen, name)

        # The llvmlite module for this library.
        self._module = None
        # CodeLibrary objects that will be "linked" into this library. The
        # modules within them are compiled from NVVM IR to LLVM IR along with the
        # IR from this module - in that sense they are "linked" by NVVM at LLVM IR
        # generation time, rather than at link time.
        self._linking_libraries = set()
        # Files to link with the generated LLVM IR. These are linked using the
        # Driver API at link time.
        self._linking_files = set()
        # Should we link libcudadevrt?
        self.needs_cudadevrt = False

        # Cache the LLVM IR string
        self._llvm_strs = None
        # Maps GPU arch -> LLVM IR string
        self._llvm_bc_cache = {}
        # Maps GPU arch -> AMD GPU code object
        self._codeobj_cache = {}
        # Maps GPU arch -> linker info output for AMD GPU codeobj
        self._linkerinfo_cache = {}
        # Maps Device numeric ID -> cufunc
        self._cufunc_cache = {}

        self._max_registers = max_registers
        self._entry_name = entry_name

    @property
    def llvm_strs(self):
        if self._llvm_strs is None:
            self._llvm_strs = [str(mod) for mod in self.modules]
        return self._llvm_strs

    def get_llvm_str(self):
        return "\n\n".join(self.llvm_strs)

    def get_asm_str(self, cc=None):
        return self._join_code_objects(self._get_ptxes(cc=cc))

    def get_codeobj(self, amdgpu_arch=None):
        if amdgpu_arch is None:
            ctx = devices.get_context()
            device: driver.Device = ctx.device
            amdgpu_arch = device.arch

        codeobj = self._codeobj_cache.get(amdgpu_arch, None)
        if codeobj:
            return codeobj

        linker = driver.Linker.new(max_registers=self._max_registers, cc=amdgpu_arch)

        ptxes = self._get_llvm_bc(cc=amdgpu_arch)
        for ptx in ptxes:
            linker.add_llvm_ir(ptx.encode())
        for path in self._linking_files:
            linker.add_file_guess_ext(path)
        if self.needs_cudadevrt:
            linker.add_file_guess_ext(get_cudalib("cudadevrt", static=True))

        codeobj = linker.complete()
        self._codeobj_cache[amdgpu_arch] = codeobj
        self._linkerinfo_cache[amdgpu_arch] = linker.info_log

        return codeobj

    def get_cufunc(self):
        if self._entry_name is None:
            msg = (
                "Missing entry_name - are you trying to get the cufunc "
                "for a device function?"
            )
            raise RuntimeError(msg)

        ctx = devices.get_context()
        device = ctx.device

        cufunc = self._cufunc_cache.get(device.id, None)
        if cufunc:
            return cufunc

        cubin = self.get_codeobj(amdgpu_arch=device.compute_capability)
        module = ctx.create_module_image(cubin)

        # Load
        cufunc = module.get_function(self._entry_name)

        # Populate caches
        self._cufunc_cache[device.id] = cufunc

        return cufunc

    def get_linkerinfo(self, cc):
        try:
            return self._linkerinfo_cache[cc]
        except KeyError:
            raise KeyError(f"No linkerinfo for CC {cc}")

    def add_ir_module(self, mod):
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError("HIPCodeLibrary only supports one module")
        self._module = mod

    def add_linking_library(self, library):
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        self._linking_libraries.add(library)

    def add_linking_file(self, filepath):
        self._linking_files.add(filepath)

    def get_function(self, name):
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f"Function {name} not found")

    @property
    def modules(self):
        return [self._module] + [
            mod for lib in self._linking_libraries for mod in lib.modules
        ]

    @property
    def linking_libraries(self):
        # Libraries we link to may link to other libraries, so we recursively
        # traverse the linking libraries property to build up a list of all
        # linked libraries.
        libs = []
        for lib in self._linking_libraries:
            libs.extend(lib.linking_libraries)
            libs.append(lib)
        return libs

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
        for library in self._linking_libraries:
            for mod in library.modules:
                for fn in mod.functions:
                    if not fn.is_declaration:
                        fn.linkage = "linkonce_odr" # TODO check if this is required

        self._finalized = True

    def _reduce_states(self):
        """
        Reduce the instance for serialization. We retain the LLVM IR and AMD GPU code objects,
        but loaded functions are discarded. They are recreated when needed
        after deserialization.
        """
        if self._linking_files:
            msg = "Cannot pickle HIPCodeLibrary with linking files"
            raise RuntimeError(msg)
        if not self._finalized:
            raise RuntimeError("Cannot pickle unfinalized HIPCodeLibrary")
        return dict(
            codegen=None,
            name=self.name,
            entry_name=self._entry_name,
            llvm_strs=self.llvm_strs,
            llvm_bc_cache=self._llvm_bc_cache,
            codeobj_cache=self._codeobj_cache,
            linkerinfo_cache=self._linkerinfo_cache,
            max_registers=self._max_registers,  # TODO
            # nvvm_options=self._nvvm_options, # TODO
            needs_cudadevrt=self.needs_cudadevrt,
        )

    @classmethod
    def _rebuild(
        cls,
        codegen,
        name,
        entry_name,
        llvm_strs,
        llvm_bc_cache,
        codeobj_cache,
        linkerinfo_cache,
        max_registers, # TODO
        # nvvm_options, # TODO
        # needs_cudadevrt, # TODO
    ):
        """
        Rebuild an instance.
        """
        instance = cls(codegen, name, entry_name=entry_name)

        instance._llvm_strs = llvm_strs
        instance._llvm_bc_cache = llvm_bc_cache
        instance._codeobj_cache = codeobj_cache
        instance._linkerinfo_cache = linkerinfo_cache

        instance._max_registers = max_registers  # TODO
        # instance._nvvm_options = nvvm_options # TODO
        # instance.needs_cudadevrt = needs_cudadevrt TODO

        instance._finalized = True

        return instance


class JITHIPCodegen(Codegen):
    """
    This codegen implementation for HIP only generates optimized LLVM IR.
    Generation of AMD GPU code objects is done separately (see numba.cuda.compiler).
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
        """
        ctx = devices.get_context()
        # cc = ctx.device.compute_capability
        device: driver.Device = ctx.device
        return (driver.get_version(), device.arch)
