# AMD_COPYRIGHT

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "AMD_AUTHOR"

import os
import warnings
import enum
import textwrap

try:
    # package to calculate word distances
    import Levenshtein

    HAVE_LEVENSHTEIN = True
except ImportError:
    HAVE_LEVENSHTEIN = False

print(f"{HAVE_LEVENSHTEIN=}")

from _codegen import cython

from _codegen.cython import (
    CythonPackageGenerator,
    DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

from _codegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from _codegen.tree import (
    Node,
    MacroDefinition,
    Function,
    Typedef,
    Enum,
    AnonymousEnum,
    Parm,
    Field,
    Record,
    FunctionPointer,
)

from _codegen.control import PointerParamIntent

from _parse_hipify_perl import render_hipify_perl_info, parse_hipify_perl

# Configuration
ROCM_PATH = os.environ.get("ROCM_PATH", None)
if not ROCM_PATH:
    ROCM_PATH = os.environ.get("ROCM_HOME")
if not ROCM_PATH:
    raise RuntimeError("Environment variable ROCM_PATH is not set")
ROCM_INC = os.path.join(ROCM_PATH, "include")
HIPIFY_PERL_PATH = os.path.join(ROCM_PATH, "bin", "hipify-perl")

CFLAGS = os.environ.get("CFLAGS", None)

HIP_PLATFORM = os.environ.get("HIP_PLATFORM", "amd")
if HIP_PLATFORM not in ("amd", "hcc"):
    raise RuntimeError("Currently only HIP_PLATFORM=amd is supported")


class HipPlatform(enum.IntEnum):
    AMD = 0
    NVIDIA = 1

    @staticmethod
    def from_string(key: str):
        valid_inputs = ("amd", "hcc", "nvidia", "nvcc")
        key = key.lower()
        if key in valid_inputs[0:2]:
            return HipPlatform.AMD
        elif key in valid_inputs[2:4]:
            return HipPlatform.NVIDIA
        else:
            raise ValueError(
                f"Input must be one of: {','.join(valid_inputs)} (any case)"
            )

    @property
    def cflags(self):
        return ["-D", f"__HIP_PLATFORM_{self.name}__"]


hip_platform = HipPlatform.from_string(HIP_PLATFORM)


def get_bool_environ_var(env_var, default):
    yes_vals = ("true", "1", "t", "y", "yes")
    no_vals = ("false", "0", "f", "n", "no")
    value = os.environ.get(env_var, default).lower()
    if value in yes_vals:
        return True
    elif value in no_vals:
        return False
    else:
        allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals) + list(no_vals))])
        raise RuntimeError(
            f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}"
        )


HIP_PYTHON_GENERATE = get_bool_environ_var("HIP_PYTHON_GENERATE", "true")
HIP_PYTHON_LIBS = os.environ.get("HIP_PYTHON_LIBS", "hip,hiprtc")
HIP_PYTHON_ERR_IF_LIB_NOT_FOUND = get_bool_environ_var(
    "HIP_PYTHON_ERR_IF_LIB_NOT_FOUND", "true"
)
HIP_PYTHON_BUILD = get_bool_environ_var("HIP_PYTHON_BUILD", "true")
HIP_PYTHON_RUNTIME_LINKING = get_bool_environ_var("HIP_PYTHON_RUNTIME_LINKING", "true")
HIP_PYTHON_VERBOSE = get_bool_environ_var("HIP_PYTHON_VERBOSE", "true")

GENERATOR_ARGS = hip_platform.cflags + [f"-I{ROCM_INC}"]
HIP_PYTHON_CLANG_RES_DIR = os.environ.get("HIP_PYTHON_CLANG_RES_DIR", None)
if HIP_PYTHON_GENERATE:
    if not HIP_PYTHON_CLANG_RES_DIR:
        raise RuntimeError(
            textwrap.dedent(
                """\
            Environment variable HIP_PYTHON_CLANG_RES_DIR is not set.
            
            Hint: If `clang` is installed and in the PATH, you can 
            run `clang -print-resource-dir` to obtain the path to
            the resource directory.
    
            Hint: If you have the HIP SDK installed, you have `amdclang` installed in
            `ROCM_PATH/bin/`. You can use it to run the above command too.
            
            Hint: If you have the HIP SDK installed, the last include folder listed in ``hipconfig --cpp_config``
            points to the `amdclang` compiler's resource dir too.
            """
            )
        )

    GENERATOR_ARGS += ["-resource-dir", HIP_PYTHON_CLANG_RES_DIR]

if HIP_PYTHON_VERBOSE:
    print("Environment variables:")
    print(f"{ROCM_PATH=}")
    print(f"{HIP_PLATFORM=}")
    print(f"{HIP_PYTHON_CLANG_RES_DIR=}")
    print(f"{HIP_PYTHON_GENERATE=}")
    print(f"{HIP_PYTHON_LIBS=}")
    print(f"{HIP_PYTHON_ERR_IF_LIB_NOT_FOUND=}")
    print(f"{HIP_PYTHON_BUILD=}")
    print(f"{HIP_PYTHON_RUNTIME_LINKING=}")
    print(f"{HIP_PYTHON_VERBOSE=}")


CYTHON_EXT_MODULES = []
LIBRARIES = []

HIP_VERSION_MAJOR = 0
HIP_VERSION_MINOR = 0
HIP_VERSION_PATCH = 0
HIP_VERSION_GITHASH = ""

(CUDA_2_HIP, HIP_2_CUDA) = parse_hipify_perl(HIPIFY_PERL_PATH)


def generate_cuda_interop_package_files(
    cuda_pkg_name: str, generator: CythonPackageGenerator, warn: bool = True
):
    global HIP_2_CUDA
    global HAVE_LEVENSHTEIN
    global CYTHON_EXT_MODULES
    output_dir = "cuda"
    indent = " " * 4
    pkg_name = generator.pkg_name
    backend = generator.backend

    c_interface_decl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT
            
            cimport hip.c{pkg_name}
            """
        ),
    ]
    python_interface_decl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT

            __author__ = "AMD_AUTHOR"

            cimport hip.c{pkg_name}
            cimport hip.{pkg_name}
            """
        ),
        f"cimport {output_dir}.c{cuda_pkg_name}",  # for checking compiler errors
    ]
    python_interface_impl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT

            __author__ = "AMD_AUTHOR"

            import os
            import enum

            import hip.{pkg_name}

            {pkg_name} = hip.{pkg_name} # makes {pkg_name} types and routines accessible without import
                                        # allows checks such as `hasattr(cuda.{cuda_pkg_name},"{pkg_name}")`
            """
        ),
        textwrap.dedent(
            """\
            def _hip_python_get_bool_environ_var(env_var, default):
                yes_vals = ("true", "1", "t", "y", "yes")
                no_vals = ("false", "0", "f", "n", "no")
                value = os.environ.get(env_var, default).lower()
                if value in yes_vals:
                    return True
                elif value in no_vals:
                    return False
                else:
                    allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
                    raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")
            """
        ),
    ]

    # impl part is always empty
    def warn_(hip_name):
        global HAVE_LEVENSHTEIN
        global HIP_2_CUDA
        msg = f"hipify-perl: no CUDA symbol found for HIP symbol {hip_name}"
        if HAVE_LEVENSHTEIN:
            cutoff = 0.9
            candidates = []
            for other_hip_name in HIP_2_CUDA:
                if (
                    Levenshtein.ratio(
                        hip_name,
                        other_hip_name,
                        processor=lambda tk: tk.lower(),  # do everything in lowercase
                        score_cutoff=cutoff,  # everything below cutoff is set to 0
                    )
                    > 0
                ):
                    candidates.append(other_hip_name)
            candidates_formatted = ", ".join(["'" + c + "'" for c in candidates])
            msg += f"; most similar hipify-perl HIP symbols (Levenshtein ratio > {cutoff}): [{candidates_formatted}]"
        warnings.warn(msg)

    def handle_enum_(node, cuda_name):
        global HIP_2_CUDA
        nonlocal indent
        nonlocal c_interface_decl_part
        nonlocal python_interface_impl_part
        enum = node if isinstance(node, Enum) else node.lookup_innermost_type()
        constants = []
        for child_cursor in enum.cursor.get_children():
            hip_constant_name = child_cursor.spelling
            if hip_constant_name in HIP_2_CUDA:
                for cuda_constant_name in HIP_2_CUDA[hip_constant_name]:
                    constants.append(
                        f"{cuda_constant_name} = hip.c{pkg_name}.{hip_constant_name}"
                    )
            else:
                warn_(hip_constant_name)
            # append hip constant too, to help workarounds
            constants.append(
                f"{hip_constant_name} = hip.c{pkg_name}.{hip_constant_name}"
            )
        if isinstance(node, AnonymousEnum):  # cannot be typedefed
            python_interface_impl_part += constants
        else:
            python_enum_metaclass_name = f"_{cuda_name}_EnumMeta"
            python_enum_hallucinate_var_name = (
                f"HIP_PYTHON_{cuda_name}_HALLUCINATE_CONSTANTS"
            )
            python_enum_metaclass = textwrap.dedent(
                f"""\
                
                {python_enum_hallucinate_var_name} = _hip_python_get_bool_environ_var("{python_enum_hallucinate_var_name}","false")

                class {python_enum_metaclass_name}(enum.EnumMeta):
                
                    class FakeEnumType():
                        \"""Mimicks the orginal enum type this 
                        is derived from.
                        \"""
                        
                        def __init__(self):
                            pass
                        
                        @property
                        def name(self):
                            return self._name_

                        @property
                        def value(self):
                            return self._value_
                            
                        def __eq__(self,other):
                            if isinstance(other,self._orig_enum_type_):
                                return self.value == other.value
                            return False
                        
                        @property
                        def __class__(self):
                            \"""Overwrite __class__ to satisfy __isinstance__ check.
                            \"""
                            return self._orig_enum_type_

                        def __repr__(self):        
                            \"""Mimicks enum.Enum.__repr__\"""
                            return "<%s.%s: %r>" % (
                                    self.__class__.__name__, self._name_, self._value_)
                                    
                        def __str__(self):
                            \"""Mimicks enum.Enum.__str__\"""
                            return "%s.%s" % (self.__class__.__name__, self._name_)

                        def __hash__(self):
                            return hash(str(self))

                    def __getattribute__(cls,name):
                        global _get_hip_name
                        global {python_enum_hallucinate_var_name}
                        try:
                            result = super().__getattribute__(name)
                            return result
                        except AttributeError as ae:
                            if not {python_enum_hallucinate_var_name}:
                                raise ae
                            else:
                                used_vals = list(cls._value2member_map_.keys())
                                if not len(used_vals):
                                    raise ae
                                new_val = min(used_vals)
                                while new_val in used_vals: # find a free enum value
                                    new_val += 1
                                enum_types = list(cls._member_map_.values())
                                enum_class = enum_types[0].__class__
                                fake_enum = type(
                                    name,
                                    (cls.FakeEnumType,),
                                    {{"_name_":name,"_value_": new_val,"_orig_enum_type_": enum_class}}
                                )()
                                return fake_enum
                """
            )
            python_enum_class = textwrap.dedent(
                f"""
                class {cuda_name}(enum.IntEnum,metaclass={python_enum_metaclass_name}):
                    @property
                    def __class__(self):
                        \"""Overwrite __class__ to satisfy __isinstance__ check.
                        \"""
                        return hip.{pkg_name}.{hip_name}
                """
            )
            python_enum_class += textwrap.indent("\n".join(constants), indent)

            python_interface_impl_part.append(python_enum_metaclass)
            python_interface_impl_part.append(python_enum_class)

        if isinstance(node, Enum) and i == 0:
            cython_enum = f"cdef enum {cuda_name}:\n" + textwrap.indent(
                "\n".join(constants), indent
            )
            c_interface_decl_part.append(cython_enum)
        else:  # if it is a typedef or there are multiple CUDA names
            hip_underlying_type_name = enum.name
            if hip_underlying_type_name in HIP_2_CUDA:
                cuda_underlying_type_name = HIP_2_CUDA[hip_underlying_type_name][
                    0
                ]  # take first
                cython_enum = f"ctypedef {cuda_underlying_type_name} {cuda_name}"
                c_interface_decl_part.append(cython_enum)
            else:
                warn_(hip_underlying_type_name)

    # main loop over nodes
    for node in backend.walk_filtered_nodes():
        hip_name = node.name
        if isinstance(node, AnonymousEnum):
            # Anonymous enums won't have a different CUDA name but their constants might
            handle_enum_(node, hip_name)  # hip_name is auto_generated in this case
        if hip_name in HIP_2_CUDA:
            cuda_names = HIP_2_CUDA[hip_name]
            for i, cuda_name in enumerate(cuda_names):
                if isinstance(node, Enum) or (
                    isinstance(node, Typedef)
                    and node.is_pointer_to_enum(degree=(0, -1))
                ):
                    # enums require special care as they are modelled as "class <type>"
                    # and not as "cdef class" in the Python interface, just like in CUDA Python.
                    handle_enum_(node, cuda_name)
                elif (
                    isinstance(
                        node,
                        (
                            MacroDefinition,
                            Function,
                        ),
                    )
                    or isinstance(node, Typedef)
                    and node.is_pointer_to_record(degree=(0, -1))
                ):
                    # These are Python objects/functions in the Python interface
                    if i == 0:
                        c_interface_decl_part.append(
                            f"from hip.c{pkg_name} cimport {hip_name}"
                        )
                    c_interface_decl_part.append(
                        f"from hip.c{pkg_name} cimport {hip_name} as {cuda_name}"
                    )
                    python_interface_impl_part.append(
                        f"{cuda_name} = hip.{pkg_name}.{hip_name}"
                    )
                    # TODO what about basic/void types? Should have a c_interface_decl_part
                elif isinstance(node, Typedef) and (
                    node.is_pointer_to_basic_type(degree=(0, -1))
                    or node.is_pointer_to_void(degree=(0, -1))
                ):
                    canonical_type = node.cursor.type.get_canonical().spelling
                    c_interface_decl_part.append(
                        f"ctypedef {canonical_type} {cuda_name}"
                    )
                elif isinstance(node, (FunctionPointer, Record)):
                    # These are cdef classes ("extension types").
                    # So Python interface declaration must be cimported.
                    # and a subclass needs to be created to define a Python object. (TODO other options?)
                    if i == 0:
                        c_interface_decl_part.append(
                            f"from hip.c{pkg_name} cimport {hip_name}"
                        )
                    c_interface_decl_part.append(
                        f"from hip.c{pkg_name} cimport {hip_name} as {cuda_name}"
                    )
                    #
                    if i == 0 and hip_name not in cuda_names:
                        python_interface_decl_part.append(
                            f"from hip.{pkg_name} cimport {hip_name} # here"
                        )  # note the missing c prefix
                    # if hip_name != cuda_name:
                    cdef_subclass = f"cdef class {cuda_name}(hip.{pkg_name}.{hip_name}):\n{indent}pass"
                    python_interface_decl_part.append(cdef_subclass)
                    python_interface_impl_part.append(cdef_subclass)
        elif warn:
            warn_(hip_name)

    python_interface_decl_path = os.path.join(output_dir, f"{cuda_pkg_name}.pxd")
    python_interface_impl_path = os.path.join(output_dir, f"{cuda_pkg_name}.pyx")
    c_interface_decl_path = os.path.join(output_dir, f"c{cuda_pkg_name}.pxd")
    with open(c_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(c_interface_decl_part))
    with open(python_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(python_interface_decl_part))
    with open(python_interface_impl_path, "w") as outfile:
        outfile.write("\n".join(python_interface_impl_part))
    CYTHON_EXT_MODULES += [
        (f"{output_dir}.{cuda_pkg_name}", [python_interface_impl_path]),
    ]


def generate_hiprtc_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    # hiprtc
    def hiprtc_node_filter(node: Node):
        if isinstance(node, MacroDefinition):
            return node.name.startswith("hiprtc")
        if node.file is None:
            print(f"node.file is None: {node.cursor.kind}")
        if node.file.endswith("hiprtc.h"):
            return True
        return False

    def hiprtc_ptr_parm_intent(parm: Parm):
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
            return PointerParamIntent.OUT
        if (parm.parent.name, parm.name) in inout_parms:
            return PointerParamIntent.INOUT
        return PointerParamIntent.IN

    def hiprtc_ptr_rank(node: Node):
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
        return 1

    def hiprtc_ptr_complicated_type_handler(parm: Parm):
        list_of_str_parms = (
            ("hiprtcCompileProgram", "options"),
            ("hiprtcCreateProgram", "headers"),
            ("hiprtcCreateProgram", "includeNames"),
        )
        if (parm.parent.name, parm.name) in list_of_str_parms:
            return "hip._util.types.ListOfBytes"
        return DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonPackageGenerator(
        "hiprtc",
        ROCM_INC,
        "hip/hiprtc.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libhiprtc.so",
        node_filter=hiprtc_node_filter,
        ptr_parm_intent=hiprtc_ptr_parm_intent,
        ptr_rank=hiprtc_ptr_rank,
        ptr_complicated_type_handler=hiprtc_ptr_complicated_type_handler,
        cflags=GENERATOR_ARGS,
    )
    CYTHON_EXT_MODULES += [
        ("hip.chiprtc", ["./hip/chiprtc.pyx"]),
        ("hip.hiprtc", ["./hip/hiprtc.pyx"]),
    ]
    LIBRARIES.append("hiprtc")
    generate_cuda_interop_package_files("nvrtc", generator)
    return generator


def generate_hip_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    global HIP_VERSION_MAJOR
    global HIP_VERSION_MINOR
    global HIP_VERSION_PATCH
    global HIP_VERSION_GITHASH

    # hip
    hip_str_macros = (
        "HIP_VERSION_GITHASH",
        "HIP_VERSION_BUILD_NAME",
    )
    hip_void_p_macros = (
        "HIP_LAUNCH_PARAM_BUFFER_POINTER",
        "HIP_LAUNCH_PARAM_BUFFER_SIZE",
        "HIP_LAUNCH_PARAM_END",
    )
    hip_int_macros = (
        #  from hip/hip_version.h
        "HIP_VERSION_MAJOR",
        "HIP_VERSION_MINOR",
        "HIP_VERSION_PATCH",
        # "HIP_VERSION_GITHASH", # no int, is char*
        "HIP_VERSION_BUILD_ID",
        # "HIP_VERSION_BUILD_NAME", # is char*
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

    def hip_node_filter(node: Node):
        if isinstance(node, Function):
            if not node.name.startswith("hip"):
                return False
        if node.name in hip_int_macros:
            return True
        if node.name in hip_str_macros:
            return True
        if node.name in hip_void_p_macros:
            return True
        if not isinstance(node, MacroDefinition):
            if "hip/" in node.file:
                # some modifications:
                if isinstance(node, Record) and node.name == "dim3":
                    node.set_defaults(x=1, y=1, z=1)
                return True
        return False

    def hip_macro_type(node: MacroDefinition):
        if node.name in hip_int_macros:
            return "int"
        if node.name in hip_void_p_macros:
            return "unsigned long int"
        if node.name in hip_str_macros:
            return "char*"
        assert False, "Not implemented!"

    def hip_ptr_parm_intent(parm: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.

        Rules
        -----

        1. We exploit that ``hip/hip_runtime_api.h``` does not
        work with typed arrays, so every pointer
        of basic type is actually a return value
        that is created internally by the function.
        Exceptions are ``char*`` parameters, which
        are C-style strings.

        2. All ``void``, ``struct``, ``union``, ``enum`` double (``**``) pointers are
        return values that are created internally by the respective function.
        """
        if (
            parm.is_pointer_to_record(degree=2)
            or parm.is_pointer_to_enum(degree=1)
            or (
                parm.is_pointer_to_basic_type(degree=1)
                and not parm.is_pointer_to_char(degree=1)
            )
        ):
            if (parm.parent.name, parm.name) == ("hipDeviceGetAttribute", "pi"):
                return PointerParamIntent.INOUT
            return PointerParamIntent.OUT
        if parm.is_pointer_to_void(degree=2):
            if parm.name in ["devPtr", "ptr", "dev_ptr", "data", "dptr"]:
                return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hip_ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if (
                node.is_pointer_to_basic_type(degree=1)
                or node.is_pointer_to_enum(degree=1)
                or node.is_pointer_to_record(degree=1)
                or node.is_pointer_to_record(degree=2)
            ):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    def toclassname(name: str):
        return name[0].upper() + name[1:]

    def hip_ptr_complicated_type_handler(parm: Node):
        if (parm.parent.name, parm.name) == ("hipModuleLaunchKernel", "extra"):
            return f"hip._hip_helpers.{toclassname(parm.parent.name)}_{parm.name}"
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
            return "hip._util.types.DeviceArray"
        return DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonPackageGenerator(
        "hip",
        ROCM_INC,
        "hip/hip_runtime.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libamdhip64.so",
        node_filter=hip_node_filter,
        ptr_parm_intent=hip_ptr_parm_intent,
        ptr_rank=hip_ptr_rank,
        ptr_complicated_type_handler=hip_ptr_complicated_type_handler,
        macro_type=hip_macro_type,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_preamble += textwrap.dedent(
        """\
    cimport hip._hip_helpers
    """
    )
    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""
    for node in generator.backend.root.walk():
        if isinstance(node, MacroDefinition):
            last_token = list(node.cursor.get_tokens())[-1].spelling
            if node.name == "HIP_VERSION_MAJOR":
                HIP_VERSION_MAJOR = int(last_token)
            elif node.name == "HIP_VERSION_MINOR":
                HIP_VERSION_MINOR = int(last_token)
            elif node.name == "HIP_VERSION_PATCH":
                HIP_VERSION_PATCH = int(last_token)
            elif node.name == "HIP_VERSION_GITHASH":
                HIP_VERSION_GITHASH = last_token.strip('"')

    CYTHON_EXT_MODULES += [
        ("hip.chip", ["./hip/chip.pyx"]),
        ("hip._hip_helpers", ["./hip/_hip_helpers.pyx"]),
        ("hip.hip", ["./hip/hip.pyx"]),
    ]
    LIBRARIES.append("amdhip64")

    generate_cuda_interop_package_files("cuda", generator)
    generate_cuda_interop_package_files(
        "cudart", generator, warn=False
    )  # already warned before, regenerate to have correctly named pxd/pyx files too. Could be done via symlinks & __init__.py mod too.
    return generator


# hipblas
def generate_hipblas_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    def hipblas_node_filter(node: Node):
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

    def hipblas_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_void(degree=2) and node.name == "handle":
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hipblas_ptr_rank(node: Node):
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
                    [TypeCategory.ARRAY, TypeCategory.POINTER, TypeCategory.BASIC],
                    [TypeCategory.ARRAY, TypeCategory.POINTER, TypeCategory.VOID],
                ):
                    return 2
                return 1
            elif len(node.name) == 1 and node.name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                return 2
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    generator = CythonPackageGenerator(
        "hipblas",
        ROCM_INC,
        "hipblas/hipblas.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libhipblas.so",
        node_filter=hipblas_node_filter,
        ptr_parm_intent=hipblas_ptr_parm_intent,
        ptr_rank=hipblas_ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    #ctypedef int16_t __int16_t
    #ctypedef uint16_t __uint16_t
    from .hip cimport ihipStream_t
    """
    )
    CYTHON_EXT_MODULES += [
        ("hip.chipblas", ["./hip/chipblas.pyx"]),
        ("hip.hipblas", ["./hip/hipblas.pyx"]),
    ]
    LIBRARIES.append("hipblas")
    return generator


# rccl
def generate_rccl_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    def rccl_node_filter(node: Node):
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

    def rccl_macro_type(node: MacroDefinition):
        rccl_str_macros = "NCCL_SUFFIX"
        if node.name in rccl_str_macros:
            return "char*"
        return "int"

    def rccl_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "comm"),
                ("pncclCommInitAll", "comm"),
            ):
                return PointerParamIntent.INOUT
            return PointerParamIntent.OUT
        if node.is_pointer_to_record(degree=1):
            if (node.parent.name, node.name) == "ncclGetUniqueId":
                return PointerParamIntent.OUT
        if node.is_pointer_to_basic_type(degree=1):
            if (node.parent.name, node.name) in (
                ("ncclCommInitAll", "devlist"),
                ("pncclCommInitAll", "devlist"),
            ):
                return PointerParamIntent.IN
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def rccl_ptr_rank(node: Node):
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
        return 1

    generator = CythonPackageGenerator(
        "rccl",
        ROCM_INC,
        "rccl/rccl.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="librccl.so",
        node_filter=rccl_node_filter,
        macro_type=rccl_macro_type,
        ptr_parm_intent=rccl_ptr_parm_intent,
        ptr_rank=rccl_ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    #ctypedef int16_t __int16_t
    #ctypedef uint16_t __uint16_t
    from .hip cimport ihipStream_t
    """
    )
    CYTHON_EXT_MODULES += [
        ("hip.crccl", ["./hip/crccl.pyx"]),
        ("hip.rccl", ["./hip/rccl.pyx"]),
    ]
    LIBRARIES.append("rccl")
    return generator


# hiprand
def generate_hiprand_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    def hiprand_node_filter(node: Node):
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

    def hiprand_macro_type(node: MacroDefinition):
        return "int"

    def hiprand_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return PointerParamIntent.OUT
        if node.is_pointer_to_basic_type(degree=1):
            if node.name == "output_data":
                return PointerParamIntent.INOUT
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hiprand_ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections.

        Most of the parameter names follow LAPACK convention.
        """
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=1):
                return 0
            if node.is_pointer_to_record(degree=2):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        elif isinstance(node, Field):
            pass  # nothing to do
        return 1

    generator = CythonPackageGenerator(
        "hiprand",
        ROCM_INC,
        "hiprand/hiprand.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libhiprand.so",
        node_filter=hiprand_node_filter,
        macro_type=hiprand_macro_type,
        ptr_parm_intent=hiprand_ptr_parm_intent,
        ptr_rank=hiprand_ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    CYTHON_EXT_MODULES += [
        ("hip.chiprand", ["./hip/chiprand.pyx"]),
        ("hip.hiprand", ["./hip/hiprand.pyx"]),
    ]
    LIBRARIES.append("hiprand")
    return generator


# hipfft
def generate_hipfft_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    def hipfft_node_filter(node: Node):
        if not isinstance(node, MacroDefinition):
            if node.name.startswith("hipfft"):
                return True
        elif node.name in (
            "HIPFFT_FORWARD",
            "HIPFFT_BACKWARD",
        ):
            return True
        return False

    def hipfft_macro_type(node: MacroDefinition):
        return "int"

    def hipfft_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return PointerParamIntent.OUT
        if node.name == "workSize":
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hipfft_ptr_rank(node: Node):
        """Actual rank of the variables underlying pointer indirections."""
        if isinstance(node, Parm):
            if node.is_pointer_to_record(degree=(1, 2)):
                return 0
            elif node.is_pointer_to_basic_type(degree=1):
                return 0
        return 1

    generator = CythonPackageGenerator(
        "hipfft",
        ROCM_INC,
        "hipfft/hipfft.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libhipfft.so",
        node_filter=hipfft_node_filter,
        macro_type=hipfft_macro_type,
        ptr_parm_intent=hipfft_ptr_parm_intent,
        ptr_rank=hipfft_ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t, float2, double2
    """
    )
    CYTHON_EXT_MODULES += [
        ("hip.chipfft", ["./hip/chipfft.pyx"]),
        ("hip.hipfft", ["./hip/hipfft.pyx"]),
    ]
    LIBRARIES.append("hipfft")
    return generator


# hipsparse
def generate_hipsparse_package_files():
    global ROCM_INC
    global HIP_PYTHON_GENERATE
    global GENERATOR_ARGS
    global CYTHON_EXT_MODULES

    def hipsparse_node_filter(node: Node):
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

    def hipsparse_macro_type(node: MacroDefinition):
        return "int"

    def hipsparse_ptr_parm_intent(node: Parm):
        """Flags pointer parameters that are actually return values
        that are passed as C-style reference, i.e. `<type>* <param>`.
        """
        if node.is_pointer_to_record(degree=2):
            return PointerParamIntent.OUT
        return PointerParamIntent.IN

    def hipsparse_ptr_rank(node: Node):
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
        return 1

    generator = CythonPackageGenerator(
        "hipsparse",
        ROCM_INC,
        "hipsparse/hipsparse.h",
        runtime_linking=HIP_PYTHON_RUNTIME_LINKING,
        dll="libhipsparse.so",
        node_filter=hipsparse_node_filter,
        macro_type=hipsparse_macro_type,
        ptr_parm_intent=hipsparse_ptr_parm_intent,
        ptr_rank=hipsparse_ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport *
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip import hipError_t, hipDataType # PY import enums
    from .hip cimport ihipStream_t, float2, double2 # C import structs/union types
    """
    )
    CYTHON_EXT_MODULES += [
        ("hip.chipsparse", ["./hip/chipsparse.pyx"]),
        ("hip.hipsparse", ["./hip/hipsparse.pyx"]),
    ]
    LIBRARIES.append("hipsparse")
    return generator


AVAILABLE_GENERATORS = dict(
    hip=generate_hip_package_files,
    hiprtc=generate_hiprtc_package_files,
    hipblas=generate_hipblas_package_files,
    rccl=generate_rccl_package_files,
    hiprand=generate_hiprand_package_files,
    hipfft=generate_hipfft_package_files,
    hipsparse=generate_hipsparse_package_files,
)

if len(HIP_PYTHON_LIBS.strip()):
    lib_names = (
        AVAILABLE_GENERATORS.keys()
        if HIP_PYTHON_LIBS == "*"
        else HIP_PYTHON_LIBS.split(",")
    )
else:
    lib_names = []
for entry in lib_names:
    libname = entry.strip()
    if libname not in AVAILABLE_GENERATORS:
        available_libs = ", ".join([f"'{a}'" for a in AVAILABLE_GENERATORS.keys()])
        msg = f"no codegenerator found for library '{libname}'; please choose one of: {available_libs}, or '*', which implies that all code generators will be used."
        if HIP_PYTHON_ERR_IF_LIB_NOT_FOUND:
            raise KeyError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)
    generator = AVAILABLE_GENERATORS[libname]()
    if generator != None and HIP_PYTHON_GENERATE:
        generator.write_package_files(output_dir="hip")

HIP_VERSION_NAME = (
    f"{HIP_VERSION_MAJOR}.{HIP_VERSION_MINOR}.{HIP_VERSION_PATCH}-{HIP_VERSION_GITHASH}"
)
HIP_VERSION = (
    HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH
)

with open("hip/__init__.py", "w") as f:
    init_content = textwrap.dedent(
        f"""\
        from ._version import *
        HIP_VERSION = {HIP_VERSION}
        HIP_VERSION_NAME = hip_version_name = "{HIP_VERSION_NAME}"
        HIP_VERSION_TUPLE = hip_version_tuple = ({HIP_VERSION_MAJOR},{HIP_VERSION_MINOR},{HIP_VERSION_PATCH},"{HIP_VERSION_GITHASH}")

        from . import _util
        """
    )

    for pkg_name in AVAILABLE_GENERATORS.keys():
        init_content += f"from . import {pkg_name}\n"

    f.write(init_content)

# Build Cython packages
if HIP_PYTHON_BUILD:
    from setuptools import setup, Extension
    from Cython.Build import cythonize

    if HIP_PYTHON_RUNTIME_LINKING:
        library_dirs = []
        libraries = []
    else:
        library_dirs = [os.path.join(ROCM_PATH, "lib")]
        libraries = LIBRARIES

    extra_compile_args = hip_platform.cflags
    if CFLAGS == None:
        extra_compile_args += ["-O3"] + ["-D", "__half=uint16_t"]

    def create_extension(name, sources):
        return Extension(
            name,
            sources=sources,
            include_dirs=[ROCM_INC],
            library_dirs=library_dirs,
            libraries=libraries,
            language="c",
            extra_compile_args=extra_compile_args,
        )

    if HIP_PYTHON_RUNTIME_LINKING:
        CYTHON_EXT_MODULES.insert(
            0, ("hip._util.posixloader", ["./hip/_util/posixloader.pyx"])
        )
    CYTHON_EXT_MODULES.insert(0, ("hip._util.types", ["./hip/_util/types.pyx"]))

    ext_modules = []
    for name, sources in CYTHON_EXT_MODULES:
        extension = create_extension(name, sources)
        ext_modules += cythonize(
            [extension],
            compiler_directives=dict(
                embedsignature=True,
                language_level=3,
            ),
        )

    setup(
        ext_modules=ext_modules,
        use_scm_version={
            "write_to": "hip/_version.py",
            "local_scheme": lambda v: f"+{HIP_VERSION_NAME.replace('-','.')}",
        },
    )
