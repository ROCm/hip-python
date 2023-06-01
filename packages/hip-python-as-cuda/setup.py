# AMD_COPYRIGHT

"""This is the packages's setup script.

It cythonizes and compiles the Cython
files in the `cuda` subfolder.
"""

__author__ = "AMD_AUTHOR"

import os
import enum

from setuptools import setup, Extension
from Cython.Build import cythonize

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

def parse_options():
    global ROCM_INC
    global ROCM_LIB
    global EXTRA_COMPILE_ARGS
    global VERBOSE

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

    rocm_path=os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME",None))
    platform=os.environ.get("HIP_PLATFORM","amd")
    verbose=os.environ.get("HIP_PYTHON_VERBOSE","amd")

    if not rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_INC = os.path.join(rocm_path, "include")
    ROCM_LIB = os.path.join(rocm_path, "lib")

    if platform not in ("amd", "hcc"):
        raise RuntimeError("Currently only platform 'amd' is supported")

    EXTRA_COMPILE_ARGS = HipPlatform.from_string(platform).cflags + [f"-I{ROCM_INC}"]

def create_extension(name, sources):
    global ROCM_INC
    global ROCM_LIB
    global HIP_MODULES
    return Extension(
        name,
        sources=sources,
        include_dirs=[ROCM_INC],
        library_dirs=[ROCM_LIB],
        libraries=[mod.lib for mod in HIP_MODULES],
        language="c",
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )

class HipModule:
    PKG_NAME = "hip"

    def __init__(self, module, lib=None, helpers=[]):
        self.name = module
        if lib == None:
            self.lib = self.name
        else:
            self.lib = lib
        self._helpers = helpers

    @property
    def ext_modules(self):
        return self._helpers + [
            (f"{self.PKG_NAME}.c{self.name}", [f"./{self.PKG_NAME}/c{self.name}.pyx"]),
            (f"{self.PKG_NAME}.{self.name}", [f"./{self.PKG_NAME}/{self.name}.pyx"]),
        ]


# differs between hip-python and hip-python-as-nv package
PKG_NAME = "cuda"
def gather_ext_modules():
    HipModule.PKG_NAME = PKG_NAME
    global CYTHON_EXT_MODULES
    global HIP_MODULES
    HIP_MODULES += [
        HipModule(
            "cuda",
            lib="amdhip64",
        ),
        HipModule(
            "cudart",
            lib="amdhip64",
        ),
        HipModule("nvrtc"),
    ]
    for mod in HIP_MODULES:
        CYTHON_EXT_MODULES += mod.ext_modules

main_ns = {}
exec(open(os.path.join(PKG_NAME,"_version.py"),"r").read(), main_ns)

if __name__ == "__main__":
    ROCM_INC = None
    ROCM_LIB = None
    EXTRA_COMPILE_ARGS = None
    VERBOSE = False
    HIP_MODULES = []
    CYTHON_EXT_MODULES = []
    
    parse_options()
    gather_ext_modules() 
    
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
        version = main_ns["__version__"],
    )
