# AMD_COPYRIGHT

"""This is the packages's setup script.

It cythonizes and compiles the Cython
files in the `hip` subfolder.
"""

__author__ = "AMD_AUTHOR"

import os
import argparse
import enum

from setuptools import setup, Extension
from Cython.Build import cythonize

def parse_options():
    global ROCM_INC
    global ROCM_LIB
    global EXTRA_COMPILE_ARGS

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

    parser = argparse.ArgumentParser(description="Generator for HIP Python packages")
    parser.add_argument("--rocm-path",type=str,required=False,dest="rocm_path",
                        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.")
    parser.add_argument("--platform",type=str,required=False,dest="platform",
                        help="The HIP platform, 'amd' or 'nvidia'. Can be set via environment variable 'HIP_PLATFORM' too.")
    parser.add_argument("-v","--verbose",required=False,action="store_true",dest="verbose",
                        default=False, help="Verbose output.")
    parser.set_defaults(
        rocm_path=os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME",None)),
        platform=os.environ.get("HIP_PLATFORM","amd"),
        verbose=False,
    )
    args = parser.parse_args()

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_INC = os.path.join(args.rocm_path, "include")
    ROCM_LIB = os.path.join(args.rocm_path, "lib")

    if args.platform not in ("amd", "hcc"):
        raise RuntimeError("Currently only platform 'amd' is supported")

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

    EXTRA_COMPILE_ARGS = HipPlatform.from_string(args.platform).cflags + [f"-I{ROCM_INC}"]

def create_extension(name, sources):
    global ROCM_INC
    global ROCM_LIB
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
            (f"{self.PKG_NAME}.c{self.name}}", [f"./{self.PKG_NAME}/c{self.name}.pyx"]),
            (f"{self.PKG_NAME}.{self.name}", [f"./{self.PKG_NAME}/{self.name}.pyx"]),
        ]

# differs between hip-python and hip-python-as-cuda package
def gather_ext_modules():
    HipModule.PKG_NAME = "hip"
    global CYTHON_EXT_MODULES
    global HIP_MODULES
    CYTHON_EXT_MODULES.append(("hip._util.types", ["./hip/_util/types.pyx"]))
    CYTHON_EXT_MODULES.append(
        ("hip._util.posixloader", ["./hip/_util/posixloader.pyx"])
    )
    HIP_MODULES += [
        HipModule(
        "hip",
        lib="amdhip64",
        helpers=[("hip._hip_helpers", ["./hip/_hip_helpers.pyx"])],
        ),
        HipModule("hiprtc"),
        HipModule("hipblas"),
        HipModule("rccl"),
        HipModule("hiprand"),
        HipModule("hipfft"),
        HipModule("hipsparse"),
    ]
    CYTHON_EXT_MODULES += [mod.ext_modules for mod in HIP_MODULES]

supported_hip_modules = [
    HipModule(
        "hip",
        lib="amdhip64",
        helpers=[("hip._hip_helpers", ["./hip/_hip_helpers.pyx"])],
    ),
    HipModule("hiprtc"),
    HipModule("hipblas"),
    HipModule("rccl"),
    HipModule("hiprand"),
    HipModule("hipfft"),
    HipModule("hipsparse"),
]

if __name__ == "__main__":
    HIP_MODULES = []
    CYTHON_EXT_MODULES = []
    
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
    )