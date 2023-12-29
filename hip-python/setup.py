# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

"""This is the package's setup script.

It cythonizes and compiles the Cython
files in the `hip` subfolder.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

import os
import argparse
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
    global HIP_PYTHON_LIBS
    global HIP_PYTHON_RUNTIME_LINKING

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
    HIP_PYTHON_LIBS=os.environ.get("HIP_PYTHON_LIBS", "*")
    HIP_PYTHON_RUNTIME_LINKING=get_bool_environ_var("HIP_PYTHON_RUNTIME_LINKING", "yes")

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
    global HIP_PYTHON_RUNTIME_LINKING
    return Extension(
        name,
        sources=sources,
        include_dirs=[ROCM_INC],
        library_dirs=[ROCM_LIB],
        libraries=[] if HIP_PYTHON_RUNTIME_LINKING else [mod.lib for mod in HIP_MODULES],
        language="c",
        extra_compile_args=EXTRA_COMPILE_ARGS + ["-D","__half=uint16_t"],
    )

class Module:
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

# differs between hip-python and hip-python-as-cuda package
def gather_ext_modules():
    global CYTHON_EXT_MODULES
    global HIP_MODULES
    global HIP_PYTHON_LIBS
    CYTHON_EXT_MODULES.append(("hip._util.types", ["./hip/_util/types.pyx"]))
    CYTHON_EXT_MODULES.append(
        ("hip._util.posixloader", ["./hip/_util/posixloader.pyx"])
    )
    HIP_MODULES += [
        Module(
        "hip",
        lib="amdhip64",
        helpers=[("hip._hip_helpers", ["./hip/_hip_helpers.pyx"])],
        ),
        Module("hiprtc"),
        Module("hipblas"),
        Module("hipsolver"),
        Module("rccl"),
        Module("hiprand"),
        Module("hipfft"),
        Module("hipsparse"),
        Module("roctx"),
    ]

    # process and check user-provided library names
    module_names = [mod.name for mod in HIP_MODULES]
    if HIP_PYTHON_LIBS == "*":
        selected_libs = module_names
    else:
        processed_libs = HIP_PYTHON_LIBS.replace(" ","")
        if processed_libs.startswith("^"):
            processed_libs = processed_libs[1:].split(",")
            selected_libs = [name for name in module_names if name not in processed_libs]
        else:
            processed_libs = processed_libs.split(",")
            selected_libs = processed_libs
        for name in processed_libs:
            if name not in module_names:
                raise ValueError(f"library name '{name}' is not valid, use one of: {', '.join(module_names)}")
            
    for mod in HIP_MODULES:
        if mod.name in selected_libs:
            CYTHON_EXT_MODULES += mod.ext_modules

if __name__ == "__main__":
    HIP_PYTHON_LIBS=None
    HIP_PYTHON_RUNTIME_LINKING=True
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

    # load _version.py
    ns = {}
    exec(open(os.path.join(Module.PKG_NAME,"_version.py"),"r").read(), ns)

    setup(
        ext_modules=ext_modules,
        version = ns["__version__"],
    )