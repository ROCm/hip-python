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

# from numba.core import config
from numba.hip.hipdrv.error import (
    HiprtcError,
    HiprtcCompilationError,
)

import ctypes
import functools
import threading
import warnings

# Opaque handle for compilation unit
hiprtc_program = ctypes.c_void_p

# Result code
hiprtc_result = ctypes.c_int

_hiprtc_lock = threading.Lock()


class HiprtcProgram:
    """
    A class for managing the lifetime of hiprtcProgram instances. Instances of
    the class own an hiprtcProgram; when an instance is deleted, the underlying
    hiprtcProgram is destroyed using the appropriate HIPRTC API.
    """

    def __init__(self, hiprtc, handle):
        self._hiprtc = hiprtc
        self._handle = handle

    @property
    def handle(self):
        return self._handle

    def __del__(self):
        if self._handle:
            self._hiprtc.destroy_program(self)


class HIPRTC:
    """
    Provides a Pythonic interface to the HIPRTC APIs, abstracting away the C API
    calls.

    The sole instance of this class is a process-wide singleton.
    Initialization is protected by a lock and uses the standard
    (for Numba) open_cudalib function to load the HIPRTC library.
    """

    # Singleton reference
    __INSTANCE = None

    _FUNCNAMES = [
        "hiprtcGetErrorString",
        "hiprtcVersion",
        "hiprtcAddNameExpression",
        "hiprtcCompileProgram",
        "hiprtcCreateProgram",
        "hiprtcDestroyProgram",
        "hiprtcGetLoweredName",
        "hiprtcGetProgramLog",
        "hiprtcGetProgramLogSize",
        "hiprtcGetCode",
        "hiprtcGetCodeSize",
        "hiprtcGetBitcode",
        "hiprtcGetBitcodeSize",
        "hiprtcLinkCreate",
        "hiprtcLinkAddFile",
        "hiprtcLinkAddData",
        "hiprtcLinkComplete",
        "hiprtcLinkDestroy",
    ]

    @classmethod
    def __new__(cls):
        """Get/return singleton instance.

        Note:
            The initialization code is responsible for
            making it possible to do calls such as `self.hiprtcVersion()`
            and so on.
        """
        with _hiprtc_lock:
            if cls.__INSTANCE is None:
                from hip import hiprtc

                cls.__INSTANCE = inst = object.__new__(cls)

                # Find & populate functions
                for name in cls._FUNCNAMES:
                    func = getattr(hiprtc, name)

                    @functools.wraps(func)
                    def checked_call(*args, func=func, name=name):
                        error = func(*args)
                        if error == hiprtc.hiprtcResult.HIPRTC_ERROR_COMPILATION:
                            raise HiprtcCompilationError()
                        elif error != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
                            try:
                                error_name = error.name
                            except ValueError:
                                error_name = (
                                    "Unknown hiprtc_result " f"(error code: {error})"
                                )
                            msg = f"Failed to call {name}: {error_name}"
                            raise HiprtcError(msg)

                    setattr(inst, name, checked_call)

        return cls.__INSTANCE

    def get_version(self):
        """
        Get the HIPRTC version as a tuple (major, minor).
        """
        return self.hiprtcVersion()

    def create_program(self, src, name):
        """
        Create an HIPRTC program with managed lifetime.
        """
        if isinstance(src, str):
            src = src.encode()
        if isinstance(name, str):
            name = name.encode()

        # The final three arguments are for passing the contents of headers -
        # this is not supported, so there are 0 headers and the header names
        # and contents are null.
        handle = self.hiprtcCreateProgram(src, name, 0, None, None)
        return HiprtcProgram(self, handle)

    def compile_program(self, program, options):
        """
        Compile an HIPRTC program. Compilation may fail due to a user error in
        the source; this function returns ``True`` if there is a compilation
        error and ``False`` on success.
        """
        # We hold a list of encoded options to ensure they can't be collected
        # prior to the call to hiprtcCompileProgram
        encoded_options = [opt.encode() for opt in options]
        try:
            self.hiprtcCompileProgram(program.handle, len(options), encoded_options)
            return False
        except HiprtcCompilationError:
            return True

    def destroy_program(self, program):
        """
        Destroy an HIPRTC program.
        """
        self.hiprtcDestroyProgram(program.handle)

    def get_compile_log(self, program):
        """
        Get the compile log as a Python string.
        """
        log_size = self.hiprtcGetProgramLogSize(program.handle)

        log = bytes(log_size)
        self.hiprtcGetProgramLog(program.handle, log)

        return log.decode()

    def get_llvm_bc(self, program):
        """
        Get LLVM bitcode of the compiled program as Python `bytes`.
        Note:
            Python type `bytes` implements the Python buffer protocol
            and is thus compatible with many HIP Python interfaces.
        """
        llvm_bc_size = self.hiprtcGetBitcodeSize(program.handle)

        llvm_bc = bytes(llvm_bc_size)
        self.hiprtcGetBitcode(program.handle, llvm_bc)

        return llvm_bc


def compile(src, name, amdgpu_arch):
    """
    Compile a HIP C/C++ source to LLVM BC for a given architecture.

    :param src: The source code to compile
    :type src: str
    :param name: The filename of the source (for information only)
    :type name: str
    :param amdgpu_arch: The AMD GPU architecture string.
    :type name: str
    :return: The compiled LLVM BC (`bytes`) and compilation log (`str`)
    :rtype: tuple
    """
    hiprtc = HIPRTC()
    program = hiprtc.create_program(src, name)

    # Compilation options:
    # - Compile for the current device's compute capability.
    # - The CUDA include path is added.
    # - Relocatable Device Code (-fgpu-rdc) is needed to prevent device functions
    #   being optimized away, further will generate LLVM bitcode for AMD GPUs.
    amdgpu_arch = f"--offload-arch={amdgpu_arch}"
    # include = f"-I{config.CUDA_INCLUDE_PATH}"

    # hipdrv_path = os.path.dirname(os.path.abspath(__file__))
    # numba_hip = os.path.dirname(hipdrv_path)
    # numba_include = f"-I{numba_hip}"
    options = [amdgpu_arch, "-fgpu-rdc"]

    # Compile the program
    compile_error = hiprtc.compile_program(program, options)

    # Get log from compilation
    log = hiprtc.get_compile_log(program)

    # If the compile failed, provide the log in an exception
    if compile_error:
        msg = f"HIPRTC Compilation failure whilst compiling {name}:\n\n{log}"
        raise HiprtcError(msg)

    # Otherwise, if there's any content in the log, present it as a warning
    if log:
        msg = f"HIPRTC log messages whilst compiling {name}:\n\n{log}"
        warnings.warn(msg)

    llvm_bc = hiprtc.get_llvm_bc(program)
    hiprtc.destroy_program(program)
    return llvm_bc, log
