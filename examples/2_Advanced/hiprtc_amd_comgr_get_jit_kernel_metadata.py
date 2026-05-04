#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Parse metadata of a JIT-compiled HIP kernel

In this example, we compile a HIP C++ kernel just-in-time
via HIPRTC and then use AMD COMGR functionality to
parse the metadata that is embedded within the resulting
code object. We further print out all symbols within
the code object and disassemble the kernel code,
which is associated with the main kernels'
symbol within the code object. Finally, we print
out the properties that
AMD COMGR associates with the compilation target.

Note that some necessary includes such as "hip/hip_runtime.h" are
prepended to the kernel by hipRTC internally as pre-processing step.
Hence, they do not appear in the kernel source.
"""

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

# [literalinclude-begin]
from rocm.bindings import hip, hiprtc
from rocm import comgr


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result


class HipProgram:
    def __init__(
        self,
        program_name,  # type: str
        arch,  # type: str
        source,  # type: bytes
        opt_level,  # type: int
    ):
        self.hip_source = source
        self.name = program_name.encode("utf-8")
        self.prog = None
        self.code = None
        self.code_size = None
        self.arch = arch
        self.opt_level = opt_level
        self._compile_to_executable()

    def _compile_to_executable(self):
        """
        Hint:
            Play around with the compiler ``cflags`` to and observe changes
            to the information stored in the code object.
        """
        global opt_level
        self.prog = hip_check(
            hiprtc.hiprtcCreateProgram(self.hip_source, self.name, 0, [], [])
        )
        cflags = [
            b"--offload-arch=" + self.arch.encode(),
            f"-O{opt_level}".encode(),
        ]
        (err,) = hiprtc.hiprtcCompileProgram(self.prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(self.prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(self.prog, log))
            raise RuntimeError(log.decode())
        self.code_size = hip_check(hiprtc.hiprtcGetCodeSize(self.prog))
        self.code = bytearray(self.code_size)
        hip_check(hiprtc.hiprtcGetCode(self.prog, self.code))

    def parse_metadata(self):
        assert self.code is not None
        return comgr.parse_code_obj_metadata(self.code, self.code_size)

    def parse_kernel_names(self):
        assert self.code is not None
        return comgr.parse_code_obj_kernel_names(
            self.code, self.code_size
        )

    def parse_symbols(self):
        assert self.code is not None
        return comgr.parse_code_symbols(self.code, self.code_size)

    def disassemble(self, func_name):
        """
        Note:
            If we do not pass a func_name explicitly that
            is set to the kernel name ('scale'), the
            below call will disassemble the first "FUNC" code
            object within the kernel. As there is only one
            such "FUNC" object, this has the same result
            as passing the argument ``func_name='scale'``.
        """
        assert self.code is not None
        assert self.arch is not None
        return comgr.disassemble_code_obj_function(
            self.code,
            f"amdgcn-amd-amdhsa--{self.arch}",
            func_name=func_name,
        )

    def __del__(self):
        if self.prog is not None:
            hip_check(hiprtc.hiprtcDestroyProgram(self.prog.createRef()))


if __name__ in ("__test__", "__main__"):
    import textwrap

    import yaml

    opt_level = 3
    autodetect_arch = False  # auto-detection only works if a GPU is installed
    # on your system
    arch = "gfx90a"

    print(
        textwrap.dedent(
            f"""
        ###  Parameters:

        - Optimization level: {opt_level}
        - Autodetect GPU architecture: {"yes" if autodetect_arch else "no"}
        - GPU architecture (if not autodetected): {arch}
        """
        )
    )

    kernel_hip = textwrap.dedent(
        """\
        __device__ void body(float arr[], float factor) {
            arr[threadIdx.x] *= factor;
        }

        // NOTE: extern "C" ensures that the kernel name
        //       and symbol name in the code object match
        //       as no C++ name mangling will be applied.
        extern "C" __global__ void scale(float arr[], float factor) {
            body(arr, factor);
        }

        // HINT: Add more __global__ functions and observe the
        //       changes in the in the code object.
        __global__ void  vectoradd_float(
            float* __restrict__ a,
            const float* __restrict__ b,
            const float* __restrict__ c,
            int width, int height)  {
                int x = blockDim.x * blockIdx.x + threadIdx.x;
                int y = blockDim.y * blockIdx.y + threadIdx.y;

                int i = y * width + x;
                if ( i < (width * height)) {
                    a[i] = b[i] + c[i];
                }
        }

        using dtype = long long;

        struct array2d {
            void* meminfo;
            void* parent;
            long size;
            long itemsize;
            dtype* data;
            long shape[2];
            long strides[2];
        };

        extern "C" __global__ void transposeFoo(
            array2d input,
            array2d output
        ) {
            __shared__ dtype tile[32][32 + 1];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x * blockDim.x;
            int by = blockIdx.y * blockDim.y;
            int x = by + tx;
            int y = bx + ty;

            if ( by + ty < input.shape[0] && bx + tx < input.shape[1]) {
                tile[ty][tx] = *reinterpret_cast<dtype*>(&input.data[ (by + ty)*input.strides[0] + (bx + tx)*input.strides[1] ]);
            }

            __syncthreads();

            if ( y < output.shape[0] && x < output.shape[1] ) {
                for (size_t i = 0; i < sizeof(dtype); i++) {
                    output.data[ y*output.strides[0] + x*output.strides[1] ] =
                    reinterpret_cast<char*>(&tile[tx][ty])[i];
                }
            }
        }
        """
    )

    _, num_devices = hip.hipGetDeviceCount()
    if autodetect_arch and num_devices > 0:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        arch = props.gcnArchName.decode()

    print(f"\n###  Properties of selected target (arch={arch}):\n\n```yaml")
    gpugen = arch.split(":")[0]
    print(
        yaml.dump(
            comgr.get_isa_metadata_all()[
                f"amdgcn-amd-amdhsa--{gpugen}"
            ],
            indent=2,
            sort_keys=False,
        )
    )
    print("```")

    program = HipProgram("kernel", arch, kernel_hip.encode(), opt_level)

    print("\n###  HIP C++ source:\n\n```c++")
    print(kernel_hip)
    print("```")

    print("\n###  Code object metadata:\n\n```yaml")
    print(
        yaml.dump(
            program.parse_metadata(),
            indent=2,
            sort_keys=False,
        )
    )
    print("```")

    kernel_names = program.parse_kernel_names()
    print("\n###  Code object kernel functions:\n")
    print("\n".join([f"- {k}" for k in kernel_names]))

    print("\n###  Code object symbols:\n\n```yaml")
    print(
        yaml.dump(
            program.parse_symbols(),
            indent=2,
            sort_keys=False,
        )
    )
    print("```")

    for k in kernel_names:
        print(f"\n###  Kernel '{k}' instructions:\n\n```asm")
        print(program.disassemble(k).replace("\t", "  "))
        print("```")

    print("\n###  Appendix: List of supported targets:\n")
    print("\n".join([f"- {k}" for k in comgr.get_isa_names()]))
    print("\n\nok")
