import ctypes
from hip import hip, hiprtc

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err,hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif isinstance(err,hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(str(err))
    return result

source = b"""\
extern "C" __global__ void set(int *a) {
  *a = 10;
}
"""

prog = hip_check(hiprtc.hiprtcCreateProgram(source, b'set', 0, [], []))
print(f"{hex(prog.ptr)=}")

#props = hip.hipDeviceProp_t()
#hip_check(hip.hipGetDeviceProperties(props,0))
#arch = props.gcnArchName
arch = "gfx90a"

print(f"Compiling kernel for {arch}")

hip_check(hiprtc.hiprtcCompileProgram(prog, 1, [f'--offload-arch={arch}'.encode()]))
code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
print(code_size)
code = bytearray(code_size)
hip_check(hiprtc.hiprtcGetCode(prog,code))
print(code)
#module = hip_check(hip.hipModuleLoadData(code))
#kernel = hip_check(hip.hipModuleGetFunction(module, 'set'))
#ptr = hip.hipMalloc(4)

#class PackageStruct(ctypes.Structure):
#  _fields_ = [("a", ctypes.c_void_p)]
#
#struct = PackageStruct(ptr)
#hip.hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, struct)
#res = ctypes.c_int(0)
#hip.hipMemcpy_dtoh(ctypes.byref(res), ptr, 4)
#print(res.value)
