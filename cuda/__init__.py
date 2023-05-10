import sys
import types
import enum

import hip
from hip.hipify import cuda2hip, hip2cuda, IntEnum

def __create__all__(hip_mod):
    global hip2cuda
    result = []
    for hip_sym in vars(hip_mod):
        if hip_sym in hip2cuda:
            result += hip2cuda[hip_sym]
    return result

class Cuda2HipModule:

    def __init__(self,wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):

        #print(f"__getattr_(..., {name})")

        if name in self.__dict__:
            return self.__dict__[name]

        if name in cuda2hip:
            hip_obj = getattr(self.wrapped,cuda2hip[name])
            if isinstance(hip_obj,enum.EnumMeta):
                setattr(self,name,hip_obj)
            elif isinstance(hip_obj,(types.BuiltinFunctionType,types.FunctionType)):
                setattr(self,name,hip_obj)
            else:
                is_primitive = type(hip_obj) in (int,float,bool,str)
                if is_primitive:
                    setattr(self,name,hip_obj)
                else:
                    #print(type(hip_obj))
                    new_type = type(name,(hip_obj,),{})
                    setattr(self,name,new_type)
                return self.__dict__[name]
        else:
            return getattr(self.wrapped,name) # allow hip names too, e.g. for rccl

hipvars=vars(hip)

for (hip_lib,cuda_lib) in (
    ("hip","cuda"),
    ("hip","cudart"),
    ("hiprtc","nvrtc"),
    ("hiprand","curand"),
    ("hipblas","cublas"),
    ("rccl","nccl"),
    ("hipsparse","cusparse"),
    ("hipfft","cufft"),
):
    if hip_lib in hipvars:
        hip_mod = hipvars[hip_lib]
        module_obj = type(cuda_lib,(Cuda2HipModule,), {
            "__name__":cuda_lib,
            "__all__": __create__all__(hip_mod)
        })(hip_mod)
        setattr(sys.modules[__name__],cuda_lib,module_obj)
        sys.modules[f"{__name__}.{cuda_lib}"] = module_obj

del __create__all__
del hipvars
del hip
del sys
