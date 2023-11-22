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

import re
import json

def parse_hipify_perl(hipify_perl_path: str):
    """_summary_

    Args:
        hipify_perl_path (str): Path to the hipify-perl script.

    Returns:
        dict: CUDA to HIP symbol mapping.

    Note:
        Multiple HIP symbols may map to the same CUDA symbol.
        Hence, the mapping can not be directly inverted.
        Therefore, the `hip2cuda` value will be a list.
    """
    cuda2hip = {}
    hip2cuda = {}
    p_subst = re.compile(r'subst\s*\(\s*"(?P<cuda>\w+)"\s*,\s*"(?P<hip>\w+)"')
    with open(hipify_perl_path,"r") as infile:
        for ln in infile.readlines():
            for m in  p_subst.finditer(ln):
                cuda = m.group("cuda")
                hip = m.group("hip")
                cuda2hip[cuda]=hip
                if not hip in hip2cuda:
                    hip2cuda[hip]=[]
                hip2cuda[hip].append(cuda)
    return (cuda2hip, hip2cuda)

def render_hipify_perl_info(hipify_perl_path: str):
    (cuda2hip, hip2cuda) = parse_hipify_perl(hipify_perl_path)
    return "\n".join([
      f"cuda2hip={json.dumps(cuda2hip,indent=4)}",
      f"hip2cuda={json.dumps(hip2cuda,indent=4)}",
    ])

if __name__ == "__main__":
    print(render_hipify_perl_info("/opt/rocm/bin/hipify-perl"))
