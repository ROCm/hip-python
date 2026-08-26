# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import json
import re


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

    # Examples:
    # old format 1: subst("cudaFuncSetAttribute", "hipFuncSetAttribute")
    # old format 2: $mappings{"cudaFuncSetAttribute"} = {rep => "hipFuncSetAttribute", type => "execution"};
    # latest format  (ROCm 7.14.0+): k("cudaFuncSetAttribute", "hipFuncSetAttribute", "execution");
    p_mapping_str = "|".join(
        [
            r'(subst\s*\(\s*"(?P<cuda>\w+)"\s*,\s*"(?P<hip>\w+)")',
            r'(\$mappings\{"(?P<cuda2>\w+)"\}\s*=\s*\{\s*rep\s*=>\s*"(?P<hip2>\w+)")',
            r'(k\s*\(\s*"(?P<cuda3>\w+)"\s*,\s*"(?P<hip3>\w+)"\s*,\s*"(?P<type3>\w+)")',
        ]
    )
    # print(p_mapping_str)
    p_mapping = re.compile(p_mapping_str)

    with open(hipify_perl_path, "r", encoding="utf-8") as infile:
        for ln in infile.readlines():
            for m in p_mapping.finditer(ln):
                cuda = m.group("cuda") or m.group("cuda2") or m.group("cuda3")
                hip = m.group("hip") or m.group("hip2") or m.group("hip3")
                cuda2hip[cuda] = hip
                if hip not in hip2cuda:
                    hip2cuda[hip] = []
                hip2cuda[hip].append(cuda)
    return (cuda2hip, hip2cuda)


def render_hipify_perl_info(hipify_perl_path: str):
    (cuda2hip, hip2cuda) = parse_hipify_perl(hipify_perl_path)
    return "\n".join(
        [
            f"cuda2hip={json.dumps(cuda2hip,indent=4)}",
            f"hip2cuda={json.dumps(hip2cuda,indent=4)}",
        ]
    )


if __name__ == "__main__":
    print(render_hipify_perl_info("/opt/rocm/bin/hipify-perl"))
