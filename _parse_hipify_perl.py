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
