# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Tests for the CUDA-to-HIP symbol mapping read out of ``hipify-perl``.

HIPIFY has rewritten how it spells its mapping tables several times, most
recently in ROCm 10.1, and each rewrite makes the parser return an empty map
for a script that is present and perfectly readable. The CUDA interop layer
is generated from that map, so the whole wheel set goes missing on a format
the parser does not know -- late, and in a build that needs a ROCm install to
reproduce. One line per format here catches the next rewrite instead.
"""

import pytest
from hip_python_codegen.hipify import parse_hipify_perl

# One line per format HIPIFY has emitted, each mapping the symbol the interop
# layer anchors on.
FORMATS = {
    "subst": 'subst("cudaRuntimeGetVersion", "hipRuntimeGetVersion");',
    "mappings_hash": (
        '$mappings{"cudaRuntimeGetVersion"} = '
        '{rep => "hipRuntimeGetVersion", type => "version"};'
    ),
    "k_call": (  # ROCm 7.14.0+
        'k("cudaRuntimeGetVersion", "hipRuntimeGetVersion", "version");'
    ),
    "fat_comma": (  # ROCm 10.1.0+
        "'cudaRuntimeGetVersion' => ['hipRuntimeGetVersion', 'version'],"
    ),
}


def write_hipify_perl(tmp_path, *lines):
    path = tmp_path / "hipify-perl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("line", FORMATS.values(), ids=list(FORMATS))
def test_every_emitted_format_is_understood(tmp_path, line):
    (cuda2hip, hip2cuda) = parse_hipify_perl(write_hipify_perl(tmp_path, line))

    assert cuda2hip == {"cudaRuntimeGetVersion": "hipRuntimeGetVersion"}
    assert hip2cuda == {"hipRuntimeGetVersion": ["cudaRuntimeGetVersion"]}


def test_formats_may_be_mixed_in_one_script(tmp_path):
    (cuda2hip, _) = parse_hipify_perl(
        write_hipify_perl(
            tmp_path,
            'subst("cudaFree", "hipFree");',
            "'cudaMalloc' => ['hipMalloc', 'memory'],",
            'k("cudaMemcpy", "hipMemcpy", "memory");',
        )
    )

    assert cuda2hip == {
        "cudaFree": "hipFree",
        "cudaMalloc": "hipMalloc",
        "cudaMemcpy": "hipMemcpy",
    }


def test_several_hip_symbols_may_share_a_cuda_symbol(tmp_path):
    (_, hip2cuda) = parse_hipify_perl(
        write_hipify_perl(
            tmp_path,
            "'cudaStreamAttrValue' => ['hipStreamAttrValue', 'stream'],",
            "'cudaStreamAttrValue' => ['hipKernelNodeAttrValue', 'stream'],",
        )
    )

    assert hip2cuda == {
        "hipStreamAttrValue": ["cudaStreamAttrValue"],
        "hipKernelNodeAttrValue": ["cudaStreamAttrValue"],
    }
