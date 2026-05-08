# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

"""Smoke-test the AMD SMI binding: init, query library version, shut down.

Mirrors the lifecycle portion of the nodrm C example
(`rocm-systems/projects/amdsmi/example/amd_smi_nodrm_example.cc`).
Socket / processor enumeration is not exercised here because the
codegen's handling of the C two-call (count → allocate → fill) pattern
for opaque-handle arrays is incomplete in this iteration of the
amdsmi profile. See the `_WHITELIST_*` notes in
`interfacegen.support.recipes.rocm.amdsmi` for the current scope.
"""

# [literalinclude-begin]
from rocm.bindings import amdsmi


def amdsmi_check(call_result):
    if isinstance(call_result, amdsmi.amdsmi_status_t):
        err, result = call_result, ()
    else:
        err, result = call_result[0], call_result[1:]
    if err != amdsmi.amdsmi_status_t.AMDSMI_STATUS_SUCCESS:
        raise RuntimeError(str(err))
    if len(result) == 1:
        return result[0]
    return result


amdsmi_check(amdsmi.amdsmi_init(amdsmi.amdsmi_init_flags_t.AMDSMI_INIT_AMD_GPUS))
try:
    version = amdsmi_check(amdsmi.amdsmi_get_lib_version())
    print(
        f"AMD SMI library version: "
        f"{version.major}.{version.minor}.{version.release}"
    )
finally:
    amdsmi_check(amdsmi.amdsmi_shut_down())
print("ok")
