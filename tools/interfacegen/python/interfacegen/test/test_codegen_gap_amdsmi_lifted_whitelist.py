# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Integration: lift the amdsmi whitelist and try to bind the full surface.

This is the single-signal end-to-end test for the long-term goal in the
``amdsmi`` recipe TODO comment (rocm.py:805-828): "Once interfacegen.cython
grows support for those patterns, the whitelist below should be lifted in
favour of the prefix-only filter."

Today the whole flow is xfail-strict: lifting the filter exposes the four
codegen-tool gaps that have their own focused tests in this directory
(incomplete-array-of-record OUT param, void-typedef array field, named-nested
struct hoist, renamer over-strip). When all four land their fixes, this
xfail flips and we can delete the recipe's ``_WHITELIST_*`` sets.

Skipped without ``/opt/rocm/include/amd_smi/amdsmi.h`` so non-ROCm dev boxes
don't see a hard failure.
"""

import os
import shutil
import subprocess
import sys

import pytest


AMDSMI_HEADER = "/opt/rocm/include/amd_smi/amdsmi.h"
AMDSMI_INCLUDE_DIR = "/opt/rocm/include"


def _has_libclang_resource_dir():
    return os.path.isfile("/opt/rocm/lib/llvm/lib/libclang.so")


pytestmark = pytest.mark.skipif(
    not (os.path.exists(AMDSMI_HEADER) and _has_libclang_resource_dir()),
    reason="needs ROCm headers + libclang at /opt/rocm",
)


@pytest.fixture(scope="module")
def codegen_module_path():
    """Add the hip-python codegen recipe to sys.path for the duration of
    these tests. Skip if it isn't checked out next to interfacegen."""
    candidate = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "recipes", "hip-python", "src",
        )
    )
    if not os.path.isdir(os.path.join(candidate, "hip_python_codegen")):
        pytest.skip(f"hip_python_codegen not found at {candidate}")
    sys.path.insert(0, candidate)
    yield candidate
    sys.path.remove(candidate)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Whole-surface amdsmi binding requires four codegen-tool fixes "
        "(see test_codegen_gap_*.py). Flip when all of them land."
    ),
)
def test_amdsmi_lifted_whitelist_compiles(tmp_path, codegen_module_path):
    from hip_python_codegen import generators_systems
    from interfacegen.support.recipes import rocm
    from interfacegen.tree import MacroDefinition

    def lifted_filter(node):
        nm = node.name or ""
        if isinstance(node, MacroDefinition):
            if nm in rocm.amdsmi._SKIPPED_MACROS:
                return False
            return nm.startswith("AMDSMI_") or nm.startswith("amdsmi_")
        return nm.startswith("amdsmi_") or nm.startswith("AMDSMI_")

    # Monkey-patch — restored by the fixture exit.
    original = rocm.amdsmi.node_filter
    rocm.amdsmi.node_filter = staticmethod(lifted_filter)
    try:
        gen = generators_systems.generate_amdsmi(
            include_dir=AMDSMI_INCLUDE_DIR,
            runtime_linking=True,
            generator_args=[
                "-I", AMDSMI_INCLUDE_DIR,
                "-resource-dir=/opt/rocm/lib/llvm/lib/clang/23",
            ],
            default_ptr_handler=lambda parm: "rocm.bindings.util.types.Pointer",
        )
        gen.write_module_files(str(tmp_path))
    finally:
        rocm.amdsmi.node_filter = original

    # Try to compile the emitted cyamdsmi.pxd — if codegen succeeded, the
    # remaining shape problems (void *T[N] field, etc.) surface as Cython
    # syntax errors here.
    cython = shutil.which("cython") or sys.executable
    cmd = [sys.executable, "-m", "cython", "--3str",
           os.path.join(str(tmp_path), "cyamdsmi.pxd")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"cython compile failed:\n--- stdout ---\n{res.stdout}\n"
        f"--- stderr ---\n{res.stderr}"
    )
