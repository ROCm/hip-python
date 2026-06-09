# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Lock-down test: prefix-stripping renamer doesn't mangle parm names.

The amdsmi recipe TODO comment (rocm.py:805-828) flags a risk: "Renamer
over-strips when a parameter name matches part of the return type
(e.g. `amdsmi_processor_type_t` → `processor_type_t`)." There is no current
crash — this test pins down the today-correct behavior so that:

* future contributors can adopt a prefix-stripping renamer in the amdsmi
  class with confidence, and
* a regression in ``interfacegen.typerender.render`` (the typeref
  substitution path that splices the renamed identifier in place of the
  elaborated ``struct``/``enum``/``union`` leaf) surfaces here.

The header below is the exact ``processor_type`` shape the comment calls
out: function name, return type, and parm name all share the
``processor_type`` token. After ``re.sub(r"^amdsmi_", "", name)``, the
renamer must:

* rename the typedef ``amdsmi_processor_type_t`` → ``processor_type_t``;
* rename the function ``amdsmi_get_processor_type`` → ``get_processor_type``;
* leave the parm name ``processor_type`` UNCHANGED (it has no ``amdsmi_``
  prefix to strip);
* not mangle the inner ``processor_type`` token of the typedef.
"""

import re
import textwrap

import pytest

from _codegen_helpers import make_generator, write_module


HEADER = """
typedef int amdsmi_processor_type_t;
amdsmi_processor_type_t amdsmi_get_processor_type(int *processor_type);
"""


def _strip_amdsmi(name: str) -> str:
    return re.sub(r"^amdsmi_", "", name)


def test_prefix_strip_renamer_preserves_parm_name(tmp_path):
    gen = make_generator(HEADER, module_name="mod_r", renamer=_strip_amdsmi)
    files = write_module(gen, tmp_path)
    cym_pxd = files["cymod_r.pxd"]

    # 1. typedef rename worked
    assert re.search(r"\bctypedef\s+int\s+processor_type_t\b", cym_pxd), (
        f"expected `ctypedef int processor_type_t`, full pxd:\n{cym_pxd}"
    )

    # 2. function rename worked
    assert re.search(
        r"\bint\s+get_processor_type\s*\(", cym_pxd
    ), f"expected `int get_processor_type(...)`, full pxd:\n{cym_pxd}"

    # 3. parm name was NOT mangled — should still be `processor_type`
    sig_match = re.search(
        r"int\s+get_processor_type\s*\(([^)]*)\)", cym_pxd
    )
    assert sig_match, "could not find function signature in cymod_r.pxd"
    parms = sig_match.group(1)
    assert "processor_type" in parms, (
        f"parm name lost / mangled — full signature parms: {parms!r}"
    )
    # And the parm name must be a STANDALONE token, not a substring of a
    # mangled name.
    assert re.search(r"\bprocessor_type\b", parms), (
        f"parm name appears mangled (not a standalone token): {parms!r}"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("amdsmi_processor_type_t", "processor_type_t"),
        ("amdsmi_get_processor_type", "get_processor_type"),
        ("AMDSMI_MAX_DEVICES", "AMDSMI_MAX_DEVICES"),  # different prefix — leave
        ("processor_type", "processor_type"),  # no prefix — leave
        ("amdsmi_amdsmi_double", "amdsmi_double"),  # only one ^amdsmi_ stripped
    ],
)
def test_prefix_strip_renamer_unit(name, expected):
    """Pure unit test of the renamer logic — no codegen involved."""
    assert _strip_amdsmi(name) == expected
