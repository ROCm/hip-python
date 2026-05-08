# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
"""Gap test: named-nested struct gets dropped when node_filter is prefix-based.

Real-world hit (from amdsmi.h:902-915):

    typedef union {
      struct bdf_ {                  // ← nested AND named "bdf_"
        uint64_t function_number : 3;
        ...
      } bdf;
      ...
    } amdsmi_bdf_t;

The amdsmi recipe's `node_filter` admits only `amdsmi_*` / `AMDSMI_*` names.
The outer `amdsmi_bdf_t` passes; the inner `struct bdf_` does NOT (its name
is just `bdf_`). Result in cyamdsmi.pxd:

    cdef union amdsmi_bdf_t:
        amdsmi_bdf_t_bdf_ bdf      # ← reference to the synthesized name
        ...
    # nowhere in the file: cdef struct amdsmi_bdf_t_bdf_:

The reference uses the codegen's synthesised qualified name
``<parent>_<inner>`` but the inner-struct definition was dropped at filter
time.

There's a second symptom in the same union: an *anonymous* sibling struct
(also bit-fielded `uint64_t function_number : 3` … but with no field name)
is rendered with a broken Cython identifier
(``cdef struct my_bdf_t_struct (anonymous at input.h:N:M):``) and the union
loses its reference to that field entirely. That symptom is documented but
this test focuses on the named-nested gap because it's the one tied to
``node_filter``.

Fix shape (sketched in `share/design/POINTER_ARGUMENTS.md` follow-ups):
walk a record's nested-non-anonymous types when their parent is admitted
by ``node_filter``, and either (a) admit them transitively or (b) emit them
as opaque/forwarded stubs above the parent's ``cdef extern from`` block.
"""

import re

import pytest

from interfacegen.tree import MacroDefinition

from _codegen_helpers import make_generator, write_module


HEADER = """
typedef union {
    struct bdf_ {
        unsigned long function_number;
        unsigned long device_number;
        unsigned long bus_number;
    } bdf;
    unsigned long as_uint;
} my_bdf_t;
"""


def _prefix_only_filter(node):
    """Mimics amdsmi's `node_filter`: prefix-only on top-level names."""
    if isinstance(node, MacroDefinition):
        return False
    return (node.name or "").startswith("my_bdf_t")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "When node_filter admits the outer typedef but not the named-nested "
        "inner struct (because its name lacks the prefix), the inner-struct "
        "definition is dropped while the outer type still references it via "
        "a synthesised <parent>_<inner> name. Fix: hoist + admit nested-named "
        "types transitively when their parent is admitted."
    ),
)
def test_named_nested_struct_definition_emitted_when_parent_admitted(tmp_path):
    gen = make_generator(
        HEADER, module_name="mod_n", node_filter=_prefix_only_filter
    )
    files = write_module(gen, tmp_path)
    pxd = files["cymod_n.pxd"]

    # All `<typename> bdf` field references in the file:
    refs = re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+bdf\b", pxd, re.MULTILINE)
    assert refs, "expected at least one field reference to the inner `bdf`"

    missing = []
    for ref in set(refs):
        if ref in ("unsigned", "long", "int"):
            continue
        pattern = rf"\bcdef\s+(?:struct|union|enum)\s+{re.escape(ref)}\b"
        if not re.search(pattern, pxd):
            missing.append(ref)
    assert not missing, (
        f"these synthesised type names are referenced but never defined: "
        f"{missing}\n--- emitted cymod_n.pxd ---\n{pxd}"
    )


def test_named_nested_struct_works_with_permissive_filter(tmp_path):
    """Sanity check: with a permissive filter, the inner struct IS emitted.

    Locks down the today-working case so a regression in the hoist path
    surfaces here even before the gap fix lands.
    """
    gen = make_generator(HEADER, module_name="mod_n_ok")  # default = admit all
    files = write_module(gen, tmp_path)
    pxd = files["cymod_n_ok.pxd"]
    assert re.search(r"\bcdef\s+struct\s+my_bdf_t_bdf_\b", pxd), (
        f"expected `cdef struct my_bdf_t_bdf_:` definition; full pxd:\n{pxd}"
    )


def _topmost_name(node):
    """Mirror of `support.recipes.rocm.amdsmi._topmost_name` — walks up
    to the outermost non-Root ancestor. Recipe authors should use this
    pattern (or call the recipe's helper) rather than the naive
    `node.name`-only filter to avoid dropping named-nested types."""
    from interfacegen import tree
    curr = node
    while curr.parent is not None and not isinstance(curr.parent, tree.Root):
        curr = curr.parent
    return curr.name or ""


def _topmost_prefix_filter(node):
    if isinstance(node, MacroDefinition):
        return False
    return _topmost_name(node).startswith("my_bdf_t")


def test_named_nested_struct_emitted_with_topmost_parent_filter(tmp_path):
    """Recipe-level workaround: when ``node_filter`` walks up to the
    topmost parent for prefix matching, the inner ``bdf_`` struct
    inherits ``my_bdf_t``'s admission and gets emitted alongside it.

    This is the pattern that the ``amdsmi`` recipe uses
    (``support.recipes.rocm.amdsmi._topmost_name``). The codegen-tool
    gap exercised by the strict-xfail test above remains real for
    recipes that don't walk the parent chain, but recipes that do
    sidestep it.
    """
    gen = make_generator(
        HEADER, module_name="mod_n_walk", node_filter=_topmost_prefix_filter
    )
    files = write_module(gen, tmp_path)
    pxd = files["cymod_n_walk.pxd"]

    # The named-nested struct definition must be present.
    assert re.search(r"\bcdef\s+struct\s+my_bdf_t_bdf_\b", pxd), (
        f"expected `cdef struct my_bdf_t_bdf_:` definition; full pxd:\n{pxd}"
    )
    # And the parent union must reference it without dangling.
    assert re.search(r"my_bdf_t_bdf_\s+bdf\b", pxd), (
        f"expected union to reference `my_bdf_t_bdf_ bdf`; full pxd:\n{pxd}"
    )
