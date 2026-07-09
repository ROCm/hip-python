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

Fix (landed): ``CythonModuleGenerator.walk_filtered_nodes`` admits a
nested record/enum (and inline anonymous function pointer) transitively
whenever its TOP-MOST enclosing declaration is admitted by
``node_filter`` — see ``_topmost_ancestor``. This closes the gap in the
codegen tool itself, so recipes no longer need a per-library
``_topmost_name`` filter (the ``amdsmi`` recipe keeps one, now redundant
but harmless).
"""

import re

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


def test_named_nested_struct_definition_emitted_when_parent_admitted(tmp_path):
    """A named-nested struct whose own name lacks the filter prefix is
    still emitted, because its TOP-MOST enclosing type (``my_bdf_t``) is
    admitted.

    Previously an ``xfail(strict=True)`` gap: the outer typedef was
    admitted but the inner ``struct bdf_`` was dropped while the outer
    type still referenced it via the synthesised ``<parent>_<inner>``
    name. Closed by the top-most-ancestor transitive-admission rule in
    ``CythonModuleGenerator.walk_filtered_nodes`` — nested records/enums
    (and inline anonymous function pointers) inherit their top-most
    enclosing declaration's admission, so no per-recipe ``_topmost_name``
    filter is required.
    """
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


UNDERSCORE_TAG_HEADER = """
typedef struct _hipFooAlgo_t {
    unsigned long data[16];
    unsigned long max_workspace_bytes;
} hipFooAlgo_t;

int hipFooSubmit(const hipFooAlgo_t *algo);
"""


def _hipfoo_prefix_filter(node):
    """Mimics per-library `node_filter` shape: prefix-only on the node's own
    name. Crucially, an underscore-prefixed struct tag like
    ``_hipFooAlgo_t`` is REJECTED while its typedef alias ``hipFooAlgo_t``
    is ACCEPTED — the same shape as `class hipblaslt.node_filter` in
    `support/recipes/rocm.py`.
    """
    if isinstance(node, MacroDefinition):
        return False
    return (node.name or "").startswith(("hipFoo", "_hipFoo")) and not (
        node.name or ""
    ).startswith("_")  # explicit: leading-underscore tags are rejected


def test_underscore_tagged_struct_admitted_via_typedef_referent(tmp_path):
    """When a `typedef struct _Tag { ... } T;` is admitted via its alias
    name (`T`) but the underlying tag (`_Tag`) is rejected by the same
    prefix filter, the codegen must still emit a `cdef struct _Tag:`
    block — otherwise `ctypedef _Tag T` and any function signature
    referencing `_Tag *` are dangling identifiers and Cython compilation
    fails with `'_Tag' is not a type identifier`.

    This locks in the transitive-admit rule in
    ``CythonModuleGenerator._transitively_admitted_records``: a Record
    named as the ``Typedef.typeref`` of an admitted typedef is admitted
    even if the user `node_filter` rejects it on its own name.
    """
    gen = make_generator(
        UNDERSCORE_TAG_HEADER,
        module_name="mod_u",
        node_filter=_hipfoo_prefix_filter,
    )
    files = write_module(gen, tmp_path)
    pxd = files["cymod_u.pxd"]

    # The struct definition for the underscore-tagged type must be present
    # so Cython can resolve `ctypedef _hipFooAlgo_t hipFooAlgo_t` and the
    # `const _hipFooAlgo_t *algo` parameter in `hipFooSubmit`.
    assert re.search(r"\bcdef\s+struct\s+_hipFooAlgo_t\b", pxd), (
        f"expected `cdef struct _hipFooAlgo_t:` definition; full pxd:\n{pxd}"
    )
    # And the typedef alias must point at it.
    assert re.search(r"ctypedef\s+_hipFooAlgo_t\s+hipFooAlgo_t", pxd), (
        f"expected `ctypedef _hipFooAlgo_t hipFooAlgo_t`; full pxd:\n{pxd}"
    )
    # Source-order matters: the struct definition must precede the typedef.
    struct_pos = pxd.find("cdef struct _hipFooAlgo_t")
    typedef_pos = pxd.find("ctypedef _hipFooAlgo_t hipFooAlgo_t")
    assert struct_pos != -1 and typedef_pos != -1 and struct_pos < typedef_pos, (
        f"struct definition must come before its typedef alias; "
        f"struct at {struct_pos}, typedef at {typedef_pos}; full pxd:\n{pxd}"
    )


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
