# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression test for the synthesized AnonymousFunctionPointer
``ctypedef`` emission gap that broke the HSA bindings.

When a C function takes an inline function-pointer parameter (no
intermediate typedef on the C side), the codegen synthesizes a
typedef-style name shaped like ``<parent>_anon_funptr_<N>`` and uses
it as the parameter type in the rendered ``cdef`` declaration.
Recipe-side ``node_filter`` predicates use a strict prefix match on
the symbol name (e.g. ``startswith("hsa_")``); the synthesized
``anon_funptr_0`` has no such prefix and gets rejected, but the
parent function's rendering still names the type — leaving Cython
with an undeclared identifier that defaults to "Python object" and
later fails compilation with
``Converting to Python object not allowed without gil``.

The fix transitively admits AnonymousFunctionPointer when the
enclosing parent decl is admitted; this test pins that behavior
against an HSA-shaped synthetic header.
"""

import textwrap

import pytest

from interfacegen.support.recipes import control
from interfacegen.test._codegen_helpers import make_generator, write_module


def _hsa_prefix_filter(node):
    """Strict-prefix node_filter mirroring `recipes.rocm.hsa.node_filter`
    — accepts only `hsa_*` / `HSA_*` symbols. Used to reproduce the
    real-world rejection of unprefixed synthesized names."""
    name = node.name
    return name.startswith("hsa_") or name.startswith("HSA_")


def test_anon_funptr_typedef_is_emitted_when_parent_admitted(tmp_path):
    """`hsa_iterate_agents` shape: a function taking an inline
    callback parameter typed as `T (*)(...)`. The strict `hsa_*`
    prefix filter rejects the synthesized AnonymousFunctionPointer
    name directly, but the parent function IS admitted — so the
    `ctypedef` must be emitted via transitive admission."""
    src = textwrap.dedent(
        """\
        typedef int hsa_status_t;
        typedef int hsa_agent_t;
        hsa_status_t hsa_iterate_agents(
            hsa_status_t (*callback)(hsa_agent_t agent, void* data),
            void* data);
        """
    )
    gen = make_generator(
        src,
        module_name="hsa",
        node_filter=_hsa_prefix_filter,
        modifiers_lazy_loader=" noexcept nogil",
    )
    files = write_module(gen, tmp_path)
    pxd = files.get("cyhsa.pxd", "")
    assert pxd, f"expected cyhsa.pxd in {list(files)}"
    # The parent function's signature must reference the synthesized
    # type by its full `<parent>_anon_funptr_<N>` name.
    assert "hsa_iterate_agents_anon_funptr_0" in pxd, (
        "parent function should reference the synthesized funptr type"
    )
    # And — the actual fix — the matching ctypedef must be emitted so
    # Cython can resolve the type at compile time.
    assert "ctypedef" in pxd
    assert (
        "ctypedef hsa_status_t (*hsa_iterate_agents_anon_funptr_0)" in pxd
    ), (
        "AnonymousFunctionPointer ctypedef missing — Cython will fail "
        "with 'Converting to Python object not allowed without gil' on "
        "the parent function's rendered declaration"
    )


def test_anon_funptr_typedef_skipped_when_parent_rejected(tmp_path):
    """If the enclosing function is itself rejected by the recipe
    filter, the synthesized funptr type is dead code — the parent
    decl never gets emitted, so its referent need not exist either.
    Negative case to make sure the transitive-admission rule doesn't
    leak typedefs from filtered-out scopes."""
    src = textwrap.dedent(
        """\
        typedef int hsa_status_t;
        typedef int hsa_agent_t;
        /* No hsa_ prefix — should be filtered out entirely. */
        hsa_status_t other_iterate(
            hsa_status_t (*callback)(hsa_agent_t agent, void* data),
            void* data);
        """
    )
    gen = make_generator(
        src,
        module_name="hsa",
        node_filter=_hsa_prefix_filter,
        modifiers_lazy_loader=" noexcept nogil",
    )
    files = write_module(gen, tmp_path)
    pxd = files.get("cyhsa.pxd", "")
    assert "other_iterate" not in pxd
    assert "other_iterate_anon_funptr_0" not in pxd


def test_anon_funptr_typedef_in_struct_field_admitted_transitively(tmp_path):
    """Same fix needs to cover the FIELD_DECL case too: an inline
    function-pointer field inside an admitted struct. The
    AnonymousFunctionPointer's parent there is the Struct, not a
    Function; the same transitive-admission predicate must apply."""
    src = textwrap.dedent(
        """\
        typedef struct hsa_dispatch_callbacks_s {
            int (*on_dispatch)(int packet, void* user);
            void* user_data;
        } hsa_dispatch_callbacks_t;
        """
    )
    gen = make_generator(
        src,
        module_name="hsa",
        node_filter=_hsa_prefix_filter,
        modifiers_lazy_loader=" noexcept nogil",
    )
    files = write_module(gen, tmp_path)
    pxd = files.get("cyhsa.pxd", "")
    assert pxd, f"expected cyhsa.pxd in {list(files)}"
    # The synthesized name is parented to the enclosing Struct
    # (`hsa_dispatch_callbacks_s`), not the FIELD_DECL — the
    # AnonymousFunctionPointer is appended as a sibling of the Field
    # under the Struct, per `treefactory.handle_param_or_field_decl_cursor_`.
    assert "hsa_dispatch_callbacks_s_anon_funptr_0" in pxd
    assert (
        "ctypedef int (*hsa_dispatch_callbacks_s_anon_funptr_0)"
        in pxd
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
