"""Regression tests for Part 12a — explicit `@ingroup` → `@defgroup`
display-title + brief fallback in `_render_doxygen_brief`.

Per doxygen spec, `@defgroup <ID> <Display Title>` separates the
unique ID (first token) from the human-readable display title (rest
of the line). `@ingroup <ID>` looks up by ID and inherits the
display title, so the attribution prefix should read
`[group: <Display Title>] …`, not `[group: <ID>] …`.
"""

import textwrap

import pytest

from interfacegen import doxyparser
from interfacegen.cython import DOXYGEN_CONV, DoxygenMixin
from interfacegen.cparser import CParser


# --- _ingroup_names_from_raw ---------------------------------------------


def test_ingroup_names_single():
    raw = "/** \\ingroup memory\n *  \\brief alloc */"
    assert DoxygenMixin._ingroup_names_from_raw(raw) == ["memory"]


def test_ingroup_names_multiple_in_source_order():
    raw = (
        "/**\n"
        " *  @ingroup first\n"
        " *  @ingroup second\n"
        " *  @brief Belongs to two groups\n"
        " */"
    )
    assert DoxygenMixin._ingroup_names_from_raw(raw) == ["first", "second"]


def test_ingroup_names_empty_when_absent():
    assert (
        DoxygenMixin._ingroup_names_from_raw("/** \\brief no group */") == []
    )


def test_ingroup_names_empty_when_raw_is_none():
    assert DoxygenMixin._ingroup_names_from_raw("") == []


def test_ingroup_names_multi_id_on_one_line():
    """Per the doxygen spec, `\\ingroup x y z` puts the entity in 3
    groups. Each id should be returned in source order."""
    raw = "/** \\ingroup foo bar baz\n *  \\brief multi */"
    assert DoxygenMixin._ingroup_names_from_raw(raw) == ["foo", "bar", "baz"]


def test_ingroup_names_multi_id_strips_trailing_comment_delim():
    """`\\ingroup foo */` — the trailing `*/` is not an id."""
    raw = "/** \\ingroup foo */"
    assert DoxygenMixin._ingroup_names_from_raw(raw) == ["foo"]


# --- _build_group_index --------------------------------------------------


class _FakeRoot:
    """Minimal Root stand-in for _build_group_index tests."""

    def __init__(self, tu):
        self.cursor = tu.cursor

    def walk(self, postorder=False):
        yield from _walk(self.cursor)


def _walk(cursor):
    """Pre-order walk yielding stand-in nodes whose `raw_comment`
    attribute mimics tree.Node's interface."""
    for child in cursor.get_children():
        if child.raw_comment:
            yield _FakeNode(child)
        yield from _walk(child)


class _FakeNode:
    def __init__(self, cursor):
        self.raw_comment = cursor.raw_comment


def _parse_and_index(src: str):
    parser = CParser("input.h", unsaved_files=[("input.h", src)])
    parser.parse()
    return DoxygenMixin._build_group_index(_FakeRoot(parser.translation_unit))


def test_build_group_index_extracts_defgroup_id_and_display_title():
    src = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management
         *
         *  Operations on device-side memory.
         */
        int placeholder(void);
        """
    )
    idx = _parse_and_index(src)
    assert "memory" in idx
    title, brief = idx["memory"]
    assert title == "Memory Management"
    assert "Operations on device-side memory." in brief


def test_build_group_index_uses_explicit_brief_when_present():
    src = textwrap.dedent(
        """\
        /**
         *  \\defgroup signals Signals
         *  \\brief Wait/notify primitives.
         *
         *  Longer body text follows.
         */
        int placeholder(void);
        """
    )
    idx = _parse_and_index(src)
    assert idx["signals"] == ("Signals", "Wait/notify primitives.")


def test_build_group_index_skips_untitled_addtogroup():
    """`\\addtogroup id` without a title argument cannot establish a
    display title — it just appends to whatever `\\defgroup` defined
    elsewhere. With no defgroup in the TU the id stays out of the
    index."""
    src = textwrap.dedent(
        """\
        /**
         *  \\addtogroup wherever
         *  \\brief Should be ignored as a defgroup source.
         */
        int placeholder(void);
        """
    )
    idx = _parse_and_index(src)
    assert "wherever" not in idx


def test_build_group_index_addtogroup_with_title_fills_in_when_no_defgroup():
    """`\\addtogroup id title` IS a fallback title source per the
    doxygen group priority hierarchy: when no `\\defgroup` defines
    the id, addtogroup's optional title argument provides the display
    title. ROCm's prefixsums / generalized_identity_operations / ~70
    other group ids are reachable only via this rule."""
    src = textwrap.dedent(
        """\
        /**
         *  \\addtogroup prefixsums Prefix Sums
         *  \\brief Inclusive and exclusive scan algorithms.
         */
        int placeholder(void);
        """
    )
    idx = _parse_and_index(src)
    assert "prefixsums" in idx
    title, brief = idx["prefixsums"]
    assert title == "Prefix Sums"
    assert "Inclusive and exclusive scan algorithms." in brief


def test_build_group_index_defgroup_wins_over_addtogroup():
    """Priority hierarchy: when both `\\defgroup id title1` and
    `\\addtogroup id title2` exist for the same id, defgroup wins."""
    src = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management (defgroup)
         *  \\brief Operations on device-side memory.
         */
        int defgroup_anchor(void);
        /**
         *  \\addtogroup memory Memory Management (addtogroup)
         *  \\brief Should NOT replace the defgroup brief.
         */
        int addtogroup_anchor(void);
        """
    )
    idx = _parse_and_index(src)
    title, brief = idx["memory"]
    assert title == "Memory Management (defgroup)"
    assert "Operations on device-side memory." in brief


def test_build_group_index_addtogroup_with_at_brace_marker():
    """`\\addtogroup id title @{` (the `@{` opens a group block) — the
    title regex must stop at `@{` and not capture it. ROCm uses this
    shape extensively (e.g. `\\addtogroup memory Memory Management @{`)."""
    src = textwrap.dedent(
        """\
        /** \\addtogroup memory Memory Management @{ */
        int member1(void);
        int member2(void);
        /** @} */
        """
    )
    idx = _parse_and_index(src)
    assert "memory" in idx
    title, _brief = idx["memory"]
    assert title == "Memory Management"
    assert "@{" not in title


# --- _render_doxygen_brief integration ----------------------------------


class _FakeHostNode:
    """Stand-in for a `tree.Function` carrying a raw_comment with
    `@ingroup` annotations and a known root for index lookup."""

    def __init__(self, raw_comment, root):
        self.raw_comment = raw_comment
        self._root = root

    def get_root(self):
        return self._root


def _render_brief(symbol_raw, defgroup_src, missing_text="(MISSING)"):
    """Helper: parse `defgroup_src` to build the group index, then
    render the brief for a symbol whose raw_comment is `symbol_raw`."""
    parser = CParser("input.h", unsaved_files=[("input.h", defgroup_src)])
    parser.parse()
    root = _FakeRoot(parser.translation_unit)
    sections = list(DOXYGEN_CONV.parse_structure(symbol_raw).children)
    host = _FakeHostNode(symbol_raw, root)
    return DoxygenMixin._render_doxygen_brief(
        sections,
        missing_text=missing_text,
        host_node=host,
    )


def test_render_brief_uses_display_title_in_attribution_prefix():
    defgroup = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management
         *
         *  Operations on device-side memory.
         */
        int placeholder(void);
        """
    )
    # Symbol's "raw" comment carries only `@ingroup memory` — no
    # @brief, no body — so the brief renderer falls through to the
    # group fallback.
    out = _render_brief("@ingroup memory\n", defgroup)
    assert "[group: Memory Management]" in out
    assert (
        "[group: memory]" not in out
    ), "attribution must use the display title, not the ID"


def test_render_brief_picks_first_resolving_group_for_multi_ingroup():
    defgroup = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management
         *  \\brief Operations on device-side memory.
         */
        int placeholder(void);
        """
    )
    raw = "@ingroup unknown\n@ingroup memory\n"
    out = _render_brief(raw, defgroup)
    # `unknown` is undefined → skipped; `memory` resolves.
    assert "Memory Management" in out


def test_render_brief_falls_back_to_missing_when_no_group_resolves():
    defgroup = "int placeholder(void);\n"
    raw = "@ingroup nonexistent\n"
    out = _render_brief(raw, defgroup, missing_text="MISSING")
    assert "MISSING" in out
    assert "[group:" not in out


def test_render_brief_prefers_explicit_brief_over_group_brief():
    defgroup = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management
         *  \\brief Operations on device-side memory.
         */
        int placeholder(void);
        """
    )
    raw = "@brief Symbol-specific brief.\n@ingroup memory\n"
    out = _render_brief(raw, defgroup)
    assert "Symbol-specific brief." in out
    assert "[group:" not in out


def test_render_brief_prefers_inferred_brief_over_group_brief():
    """Part 11 (first-sentence-of-body inference) wins over Part 12a."""
    defgroup = textwrap.dedent(
        """\
        /**
         *  \\defgroup memory Memory Management
         *  \\brief Operations on device-side memory.
         */
        int placeholder(void);
        """
    )
    raw = (
        "Inferred symbol description.\n\n"
        "More details follow.\n"
        "@ingroup memory\n"
    )
    out = _render_brief(raw, defgroup)
    assert "Inferred symbol description." in out
    assert "[group:" not in out


def test_render_brief_no_host_node_means_no_group_fallback():
    """Backwards-compat: callers that don't pass host_node still get the
    placeholder when there's no @brief or inferable body."""
    sections = list(DOXYGEN_CONV.parse_structure("").children)
    out = DoxygenMixin._render_doxygen_brief(sections, missing_text="P")
    assert "P" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
