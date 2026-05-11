"""Regression tests for cython._build_inherited_comment_index — the
libclang token-walk that inherits an opener `@{ … @}` comment for
function declarations whose own raw_comment is empty.

This is Part 12b of the docs-polish plan: hipblas alone has 989
`(No short description, might be part of a group.)` placeholders
because libclang only attaches the opener comment to the FIRST
declaration in each `@{ … @}` block.
"""

import logging
import textwrap

import pytest

from interfacegen import cython
from interfacegen.cparser import CParser


@pytest.fixture
def interfacegen_warnings():
    """Capture warnings emitted on the `interfacegen` logger.
    pytest's caplog doesn't capture from this logger because
    `interfacegen.__init__.disable_logging()` runs on import.
    Re-enable for the test, install our own handler, and restore."""
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    logger = logging.getLogger("interfacegen")
    prev_level = logger.level
    prev_disabled = logger.disabled
    logger.disabled = False
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.disabled = prev_disabled


def _parse_header(text: str):
    """Return (translation_unit, root). Mirrors test conftest setup;
    the in-memory header is parsed via the same CParser the codegen
    uses, so libclang sees the comments as it would on disk."""
    parser = CParser("input.h", unsaved_files=[("input.h", text)])
    parser.parse()
    return parser.translation_unit


class _FakeRoot:
    """Minimal Root stand-in — owns a cursor whose translation_unit
    drives the token walk. Mirrors `tree.Root` for the only attributes
    `_build_inherited_comment_index` actually touches."""

    def __init__(self, tu):
        self.cursor = tu.cursor


def _index_for(text: str):
    tu = _parse_header(text)
    root = _FakeRoot(tu)
    return cython._build_inherited_comment_index(root), tu


def _function_cursors_by_name(tu):
    """Return {name: cursor} for every FUNCTION_DECL in the TU (top-
    level only)."""
    import clang.cindex as cci
    out = {}
    for c in tu.cursor.get_children():
        if c.kind == cci.CursorKind.FUNCTION_DECL:
            out[c.spelling] = c
    return out


# -- Anonymous-group form (hipblas style) ----------------------------------


def test_inheritance_index_anonymous_group():
    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief BLAS Level 2 API
         */
        int a(int x);
        int b(int x);
        int c(int x);
        //! @}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # `a` already has the opener attached by libclang — not in the
    # inheritance index.
    assert cursors["a"].hash not in idx
    # `b` and `c` lack their own raw_comment — should inherit.
    assert cursors["b"].hash in idx
    assert cursors["c"].hash in idx
    assert "BLAS Level 2 API" in idx[cursors["b"].hash]
    assert "BLAS Level 2 API" in idx[cursors["c"].hash]


# -- Named addtogroup form (HSA style) -------------------------------------


def test_inheritance_index_named_addtogroup():
    src = textwrap.dedent(
        """\
        /** \\addtogroup memory Memory Management
         *  @{ */
        int hsa_memory_allocate(int n);
        int hsa_memory_free(int n);
        /** @} */
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # First decl has the comment; second inherits.
    assert cursors["hsa_memory_free"].hash in idx
    assert "Memory Management" in idx[cursors["hsa_memory_free"].hash]


# -- Nested groups ---------------------------------------------------------


def test_inheritance_index_nested_groups():
    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief OUTER
         */
        int outer_first(int x);
        int outer_second(int x);
        /*! @{
         *  \\brief INNER
         */
        int inner_first(int x);
        int inner_second(int x);
        //! @}
        int outer_third(int x);
        //! @}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # outer_second has no comment of its own and the outer block is
    # the only active opener at that point.
    assert "OUTER" in idx[cursors["outer_second"].hash]
    # inner_second is inside the inner block — inherits INNER.
    assert "INNER" in idx[cursors["inner_second"].hash]
    # outer_third sits between inner @} and outer @} — inherits OUTER again.
    assert "OUTER" in idx[cursors["outer_third"].hash]


# -- Decl with its own comment -------------------------------------------


def test_inheritance_index_skip_commented_decl():
    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief OUTER
         */
        int a(int x);
        /** \\brief MY OWN BRIEF */
        int b(int x);
        //! @}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    assert cursors["b"].hash not in idx, (
        "decl with its own raw_comment must not be added to the "
        "inheritance index"
    )


# -- Decls outside any block ----------------------------------------------


def test_inheritance_index_skip_outside_block():
    src = textwrap.dedent(
        """\
        int before(int x);
        /*! @{
         *  \\brief OUTER
         */
        int inside(int x);
        //! @}
        int after(int x);
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    assert cursors["before"].hash not in idx
    assert cursors["after"].hash not in idx


# -- Backslash open/close spelling ----------------------------------------


def test_inheritance_index_handles_backslash_open_close():
    src = textwrap.dedent(
        """\
        /*! \\{
         *  \\brief BACKSLASH BRIEF
         */
        int a(int x);
        int b(int x);
        //! \\}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    assert cursors["b"].hash in idx
    assert "BACKSLASH BRIEF" in idx[cursors["b"].hash]


# -- Function-only scope ---------------------------------------------------


def test_inheritance_index_only_includes_functions():
    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief BATCH
         */
        int batch_func(int x);
        typedef int batch_typedef;
        struct batch_struct { int x; };
        //! @}
        """
    )
    idx, tu = _index_for(src)
    import clang.cindex as cci
    # Build a {name: hash} for every top-level cursor so we can
    # assert non-function cursors aren't in the index.
    name_to_hash = {}
    for c in tu.cursor.get_children():
        try:
            name_to_hash[c.spelling] = c.hash
        except Exception:
            continue
    # Function with no own comment IS in the index. (`batch_func` is
    # the FIRST decl after the opener; libclang attaches the opener
    # to it directly. To force-test the function-only filter we'd
    # need a second function — but the typedef/struct cursors below
    # cover the negative side.)
    # Negative side: typedef and struct are not in the inheritance
    # index regardless of their position in the block.
    if "batch_typedef" in name_to_hash:
        assert name_to_hash["batch_typedef"] not in idx
    if "batch_struct" in name_to_hash:
        assert name_to_hash["batch_struct"] not in idx


# -- Stray @} ---------------------------------------------------------------


def test_inheritance_index_warns_on_stray_close(interfacegen_warnings):
    src = textwrap.dedent(
        """\
        //! @}
        /*! @{
         *  \\brief AFTER STRAY
         */
        int a(int x);
        int b(int x);
        //! @}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # The valid block after the stray close still indexes b.
    assert cursors["b"].hash in idx
    # And we logged a warning about the stray close.
    assert any("stray @}" in m for m in interfacegen_warnings)


# -- Unclosed @{ ----------------------------------------------------------


def test_inheritance_index_warns_on_unclosed_open(interfacegen_warnings):
    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief UNCLOSED
         */
        int a(int x);
        int b(int x);
        int c(int x);
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # Tail decls still inherit (graceful degradation).
    assert cursors["b"].hash in idx
    assert cursors["c"].hash in idx
    assert any("unclosed" in m for m in interfacegen_warnings)


def test_raw_comment_cleaned_strips_group_brackets_from_own_comment(tmp_path):
    """The FIRST decl after `/*! @{ ... */` has the opener attached to
    it directly via libclang. The `@{` / `@}` markers must be stripped
    from `_raw_comment_cleaned` output too — not just from inherited
    comments — otherwise they leak verbatim into the rendered docstring
    (the user-visible regression that motivated this test)."""
    from interfacegen.test._codegen_helpers import build_root

    src = textwrap.dedent(
        """\
        /*! @{
         *  \\brief BLAS Level 2 API
         */
        int hipblasStpsvBatched(int x);
        int hipblasDtpsvBatched(int x);
        //! @}
        """
    )
    root = build_root(src)
    # Attach the default raw_comment_cleaner so _raw_comment_cleaned can
    # complete (treefactory leaves it unset on a bare parse).
    from interfacegen.cython import DEFAULT_RAW_COMMENT_CLEANER
    funcs = [n for n in root.walk(postorder=False) if hasattr(n, "_raw_comment_cleaned") and n.raw_comment]
    assert funcs, "expected at least one node with a raw_comment"
    funcs[0].raw_comment_cleaner = DEFAULT_RAW_COMMENT_CLEANER
    cleaned = funcs[0]._raw_comment_cleaned()
    assert "@{" not in cleaned, (
        f"@{{ should be stripped from own raw_comment; got: {cleaned!r}"
    )
    assert "@}" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
