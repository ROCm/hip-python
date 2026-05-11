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


def test_inheritance_index_pattern_a_doc_before_bare_opener_hipsparse_shape():
    """Pattern A as used throughout hipSPARSE (and many other ROCm
    libraries): a doc + `\\ingroup` comment placed BEFORE a bare
    `/**@{*/` opener.

    libclang merges the two adjacent comments into a single
    raw_comment for the first decl. Without the pattern-A walker
    fix, the rest of the lexical-block decls would inherit only the
    bare `/**@{*/` (empty after the `_strip_group_brackets` pass);
    with the fix the walker treats the prior comment as the active
    opener content, so D/C/Z get the family description too.

    Reference: /opt/rocm/include/hipsparse/internal/level1/hipsparse_axpyi.h
    """
    src = textwrap.dedent(
        """\
        /*! \\ingroup level1_module
         *  \\brief Scale a sparse vector and add it to a dense vector.
         *  \\details Long body, params, retvals, etc.
         */
        /**@{*/
        int hipsparseSaxpyi(int handle, int nnz);
        int hipsparseDaxpyi(int handle, int nnz);
        int hipsparseCaxpyi(int handle, int nnz);
        int hipsparseZaxpyi(int handle, int nnz);
        /**@}*/
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # Saxpyi has its own merged raw_comment via libclang — not in
    # the inheritance index by design.
    assert cursors["hipsparseSaxpyi"].hash not in idx
    # D/C/Z inherit the prior doc comment via the pattern-A fold.
    for name in ("hipsparseDaxpyi", "hipsparseCaxpyi", "hipsparseZaxpyi"):
        assert cursors[name].hash in idx, (
            f"{name} not in inheritance index (pattern-A fold failed)"
        )
        inherited = idx[cursors[name].hash]
        assert "Scale a sparse vector" in inherited, (
            f"{name} got: {inherited!r}"
        )
        assert "level1_module" in inherited


def test_inheritance_index_pattern_b_doc_inside_at_brace_block():
    """Pattern B from the official doxygen Member Groups example:
    a doc comment placed INSIDE the `@{ … @}` block (after the opener,
    before the first decl) is shared with every member of the lexical
    group via doxygen's DISTRIBUTE_GROUP_DOC behavior.

    libclang attaches the merged opener+doc to the first decl only;
    our walker should recognise the inside-block doc comment and use
    it as the inherited content for subsequent un-attached decls in
    the block.

    The reference snippet (from
    https://www.doxygen.nl/manual/grouping.html#memgroup):

        ///@{
        /** Same documentation for both members. Details */
        void func1InGroup1();
        void func2InGroup1();
        ///@}
    """
    src = textwrap.dedent(
        """\
        ///@{
        /** Same documentation for both members. Details */
        void func1InGroup1(void);
        void func2InGroup1(void);
        ///@}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # func1InGroup1 has its own (merged) raw_comment via libclang —
    # not in the inheritance index by design.
    assert cursors["func1InGroup1"].hash not in idx
    # func2InGroup1 has no immediately-preceding comment per
    # libclang. Without the pattern-B walker fix it would inherit
    # the empty `///@{` opener; with the fix it inherits the
    # inside-block doc comment.
    assert cursors["func2InGroup1"].hash in idx
    inherited = idx[cursors["func2InGroup1"].hash]
    assert "Same documentation for both members" in inherited, (
        f"Pattern-B inheritance failed: func2InGroup1 got {inherited!r}"
    )


def test_inheritance_index_pattern_b_most_recent_doc_wins():
    """When multiple inside-block doc comments appear before any
    decl, the most recent one wins — matching doxygen's
    `comment immediately preceding declaration` rule.
    """
    src = textwrap.dedent(
        """\
        ///@{
        /** First doc — should be replaced. */
        /** Second doc — should win. */
        void f1(void);
        void f2(void);
        ///@}
        """
    )
    idx, tu = _index_for(src)
    cursors = _function_cursors_by_name(tu)
    # f2 inherits the most recent inside-block doc comment.
    assert cursors["f2"].hash in idx
    inherited = idx[cursors["f2"].hash]
    assert "Second doc" in inherited
    assert "First doc" not in inherited


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
