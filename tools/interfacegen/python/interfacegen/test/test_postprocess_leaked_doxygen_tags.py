"""Regression tests for the cython.DoxygenMixin._postprocess_leaked_doxygen_tags
defensive cleanup pass.

The doxyparser grammar has documented gaps:
- `@retval #FOO_BAR ...` — IDENT can't match a `#`-prefixed reference, so
  the entire line stays as text.
- `@p word` wrapped between two lines — `with_word` requires the argument
  on the same line.
- `@brief` markers that survived a section that wasn't promoted to a
  brief slot.

The cleanup runs as a final pass over the assembled docstring (after
references like `#NAME` have already been turned into `` `~.NAME` ``)
and rewrites these into Sphinx-friendly equivalents.
"""

import textwrap

import pytest

from interfacegen.cython import DoxygenMixin


postprocess = DoxygenMixin._postprocess_leaked_doxygen_tags


def _normalize(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()


def test_retval_with_backtick_reference_becomes_rst_list_item():
    src = textwrap.dedent(
        """\
        Query foo.

            @retval `~.AMD_COMGR_STATUS_SUCCESS` The function has
                been executed successfully.
        """
    )
    out = postprocess(src)
    assert "@retval" not in out
    assert "* :py:obj:`~.AMD_COMGR_STATUS_SUCCESS`:" in out


def test_retval_with_plain_ident_becomes_rst_list_item():
    src = "    @retval AMD_COMGR_STATUS_SUCCESS desc text"
    out = postprocess(src)
    assert "@retval" not in out
    assert "* :py:obj:`AMD_COMGR_STATUS_SUCCESS`: desc text" in out


def test_brief_marker_is_stripped():
    src = "@brief This is the short description."
    out = postprocess(src)
    assert "@brief" not in out
    assert "This is the short description." in out


def test_note_block_becomes_rst_directive():
    src = textwrap.dedent(
        """\
            @note This API is experimental.
                It may change without warning.
        """
    )
    out = postprocess(src)
    assert "@note" not in out
    assert ".. note::" in out
    assert "This API is experimental." in out


def test_warning_block_becomes_rst_directive():
    src = "    @warning Do not call from a signal handler."
    out = postprocess(src)
    assert "@warning" not in out
    assert ".. warning::" in out


def test_deprecated_block_becomes_rst_directive():
    src = "    @deprecated Use foo() instead."
    out = postprocess(src)
    assert "@deprecated" not in out
    assert ".. deprecated::" in out


def test_p_tag_with_line_wrap_becomes_inline_literal():
    src = "or @p\n    status_string is NULL."
    out = postprocess(src)
    assert "@p" not in out
    assert "`status_string`" in out


def test_c_tag_becomes_inline_literal():
    src = "Set the flag using @c FOO_FLAG to enable."
    out = postprocess(src)
    assert "@c " not in out
    assert "`FOO_FLAG`" in out


def test_a_tag_becomes_italic():
    src = "Affects @a parameter behaviour."
    out = postprocess(src)
    assert "@a " not in out
    assert "*parameter*" in out


def test_b_tag_becomes_bold():
    src = "Behaviour: @b mandatory for thread safety."
    out = postprocess(src)
    assert "@b " not in out
    assert "**mandatory**" in out


def test_see_with_reference_becomes_pyobj_role():
    src = "Also see @see `~.amd_comgr_release` for details."
    out = postprocess(src)
    assert "@see" not in out
    assert ":py:obj:`~.amd_comgr_release`" in out


def test_text_without_doxygen_tags_is_unchanged():
    src = "Plain prose with no doxygen tags."
    assert postprocess(src) == src


def test_html_bold_tag_becomes_rst_bold():
    src = "Section: <b> BLAS Level 2 API </b>"
    out = postprocess(src)
    assert "<b>" not in out and "</b>" not in out
    assert "**BLAS Level 2 API**" in out


def test_html_italic_tag_becomes_rst_italic():
    src = "<i>foo</i> and <em>bar</em>"
    out = postprocess(src)
    assert "<i>" not in out and "<em>" not in out
    assert "*foo*" in out and "*bar*" in out


def test_html_code_tag_becomes_rst_inline_literal():
    src = "Use <code>HIPBLAS_OP_N</code> here."
    out = postprocess(src)
    assert "<code>" not in out
    assert "``HIPBLAS_OP_N``" in out


def test_html_br_becomes_blank_line():
    src = "first<br/>second<br>third"
    out = postprocess(src)
    assert "<br" not in out
    assert "first\n\nsecond\n\nthird" == out


def test_already_clean_rst_passes_through():
    src = textwrap.dedent(
        """\
        Already-rendered docstring.

        Returns:
            * :py:obj:`~.X`: success
        """
    )
    assert postprocess(src) == src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
