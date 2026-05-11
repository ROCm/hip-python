"""Regression tests for cython.DoxygenMixin._infer_brief_from_first_text
and the corresponding fallback in _render_doxygen_brief.

Many ROCm headers omit `@brief` and just write a free-form description.
Without inference the rendered docstring's first line was the literal
`(No short description)` placeholder, wasting a perfectly usable first
sentence already present in the doxygen comment.
"""

import textwrap

import pytest

from interfacegen import doxyparser
from interfacegen.cython import DOXYGEN_CONV, DoxygenMixin


infer = DoxygenMixin._infer_brief_from_first_text


def _render_brief(raw):
    """Helper: parse `raw` doxygen text and run the brief renderer."""
    tree = DOXYGEN_CONV.parse_structure(raw)
    sections = list(tree.children)
    return DoxygenMixin._render_doxygen_brief(sections)


# --- _infer_brief_from_first_text -----------------------------------------


def test_single_line_paragraph_returned_verbatim():
    src = "Short one-liner description.\n\nFurther details follow here."
    assert infer(src) == "Short one-liner description."


def test_multiline_paragraph_takes_first_sentence():
    src = (
        "First sentence ends here. Second sentence in the same paragraph "
        "and across\nlines."
    )
    assert infer(src) == "First sentence ends here."


def test_long_sentence_truncated_with_ellipsis():
    long_word = "quitelong " * 30
    src = f"This sentence is intentionally extremely long: {long_word} end."
    out = infer(src)
    assert out.endswith(" [...]")
    assert len(out) <= DoxygenMixin._INFERRED_BRIEF_MAX_LEN + len(" [...]")


def test_empty_body_returns_empty():
    assert infer("") == ""
    assert infer("\n\n\t  \n") == ""


def test_leading_colon_and_whitespace_stripped():
    src = ":   Short description after colon."
    assert infer(src) == "Short description after colon."


def test_period_in_identifier_does_not_split_prematurely():
    # Period followed by a non-space character should not terminate
    # the sentence — covers things like "rocm.bindings.foo".
    src = (
        "Use rocm.bindings.foo to access the API.\nAdditional details."
    )
    out = infer(src)
    assert out == "Use rocm.bindings.foo to access the API."


# --- _render_doxygen_brief integration ------------------------------------


def test_render_brief_uses_explicit_brief_when_present():
    raw = textwrap.dedent(
        """\
        @brief Explicit short description.

        @details Long details body that should be ignored for the brief.
        """
    )
    out = _render_brief(raw)
    assert out.strip() == "Explicit short description."


def test_render_brief_infers_from_details_when_no_brief_present():
    raw = (
        "This is a free-form description with no doxygen brief marker.\n\n"
        "Additional paragraphs follow here.\n"
    )
    out = _render_brief(raw)
    assert "(No short description)" not in out
    assert "free-form description" in out


def test_render_brief_falls_back_to_placeholder_when_no_text():
    raw = ""
    out = _render_brief(raw)
    assert "(No short description)" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
