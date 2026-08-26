# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for transitive `\\copydoc` resolution.

`libclang` does NOT resolve doxygen's `\\copydoc <ref>` directive —
it returns the literal forwarding string in `cursor.raw_comment`.
The codegen plugs a resolver into the existing
`DoxygenMixin._raw_comment_cleaned` chokepoint so every downstream
consumer (doxyparser, brief inference, group fallback, python-comment
rendering) operates on the merged content transparently.

Pinned behaviors:
  - delimiter-stripping happens BEFORE substitution (no nested
    `/* */` artifacts in the merged text)
  - chains of A→B→C resolve fully
  - cycles are detected and broken (innermost directive left literal)
  - unknown references degrade gracefully (directive left literal)
  - both `\\copydoc` and `@copydoc` spellings are recognized
  - PR #7346-shape comments (`\\ingroup foo \\copydoc Bar`) preserve
    the ingroup tag while resolving the copydoc reference
  - resolved content reaches the rendered docstring (end-to-end)
"""

import textwrap

import pytest
from interfacegen.test._codegen_helpers import find_function, make_generator


def _root_for(src: str):
    """Build a CythonModuleGenerator-backed tree so every Function
    node gets ``raw_comment_cleaner`` set via
    ``CythonModuleGenerator.initialize_nodes`` — required for
    ``_raw_comment_cleaned()`` to be callable. ``build_root`` alone
    skips that initialization step."""
    return make_generator(src).backend.root


# -- Case 0: no-nesting invariant (the regression test for the
# "must clean before substituting" decision) ------------------------


def test_copydoc_substitution_does_not_carry_comment_delimiters():
    # Multi-line comments — the realistic ROCm-header shape.
    # `remove_doxygen_comment_chars` has a pre-existing edge case
    # on single-line `/*! ... */` blocks (the trailing `*/` isn't
    # stripped because the i==0 and i==last branches are exclusive),
    # which would falsely fail this regression. The pre-cache
    # cleaning step in the resolver still operates correctly even
    # on those edge cases — but for THIS test we use the multi-line
    # form to assert the no-nesting property cleanly.
    src = textwrap.dedent(
        """\
        /*! \\brief Foo body
         */
        int A(int x);

        /*! \\copydoc A
         */
        int B(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "B")._raw_comment_cleaned()
    # Resolved A content is present.
    assert "Foo body" in cleaned
    # No nested-comment delimiters bled through. `*/` is the
    # smoking gun — it would close any enclosing comment block in
    # downstream consumers.
    assert "*/" not in cleaned
    assert "/*!" not in cleaned
    assert "/**" not in cleaned
    # The literal copydoc directive is gone.
    assert "\\copydoc" not in cleaned
    assert "@copydoc" not in cleaned


# -- Case 1: simple forward reference -------------------------------


def test_copydoc_simple_forward_reference():
    src = textwrap.dedent(
        """\
        /*! \\brief Foo */
        int A(int x);

        /*! \\copydoc A */
        int B(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "B")._raw_comment_cleaned()
    assert "Foo" in cleaned
    assert "\\copydoc A" not in cleaned


# -- Case 2: transitive A→B→C ---------------------------------------


def test_copydoc_transitive_chain():
    src = textwrap.dedent(
        """\
        /*! \\brief Real description */
        int A(int x);

        /*! \\copydoc A */
        int B(int x);

        /*! \\copydoc B */
        int C(int x);
        """
    )
    root = _root_for(src)
    cleaned_c = find_function(root, "C")._raw_comment_cleaned()
    assert "Real description" in cleaned_c
    assert "\\copydoc" not in cleaned_c
    assert "@copydoc" not in cleaned_c


# -- Case 3: cycle --------------------------------------------------


def test_copydoc_cycle_does_not_loop():
    src = textwrap.dedent(
        """\
        /*! \\copydoc B */
        int A(int x);

        /*! \\copydoc A */
        int B(int x);
        """
    )
    root = _root_for(src)
    # No infinite loop, no recursion error — the test just completing
    # is half the assertion.
    cleaned_a = find_function(root, "A")._raw_comment_cleaned()
    cleaned_b = find_function(root, "B")._raw_comment_cleaned()
    # The innermost directive in the cycle is left as a literal
    # (matches doxygen's "no matching definition" graceful-degradation
    # behavior on cycles).
    assert "\\copydoc" in cleaned_a or "\\copydoc" in cleaned_b


# -- Case 4: unknown reference --------------------------------------


def test_copydoc_unknown_reference_left_literal():
    src = textwrap.dedent(
        """\
        /*! \\copydoc nonexistent_function */
        int A(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "A")._raw_comment_cleaned()
    # No crash, directive preserved verbatim.
    assert "\\copydoc nonexistent_function" in cleaned


# -- Case 5: PR #7346 shape (`\\ingroup` + `\\copydoc`) -------------


def test_copydoc_pr7346_shape_preserves_surrounding_tags():
    src = textwrap.dedent(
        """\
        /*! \\brief Scale a sparse vector and add it to a dense vector.
         *
         *  \\param[in] x sparse vector input.
         */
        int hipsparseSaxpyi(int x);

        /*! \\ingroup level1_module
         *  \\copydoc hipsparseSaxpyi
         */
        int hipsparseDaxpyi(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "hipsparseDaxpyi")._raw_comment_cleaned()
    # `\ingroup level1_module` survives — copydoc inserts content,
    # doesn't suppress surrounding tags.
    assert "\\ingroup level1_module" in cleaned
    # The resolved S-variant content is present.
    assert "Scale a sparse vector" in cleaned
    assert "sparse vector input" in cleaned
    # The literal directive is gone.
    assert "\\copydoc hipsparseSaxpyi" not in cleaned


# -- Case 6: mixed real + forward -----------------------------------


def test_copydoc_mixed_real_and_forwarded_content():
    src = textwrap.dedent(
        """\
        /*! \\brief Foo content */
        int A(int x);

        /*! \\brief Other content
         *  \\copydoc A
         */
        int B(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "B")._raw_comment_cleaned()
    # Both B's own brief AND the substituted A content present.
    assert "Other content" in cleaned
    assert "Foo content" in cleaned
    assert "\\copydoc A" not in cleaned


# -- Case 7: `@copydoc` at-sign spelling ----------------------------


def test_copydoc_at_sign_spelling_resolves():
    src = textwrap.dedent(
        """\
        /*! \\brief Foo */
        int A(int x);

        /*! @copydoc A */
        int B(int x);
        """
    )
    root = _root_for(src)
    cleaned = find_function(root, "B")._raw_comment_cleaned()
    assert "Foo" in cleaned
    assert "@copydoc A" not in cleaned


# -- Case 8: end-to-end via render_python_docstring -----------------


def test_copydoc_resolution_reaches_rendered_docstring():
    # Multi-line shape — see comment in case 0 about why we use this
    # over a single-line `/*! ... */` block. The brief inference
    # walker reads the post-resolved cleaned text via
    # `_raw_comment_cleaned`, so the resolved content must surface
    # in the rendered docstring without the literal directive.
    src = textwrap.dedent(
        """\
        /*! \\brief Real brief here
         */
        int A(int x);

        /*! \\copydoc A
         */
        int B(int x);
        """
    )
    root = _root_for(src)
    fn_b = find_function(root, "B")
    # render_python_docstring wraps the assembled docstring as
    # r"""..."""; the resolved brief must appear inside.
    # Function.render_python_docstring returns (docstring, indent).
    rendered, _indent = fn_b.render_python_docstring(cprefix="")
    assert "Real brief here" in rendered
    # The literal directive must NOT be in the rendered docstring.
    assert "\\copydoc" not in rendered
    assert "@copydoc" not in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
