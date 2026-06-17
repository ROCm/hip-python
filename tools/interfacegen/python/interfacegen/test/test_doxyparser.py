# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import textwrap

from interfacegen import doxyparser

grammar = doxyparser.DoxygenGrammar()


def test_specific():
    print(grammar.all.parse_string(r"\a TEST"))
    print(grammar.code.parse_string(r"\code{.c}\endcode", parse_all=True))


def test_grammar():

    doxygen_input = r"""
    @brief Gets an opaque interprocess handle for an event.

    This \p opaque handle may be copied into \f$3\times 4\f$ other processes and opened with hipIpcOpenEventHandle.
    Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
    either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
    will result in undefined behavior.

    \details
    \p hipsparseCreateIdentityPermutation stores the identity map in \p p, such that
    \f$p = 0:1:(n-1)\f$.

    \param[in]
    param1 My description ending at the next param section. \a Italic text.
    \param[in,out] param2 My multiline\>\n
                    description ending
                    at a blank line. \b BOLD text.
                    \f[
                    a = 2*b
                    \f]


    \note My note ending at a blank line.

    \note My multiline note
          ending at a blank line.

    \note My multiline note ending at the begin
          of a parameter.
    \param[out] param3 My multiline
                    description ending
                    at the end of the text. \c Monotype text.

    \verbatim

    \param[in,out] param4 this will not be changed.

    \endverbatim

    \f[

      \textit{\param[in,out] param5 this will not be changed.}

    \f]

    @f{eqnarray}{

      \textit{\param[in,out] param6 this will not be changed.}

    @f}

    \code{.c}
        for(i = 0; i < n; ++i)
        {
            p[i] = i;
        }
    \endcode


     @note hipMallocPitch, hipFree, hipMallocArray, hipFreeArray, hipMalloc3D, hipMalloc3DArray,
    hipHostFree, hipHostMalloc
    """

    # doxygen_input = r"""
    #   \brief Sparse matrix dense matrix multiplication using CSR storage format
    #
    #   \details
    #   \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
    #   matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
    #   matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
    #   is multiplied by the scalar \f$\beta\f$, such that
    #   \f[
    #     C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
    #   \f]
    # """

    # doxygen_input = r"""
    #   \brief Sparse matrix dense matrix multiplication using CSR storage format
    #
    #   \details
    #   \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$.
    # """

    for mtch in grammar.all.scan_string(doxygen_input):
        print(mtch)

    grammar.escaped.set_parse_action(doxyparser.format.PythonDocstrings.escaped)
    grammar.with_word.set_parse_action(
        doxyparser.format.PythonDocstrings.with_word
    )
    grammar.fdollar.set_parse_action(doxyparser.format.PythonDocstrings.fdollar)

    for node in grammar.parse_structure(doxygen_input).walk():
        indent = " " * node.level
        print(f"{indent}{str(node)}")
        if isinstance(node, doxyparser.Section):
            print(f"{indent}{node.kind}")
        if isinstance(node, doxyparser.MathBlock):
            print(textwrap.indent('"""' + node.text + '"""', indent))
        elif isinstance(node, doxyparser.VerbatimBlock):
            print(textwrap.indent('"""' + node.text + '"""', indent))
        elif isinstance(node, doxyparser.TextBlock):
            print(
                textwrap.indent('"""' + node.transformed_text + '"""', indent)
            )


#


def test_comments():
    comments = r"""\

    /// My /// docu line 1
    /// My /// docu line 2
    /// My /// docu line 3

    //! My //! docu line 1
    //! My //! docu line 2
    //! My //! docu line 3

    /** My /** * docu line 1
     *  My /** * docu line 2
     */

    /*! My /*! * docu line 1
     *  My /*! * docu line 2
     */

      /*! My /*! docu line 1
          My /*! docu line 2
       */

      /** My /** docu line 1
          My /** docu line 2
       */


      /** My /** docu ending with *\/ line 1
          My /** docu ending with *\/ line 2*/

    /* Normal C comment */

    // Normal C comment

    /*!
    */
    """

    # for tokens,start,end in pyp.cppStyleComment.scan_string(comments):
    #    print(tokens)

    print(doxyparser.remove_doxygen_comment_chars(comments))


def test_sections():
    print(
        grammar.section.parse_string(
            r"""  \details
      \p first line
    """
        )
    )


def test_fdollar_inline_math_single_line():
    """Inline ``\\f$..\\f$`` math must collapse to a single line with no space
    before the closing backtick (docutils inline-markup requirement)."""
    local_grammar = doxyparser.DoxygenGrammar()
    local_grammar.fdollar.set_parse_action(
        doxyparser.format.PythonDocstrings.fdollar
    )

    # trailing space before the closing delimiter must be dropped
    (result,) = local_grammar.fdollar.parse_string(
        r"\f$ D = alpha * opReduce(opA(A)) + beta * opC(C) \f$",
        parse_all=True,
    )
    assert result == ":math:`D = alpha * opReduce(opA(A)) + beta * opC(C)`"
    assert " `" not in result  # no space immediately before closing backtick

    # multi-line content must be collapsed onto a single line
    (result,) = local_grammar.fdollar.parse_string(
        "\\f$\\mathcal{E} \\gets \\alpha \\mathcal{A}\n"
        "    \\mathcal{B}\n"
        "    + \\beta \\mathcal{D}\\f$",
        parse_all=True,
    )
    assert "\n" not in result
    assert (
        result
        == r":math:`\mathcal{E} \gets \alpha \mathcal{A} \mathcal{B} + \beta \mathcal{D}`"
    )


def test_plain_block_comment_stripped():
    """Plain (non-doxygen) ``/* .. */`` comments must have their delimiters and
    leading ``*`` markers stripped so stray asterisks do not leak into
    docstrings."""
    comment = textwrap.dedent(
        """\
        /* matrix A values are updated inplace
         * to be the preconditioner M values */
        """
    )
    result = doxyparser.remove_doxygen_comment_chars(comment)
    assert "/*" not in result
    assert "*/" not in result
    assert "matrix A values are updated inplace" in result
    assert "to be the preconditioner M values" in result
    # the leading continuation '*' must be gone
    assert not any(ln.lstrip().startswith("*") for ln in result.splitlines())


def test_single_line_doc_comment_strips_closer():
    """A single-line ``/*! .. */`` (or ``/** .. */``) must have its trailing
    ``*/`` stripped, otherwise the residue leaks into the docstring."""
    assert "*/" not in doxyparser.remove_doxygen_comment_chars("/*! @endcond */")
    assert "*/" not in doxyparser.remove_doxygen_comment_chars("/** brief */")
    assert (
        doxyparser.remove_doxygen_comment_chars("/** brief */").strip() == "brief"
    )


def test_group_and_cond_only_comment_is_not_documentation():
    """Comments made up solely of group/condition markers and delimiters carry
    no documentation and must be classified as bare so they are dropped."""
    from interfacegen.cython import _doxygen

    assert _doxygen._raw_comment_is_only_group_bracket("/*! @endcond */\n/*! @} */")
    assert _doxygen._raw_comment_is_only_group_bracket("///@{")
    # a real doc comment must NOT be treated as bare
    assert not _doxygen._raw_comment_is_only_group_bracket(
        "/** Does a real thing. */"
    )


def test_see_reference_drops_trailing_parens():
    """``@see foo()`` must produce a single clean role with no trailing ``()``
    (a ``(`` right after the closing backtick is invalid RST inline-markup)."""
    from interfacegen.cython import _doxygen

    out = _doxygen.DOXYGEN_CONV.see_reference.transform_string(
        "hipdnnBackendCreateDescriptor()"
    )
    assert out == ":py:obj:`.hipdnnBackendCreateDescriptor`"
    assert "`(" not in out


def test_cxx_namespace_reference_left_untouched():
    """``::``-qualified (C++ namespace) references are not exposed by the C-only
    bindings; they must be left as plain text, not converted to a role."""
    from interfacegen.cython import _doxygen

    out = _doxygen.DOXYGEN_CONV.see_reference.transform_string(
        "llvm::llvm_shutdown"
    )
    assert out == "llvm::llvm_shutdown"
    assert ":py:obj:" not in out


def test_see_reference_no_double_wrap():
    """A ``#``-qualified reference must yield exactly one role. Rendering a
    see/sa body with the reference pass disabled (``transform_references=False``)
    and then a single ``see_reference`` pass must not nest roles."""
    from interfacegen.cython import _doxygen

    g = _doxygen.DOXYGEN_CONV
    body = g.transform_text_block(
        "#llvm_shutdown",
        transform_formatting=True,
        transform_other=True,
        transform_references=False,
    )
    out = g.see_reference.transform_string(body)
    assert out == ":py:obj:`.llvm_shutdown`"
    assert out.count(":py:obj:") == 1
