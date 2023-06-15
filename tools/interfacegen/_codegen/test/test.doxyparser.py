import textwrap

import addtoplevelpath
import doxyparser

grammar = doxyparser.DoxygenGrammar()

print(grammar.all.parseString(r"\a TEST"))
print(grammar.code.parseString(r"\code{.c}\endcode",parseAll=True))

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

#doxygen_input = r"""
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
#"""

#doxygen_input = r"""
#   \brief Sparse matrix dense matrix multiplication using CSR storage format
#
#   \details
#   \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$.
#"""


for mtch in grammar.all.scanString(doxygen_input):
    print(mtch)

grammar.escaped.setParseAction(doxyparser.format.PythonDocstrings.escaped)
grammar.with_word.setParseAction(doxyparser.format.PythonDocstrings.with_word)
grammar.fdollar.setParseAction(doxyparser.format.PythonDocstrings.fdollar)

for node in grammar.parse_structure(doxygen_input).walk():
    indent = ' '*node.level
    print(f"{indent}{str(node)}")
    if isinstance(node,doxyparser.Section):
        print(f"{indent}{node.kind}")
    if isinstance(node,doxyparser.MathBlock):
        print(textwrap.indent(
          '"""'
          + node.text
          + '"""',
          indent
        ))
    elif isinstance(node,doxyparser.VerbatimBlock):
        print(textwrap.indent(
          '"""'
          + node.text
          + '"""',
          indent
        ))
    elif isinstance(node,doxyparser.TextBlock):
        print(textwrap.indent(
          '"""'
          + node.transformed_text
          + '"""',
          indent
        ))

#

import pyparsing as pyp


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

#for tokens,start,end in pyp.cppStyleComment.scanString(comments):
#    print(tokens)

print(doxyparser.remove_doxygen_cpp_comments(comments))


print(grammar.section.parseString(
r"""  \details 
  \p first line
"""
))
