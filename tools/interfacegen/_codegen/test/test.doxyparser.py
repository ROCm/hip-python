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

\param[in] param1 My description ending at the next param section. \a Italic text.
\param[in,out] param2 My multiline
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
# changing doxygen command to lower case in verbatim-like environments will break the parser
\PARAM[in,out] param4 this will not be changed.

\endverbatim

\f[

  \textit{\PARAM[in,out] param5 this will not be changed.}

\f]

@f{eqnarray}{

  \textit{\PARAM[in,out] param6 this will not be changed.}

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

for mtch in grammar.all.scanString(doxygen_input):
    print(mtch)

#class MyParamFormatter(doxyparser.styles.PythonDocstrings):
#
#    @staticmethod
#    def paragraphs_no_args(tokens):
#        cmd = tokens[0][1:]
#        text_lines = tokens[1].lstrip().splitlines(keepends=False)
#        if cmd in ("short","brief"):
#            doxygen_brief =  " ".join(text_lines)
#            return doxygen_brief + "\n"
#        if cmd == "note":
#            return "Note:\n"+textwrap.indent("\n".join([ln.lstrip() for ln in text_lines])," "*3)
#        return None
#
#    @staticmethod
#    def param(tokens):
#        # ['\\param', '[in]', 'param1', 'My description ending at the next param section.']
#        print(tokens)
#        dir_map = {
#            "in": "IN",
#            "in,out": "INOUT",
#            "out": "OUT",
#            None: "Unspecified",
#        }
#        name = tokens[2]
#        descr = tokens[3]
#        dir = dir_map[tokens[1]]
#        return f":param {name}: {descr.rstrip()}. Direction: {dir}\n"
#
#    @staticmethod
#    def verbatim_begin(tokens):
#        cmd = tokens[0][1:]
#        if cmd == "code" and len(tokens) > 2:
#            lang = tokens[2][1:]
#            return f".. code-block:: {lang}\n"
#        return ".. code-block::\n"
#        
#    
#    @staticmethod
#    def verbatim_end(tokens):
#        return []
#    
#    @staticmethod
#    def math_end(tokens):
#        cmd = tokens[0][1:]
#        if cmd == "f$":
#            return "`"
#        return []
#    
#    @staticmethod
#    def math_begin(tokens):
#        cmd = tokens[0][1:]
#        if cmd == "f$":
#            return ":math:`"
#        return ".. math::"

for node in grammar.parse(doxygen_input).walk():
    print(f"{' '*node.level}{str(node)}")
    if isinstance(node,doxyparser.MathBlock):
        print(node.body)
    if isinstance(node,doxyparser.VerbatimBlock):
        print(node.body)
    #if isinstance(node,doxyparser.Section):
    #    print(f"{' '*node.level}{node.kind}")
    #elif isinstance(node,doxyparser.TextBlock):
    #    print(f"{' '*node.level}'{node.tokens[-1]}'")
        
    
#grammar.style = MyParamFormatter
#
#text = grammar.transform_part_1(doxygen_input)
##print(text)
#text = grammar.transform_part_2(text,verbatim_indent=" "*3)
#print(text)

#

import pyparsing as pyp


comments = """\

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

/* Normal C comment */

// Normal C comment
"""

#for tokens,start,end in pyp.cppStyleComment.scanString(comments):
#    print(tokens)

print(doxyparser.remove_doxygen_cpp_comments(comments))
