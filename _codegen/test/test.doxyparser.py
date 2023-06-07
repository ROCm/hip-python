import textwrap

import addtoplevelpath
import doxyparser

grammar = doxyparser.DoxygenGrammar()

print(grammar.all.parseString(r"\a TEST"))

doxygen_input = r"""
@brief Gets an opaque interprocess handle for an event.
  
This opaque handle may be copied into other processes and opened with hipIpcOpenEventHandle.
Then hipEventRecord, hipEventSynchronize, hipStreamWaitEvent and hipEventQuery may be used in
either process. Operations on the imported event after the exported event has been freed with hipEventDestroy
will result in undefined behavior.


\param[in] param1 My description ending at the next param section. \a Italic text.
\param[in,out] param2 My multiline
                description ending
                at a blank line. \b BOLD text.

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

@f{eqnarray}

  \textit{\param[in,out] param6 this will not be changed.}

@f}
"""

for mtch in grammar.all.scanString(doxygen_input):
    print(mtch)

class MyParamFormatter(doxyparser.styles.PythonDocstrings):

    @staticmethod
    def paragraphs_no_args(tokens):
        cmd = tokens[0][1:]
        text_lines = tokens[1].lstrip().splitlines(keepends=False)
        if cmd in ("short","brief"):
            doxygen_brief =  " ".join(text_lines)
            return doxygen_brief + "\n"
        if cmd == "note":
            return "Note:\n"+textwrap.indent("\n".join([ln.lstrip() for ln in text_lines])," "*3)
        return None

    @staticmethod
    def param(tokens):
        # ['\\param', '[in]', 'param1', 'My description ending at the next param section.']
        dir_map = {
            "in": "IN",
            "in,out": "INOUT",
            "out": "OUT",
        }
        name = tokens[2]
        descr = tokens[3]
        dir = dir_map[tokens[1]]
        return f":param {name}: {descr} Direction: {dir}"
    
grammar.style = MyParamFormatter

print(grammar.transform_string(doxygen_input))

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
