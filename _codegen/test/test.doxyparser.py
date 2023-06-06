import addtoplevelpath
from doxyparser import DoxygenGrammar, styles

grammar = DoxygenGrammar()

param_list = r"""

\param[in] param1 My description ending at the next param section. \a Italic text.
\param[in,out] param2 My multiline
                description ending
                at a blank line. \b BOLD text.

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

\f$\textit{\param[in,out] param7 this will not be changed.}\f$
"""

for mtch in grammar.all.scanString(param_list):
    print(mtch)

class MyParamFormatter(styles.PythonDocstrings):

    @staticmethod
    def param(tokens):
        # ['\\param', '[in]', 'param1', 'My description ending at the next param section.']
        dir_map = {
            "[in]": "IN",
            "[in,out]": "INOUT",
            "[out]": "OUT",
        }
        name = tokens[2]
        descr = tokens[3]
        dir = dir_map[tokens[1]]
        return f":param {name}: {descr} Direction: {dir}"
    
grammar.style = MyParamFormatter

print(grammar.transform_string(param_list))
