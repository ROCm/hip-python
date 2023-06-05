from doxyparser import DoxygenGrammar

grammar = DoxygenGrammar()

print(grammar.all.parseString(r"\a TEST"))

param_list = r"""

\param[in] param1 My description ending at the next param section. \a Italic text.
\param[in,out] param2 My multiline
                description ending
                at a blank line. \b BOLD text.

\param[out] param3 My multiline 
                description ending
                at the end of the text. \c Monotype text.
"""

for mtch in grammar.all.scanString(param_list):
    print(mtch)

class MyParamFormatter(grammar.PythonDocstrings):

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
    
grammar.output_style = MyParamFormatter

print(grammar.transform(param_list))

