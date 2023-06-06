# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import textwrap

import pyparsing as pyp

import warnings

def remove_doxygen_cpp_comments(text: str, dedent=True):
    """Strip away doxygen C++ comment delimiters.

    Args:
        dedent (bool): If the result should be dedented, i.e. the outermost level of indentation removed. Defaults to True.
    """
    result = ""
    last_end = 0

    for _,start,end in pyp.cppStyleComment.scanString(text):
        result += text[last_end:start]
        comment = text[start:end]
        if comment.lstrip().startswith("//"):
            comment = comment.replace("///","",1)
            comment = comment.replace("//!","",1)
            result += comment
        elif comment.lstrip()[0:3] in ("/**","/*!"):
            lines = comment.splitlines(keepends=True)
            for i,ln in enumerate(lines):
                has_linebreak = ln.endswith("\n")
                result_line = ln.rstrip()
                if i == 0:
                    idx = result_line.find("/*")
                    result_line = result_line.replace(result_line[idx:idx+3]," "*3,1)
                elif i == len(lines)-1:
                    idx = result_line.rfind("*/")
                    if idx > 0:
                        result_line = result_line[:idx]
                if result_line.lstrip().startswith("*"):
                    result_line = result_line.replace("*"," ",1)
                result += result_line
                if has_linebreak:
                        result += "\n"
        else: # other commnet
            result += comment
        last_end = end
    result += text[last_end:]
    if dedent:
        return textwrap.dedent(result)
    else:
        return result

class styles:

    """Collection of basic styles that users can base their custom
    style on.

    To set at parse action for the DoxygenGrammar parsers,
    a user must specify a ``@staticmethod`` with the same name 
    in a class definition and pass this one
    to an ``DoxygenGrammar`` instance via ``<mygrammar>.output_style = <mystyle>``.
    """

    class SuppressAll(object):
        """Suppresses all found expressions.
        Note:
            The implementation is complete as it does
            not specify a parse action for any DoxygenGrammar parser.
        """

        pass

    class KeepAll(object):
        """Keeps all found expressions as they are.
        """

        @staticmethod
        def identity(s: str,loc: int,tokens: list):
            
            tks = list(tokens)
            offset = loc
            while len(tks):
                tk = tks.pop(0)
                offset = s.find(tk,offset) + len(tk)
            return s[loc:offset]
        
        def __getattribute__(self, name: str):
            return self.identity


    class PythonDocstrings:
        """Base class for Python docstring output that
        defines parse actions for some DoxygenGrammar parsers such as
        ``escaped`` and ``with_word``
        """

        @staticmethod
        def escaped(tokens):
            expr = tokens[0]
            if expr == r"\n":
                return "\n"
            else:
                return expr

        @staticmethod
        def with_word(tokens):
            cmd = tokens[0][1:]
            arg = tokens[1]
            if cmd in ("a", "e", "em"):  # italic
                return f"*{arg}*"
            elif cmd == "anchor":  # reference anchor
                return arg  # TODO
            elif cmd == "b":  # bold
                return f"**{arg}**"
            else:  # if cmd in ("p","c"): # monotype
                return f"``{arg}``"

class DoxygenGrammar:

    kinds = {
        "escaped": [
            "amp",
            "at",
            "backslash",
            "chardot",
            "dcolon",
            "dollar",
            "eq",
            "gt",
            "hash",
            "lt",
            "mdash",
            "n",
            "ndash",
            "perc",
            "pipe",
            "quot",
        ],
        "no_args": [
            "callergraph",
            "callgraph",
            "docbookonly",
            "else",
            "endcode",
            "endcond",
            "enddocbookonly",
            "enddot",
            "endhtmlonly",
            "endif",
            "endinternal",
            "endlatexonly",
            "endlink",
            "endmanonly",
            "endmsc",
            "endparblock",
            "endrtfonly",
            "endsecreflist",
            "enduml",
            "endverbatim",
            "endxmlonly",
            "hidecallergraph",
            "hidecallgraph",
            "hideinitializer",
            "hiderefby",
            "hiderefs",
            "internal",
            "latexonly",
            "lineinfo",
            "manonly",
            "nosubgrouping",
            "parblock",
            "private",
            "privatesection",
            "protected",
            "protectedsection",
            "public",
            "publicsection",
            "pure",
            "rtfonly",
            "secreflist",
            "showinitializer",
            "showrefby",
            "showrefs",
            "static",
            "verbatim",
            "xmlonly",
        ],
        "with_single_line_text": [
            "addindex",
            "elseif",
            "fn",
            "if",
            "ifnot",
            "ingroup",
            "line",
            "noop",
            "property",
            "raisewarning",
            "skip",
            "skipline",
            "typedef",
            "until",
            "var",
        ],
        "cond": ["cond"],
        "section_like": ["paragraph", "section", "subsection", "subsubsection"],
        "paragraphs_no_args": [
            "arg",
            "attention",
            "author",
            "authors",
            "brief",
            "bug",
            "copyright",
            "date",
            "deprecated",
            "details",
            "invariant",
            "li",
            "note",
            "post",
            "pre",
            "remark",
            "remarks",
            "result",
            "return",
            "returns",
            "sa",
            "see",
            "short",
            "since",
            "test",
            "todo",
            "version",
            "warning",
        ],
        "with_word": ["a", "anchor", "b", "c", "e", "em", "p"],
        "with_name": [
            "concept",
            "def",
            "enum",
            "extends",
            "idlexcept",
            "implements",
            "memberof",
            "namespace",
            "package",
            "refitem",
            "related",
            "relatedalso",
            "relates",
            "relatesalso",
        ],
        "with_filename": [
            "docbookinclude",
            "includedoc",
            "includelineno",
            "latexinclude",
            "maninclude",
            "rtfinclude",
            "verbinclude",
            "xmlinclude",
        ],
        "with_headerfile_headername": [
            "category",
            "class",
            "interface",
            "protocol",
            "struct",
            "union",
        ],
        "with_exceptionobject": ["exception", "throw", "throws"],
        "with_file_caption": ["diafile", "dotfile", "mscfile"],
        "with_caption": ["dot", "msc"],
        "with_linkobject": ["copybrief", "copydetails", "copydoc", "link"],
        "with_name_title": ["addtogroup", "weakgroup"],
        "with_name_text": ["ref", "subpage"],
        "with_filename_blockid": ["snippetdoc", "snippetlineno"],
        "with_lineno_filename": ["dontinclude", "example"],
        "cite": ["cite"],
        "code": ["code"],
        "defgroup": ["defgroup"],
        "dir": ["dir"],
        "doxyconfig": ["doxyconfig"],
        "emoji": ["emoji"],
        "fbrclose": ["fbrclose"],
        "fbropen": ["fbropen"],
        "fcurlyclose": ["fcurlyclose"],
        "fcurlyopen": ["fcurlyopen"],
        "fdollar": ["fdollar"],
        "file": ["file"],
        "fileinfo": ["fileinfo"],
        "frndclose": ["frndclose"],
        "frndopen": ["frndopen"],
        "headerfile": ["headerfile"],
        "htmlinclude": ["htmlinclude"],
        "htmlonly": ["htmlonly"],
        "image": ["image"],
        "include": ["include"],
        "mainpage": ["mainpage"],
        "name": ["name"],
        "overload": ["overload"],
        "page": ["page"],
        "par": ["par"],
        "param": ["param"],
        "qualifier": ["qualifier"],
        "retval": ["retval"],
        "showdate": ["showdate"],
        "snippet": ["snippet"],
        "startuml": ["startuml"],
        "tableofcontents": ["tableofcontents"],
        "tilde": ["tilde"],
        "tparam": ["tparam"],
        "vhdlflow": ["vhdlflow"],
        "xrefitem": ["xrefitem"],
    }
    has_end = [
        "code",
        "cond",
        "docbookonly",
        "dot",
        "htmlonly",
        "if",
        "internal",
        "latexonly",
        "link",
        "manonly",
        "msc",
        "parblock",
        "rtfonly",
        "secreflist",
        "verbatim",
        "xmlonly",
    ]

    def __init__(self,style = styles.KeepAll):
        self._construct_grammer()
        self._output_style = None
        self.style = style

    def _pyp_cmd(self, cmd, words=True):
        if isinstance(cmd, list):
            cmds = cmd
        else:
            cmds = [cmd]
        if words:
            expr = r"[\\@](" + "|".join(cmds) + r")\b"
        else:
            expr = r"[\\@](" + "|".join(cmds) + r")"
        #print(expr)
        return pyp.Regex(expr)

    def _pyp_section_indicator(self):
        """An pyparsing expression for a section indicatior.

        While mentioning the term, doxygen documentation does not clearly define
        what a 'section indicator' is; see https://www.doxygen.nl/manual/commands.html.

        Here, we interpret section indicators as all commands other
        than escaped characters and commands with a single <word> argument.

        todo:
            consider inline HTML
        """
        section_indicators = []
        for kind, cmds in self.kinds.items():
            if kind not in ["escaped", "with_word"]:
                section_indicators += cmds
        return self._pyp_cmd(section_indicators)

    def _construct_grammer(self):
        """
        Todo:
            Use better filename expression than Word of printables, as this
            does not allow whitespace.
        """
        LPAR, RPAR = pyp.Literal("{"), pyp.Literal("{")
        LBPAR, RBPAR = pyp.Literal("["), pyp.Literal("]")
        DQUOT = pyp.Literal('"')
        IDENT = pyp.pyparsing_common.identifier
        INTEGER = pyp.pyparsing_common.integer
        BLANK_LINE = (pyp.LineStart() + pyp.LineEnd())
        UNTIL_LINE_END = (
            pyp.SkipTo(pyp.LineEnd(), failOn=BLANK_LINE)
        )
        OPT_UNTIL_LINE_END = pyp.Optional(UNTIL_LINE_END,default=None)
        SECTION_INDICATOR = self._pyp_section_indicator()
        SECTION_TERMINATOR = SECTION_INDICATOR | BLANK_LINE | pyp.StringEnd()
        UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE = pyp.SkipTo(
            SECTION_TERMINATOR
        )
        WORD_OF_PRINTABLES = pyp.Word(pyp.printables, pyp.printables)
        OPT_WORD_OF_PRINTABLES = pyp.Optional(WORD_OF_PRINTABLES,default=None)

        # ex: \&
        escaped = self._pyp_cmd(self.kinds["escaped"])
        # ex: \callergraph
        no_args = self._pyp_cmd(self.kinds["no_args"])
        # ex: # \addindex (text)
        with_single_line_text = (
            self._pyp_cmd(self.kinds["with_single_line_text"]) + UNTIL_LINE_END
        )
        # \paragraph <paragraph-name> (paragraph title)
        section_like = (
            self._pyp_cmd(self.kinds["section_like"]) + IDENT + UNTIL_LINE_END
        )
        # ex: \arg { item-description }
        paragraphs_no_args = (
            self._pyp_cmd(self.kinds["paragraphs_no_args"])
            + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )
        # ex: \a <word>
        with_word = self._pyp_cmd(self.kinds["with_word"]) + WORD_OF_PRINTABLES
        # ex: \concept <name>
        with_name = self._pyp_cmd(self.kinds["with_name"]) + IDENT
        # ex: \docbookinclude <file-name>
        with_filename = self._pyp_cmd(self.kinds["with_filename"]) + WORD_OF_PRINTABLES
        # ex: \category <name> [<header-file>] [<header-name>]
        with_headerfile_headername = (
            self._pyp_cmd(self.kinds["with_headerfile_headername"])
            + IDENT
            + OPT_WORD_OF_PRINTABLES
            + OPT_WORD_OF_PRINTABLES
        )
        # ex: \exception <exception-object> { exception description }
        with_exceptionobject = (
            self._pyp_cmd(self.kinds["with_exceptionobject"])
            + IDENT
            + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )
        # ex: \diafile <file> ["caption"] [<sizeindication>=<size>]
        size_indication = pyp.Regex(r"(width|height)=[0-9]+[a-z]{1,2}")
        opt_size_indications = pyp.Optional(size_indication) + pyp.Optional(
            size_indication
        )
        opt_caption = pyp.Optional(pyp.QuotedString('"'))
        with_file_caption = (
            self._pyp_cmd(self.kinds["with_file_caption"])
            + WORD_OF_PRINTABLES
            + opt_caption
            + opt_size_indications
        )
        # ex: \dot ["caption"] [<sizeindication>=<size>]
        with_caption = (
            self._pyp_cmd(self.kinds["with_caption"])
            + opt_caption
            + opt_size_indications
        )
        # ex: \copybrief <link-object>
        LINK_OBJECT = pyp.Regex("\w+\s*(\(\))?")
        with_linkobject = self._pyp_cmd(self.kinds["with_linkobject"]) + LINK_OBJECT
        # ex: \addtogroup <name> [(title)]
        with_name_title = (
            self._pyp_cmd(self.kinds["with_name_title"]) + IDENT + OPT_UNTIL_LINE_END
        )
        # ex: \ref <name> ["(text)"]
        with_name_text = (
            self._pyp_cmd(self.kinds["with_name_text"]) + IDENT + OPT_UNTIL_LINE_END
        )
        # ex: \snippetdoc <file-name> ( block_id )
        with_filename_blockid = (
            self._pyp_cmd(self.kinds["with_filename_blockid"])
            + WORD_OF_PRINTABLES
            + OPT_UNTIL_LINE_END
        )
        # ex: \dontinclude['{lineno}'] <file-name>
        # note: No whitespace between first bracket and
        with_lineno_filename = (
            self._pyp_cmd(self.kinds["with_lineno_filename"])
            + pyp.Optional(LPAR + INTEGER + RPAR)
            + WORD_OF_PRINTABLES
        )

        ## 

        # ex: \cite <label>
        cite = self._pyp_cmd("cite") + WORD_OF_PRINTABLES

        # \code['{'<word>'}']
        # ex: \code{.py}
        code = self._pyp_cmd("code") + pyp.Optional(LPAR + WORD_OF_PRINTABLES + RPAR)

        # \cond [(section-label)]
        cond = self._pyp_cmd("cond") + OPT_UNTIL_LINE_END

        # \defgroup <name> (group title)
        defgroup = self._pyp_cmd("defgroup") + IDENT + UNTIL_LINE_END

        # \dir [<path fragment>]
        dir = self._pyp_cmd("dir") + IDENT + UNTIL_LINE_END

        # \doxyconfig <config_option>
        doxyconfig = self._pyp_cmd("doxyconfig") + IDENT

        # \emoji "name"
        emoji = self._pyp_cmd("emoji") + pyp.QuotedString('"')

        # \f]
        fbrclose = self._pyp_cmd(r"f\]",words=False)

        # \f[
        fbropen = self._pyp_cmd(r"f\[",words=False)

        # \f}
        fcurlyclose = self._pyp_cmd(r"f\}",words=False)

        # \f{environment}{
        fcurlyopen = self._pyp_cmd(r"f\{",words=False) + IDENT + RPAR + LPAR

        # \f$
        fdollar = self._pyp_cmd(r"f\$",words=False)

        # \file [<name>]
        file = self._pyp_cmd("file") + OPT_WORD_OF_PRINTABLES

        # \fileinfo['{'option'}']
        fileinfo = self._pyp_cmd("code") + pyp.Optional(
            LPAR + WORD_OF_PRINTABLES + RPAR
        )

        # \f)
        frndclose = self._pyp_cmd(r"f\)",words=False)

        # \f(
        frndopen = self._pyp_cmd(r"f\(",words=False)

        # \headerfile <header-file> [<header-name>]
        headerfile = (
            self._pyp_cmd("headerfile") + WORD_OF_PRINTABLES + OPT_WORD_OF_PRINTABLES
        )

        # \htmlinclude ["[block]"] <file-name>
        htmlinclude = (
            self._pyp_cmd("htmlinclude")
            + pyp.Optional(LBPAR + IDENT + RBPAR)
            + WORD_OF_PRINTABLES
        )

        # \htmlonly ["[block]"]
        htmlonly = self._pyp_cmd("htmlonly") + pyp.Optional(LBPAR + IDENT + RBPAR)

        # \image['{'option[,option]'}'] <format> <file> ["caption"] [<sizeindication>=<size>]
        image_options = pyp.Group(pyp.Optional(LPAR + pyp.delimitedList(IDENT) + RPAR))
        image = (
            self._pyp_cmd("image")
            + image_options
            + IDENT
            + WORD_OF_PRINTABLES
            + opt_caption
            + opt_size_indications
        )

        # \include['{'option'}'] <file-name>
        include = (
            self._pyp_cmd("include")
            + pyp.Optional(LPAR + IDENT + RPAR)
            + WORD_OF_PRINTABLES
        )

        # \mainpage [(title)]
        mainpage = self._pyp_cmd("mainpage") + OPT_UNTIL_LINE_END

        # \name [(header)]
        name = self._pyp_cmd("name") + OPT_UNTIL_LINE_END

        # \overload [(function declaration)]
        overload = self._pyp_cmd("overload") + OPT_UNTIL_LINE_END

        # \page <name> (title)
        page = self._pyp_cmd("page") + IDENT + UNTIL_LINE_END

        # \par [(paragraph title)] { paragraph }
        par = (
            self._pyp_cmd("par")
            + OPT_UNTIL_LINE_END
            + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )

        # \param '['dir']' <parameter-name> { parameter description }
        param_dir = pyp.Regex(r"\[\s*(in|out|(\s*in,\s*out))\s*\]")
        param_names = pyp.delimitedList(IDENT)
        param = (
            self._pyp_cmd("param")
            + param_dir
            + param_names
            + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )

        # \qualifier <label> | "(text)"
        qualifier = self._pyp_cmd("qualifier") + (
            IDENT | (DQUOT + UNTIL_LINE_END + DQUOT)
        )

        # \retval <return value> { description }
        retval = (
            self._pyp_cmd("retval") + IDENT + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )

        # \showdate "<format>" [ <date_time> ]
        showdate = self._pyp_cmd("showdate") + pyp.QuotedString('"') + UNTIL_LINE_END

        # \snippet['{'option'}'] <file-name> ( block_id )
        snippet = (
            self._pyp_cmd("snippet")
            + pyp.Optional(LPAR + IDENT + RPAR)
            + WORD_OF_PRINTABLES
            + UNTIL_LINE_END
        )

        # \startuml ['{'option[,option]'}'] ["caption"] [<sizeindication>=<size>]
        startuml_options = image_options.copy()
        startuml = (
            self._pyp_cmd("startuml")
            + startuml_options
            + opt_caption
            + opt_size_indications
        )

        # \tableofcontents['{'[option[:level]][,option[:level]]*'}']
        tableofcontents = self._pyp_cmd("tableofcontents")

        # \~[LanguageId]
        tilde = self._pyp_cmd("\~[a-z]*")

        # \tparam <template-parameter-name> { description }
        tparam = (
            self._pyp_cmd("tparam") + IDENT + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )

        # \vhdlflow [(title for the flow chart)]
        vhdlflow = self._pyp_cmd("vhdlflow") + OPT_UNTIL_LINE_END

        # \xrefitem <key> "heading" "list title" { text }
        xrefitem = (
            self._pyp_cmd("xrefitem")
            + IDENT
            + pyp.QuotedString('"')
            + pyp.QuotedString('"')
            + UNTIL_NEXT_SECTION_INDICATOR_OR_BLANK_LINE
        )

        formatters = escaped | with_word
        transformers = no_args
        exprs = locals()
        for expr_name in exprs:
            if expr_name not in ("escaped","with_word") and expr_name in self.kinds:
                # note: Anchors have unique names, so order is not important
                transformers = transformers | exprs[expr_name]
        all = formatters | transformers
        self.__dict__.update(locals())

    def get_parser_name_for_command(self, cmd: str):
        """Return pyparsing parser name for the given command name.

        Returns:
            A triple consisting of the pyparsing parser, its attribute name, and a list of the names of all commands
            that share the parser.
        Note:
            Command names can be obtained by visiting
            https://www.doxygen.nl/manual/commands.html
            and clicking on the individual commands in the alphabetic list.
            The command name then appears as #cmd{name} in the URL shown
            by the browser.
            Example: ``https://www.doxygen.nl/manual/commands.html#cmdfdollar``
        Note:
            Multiple commands may share the same parser.
        Raises:
            KeyError: If 'cmd' could not be mapped to a parser.
        See:
            https://www.doxygen.nl/manual/commands.html
        """
        if cmd.startswith("cmd"):
            cmd = cmd[3:]
        for kind, cmds in self.kinds:
            if cmd in cmds:
                return kind
        raise KeyError("No parser found for command '{cmd}'")

    def walk_pyparsers(self):
        for kind in self.kinds:
            yield self.__dict__[kind]

    @property
    def style(self):
        return self._output_style

    @style.setter
    def style(self,output_style):
        self._output_style = output_style
        # suppress all
        for kind in self.kinds:
            pyparser = self.__dict__[kind]
            try:
                pyparser.setParseAction(getattr(output_style, kind))
            except AttributeError:
                pyparser.setParseAction(styles.KeepAll.identity)

    def _create_text_blocks(self,text: str):
        """Splits the text into verbatim and non-verbatim blocks.
        """
        blocks = []
        previous_end = 0
        verbatim_environment = None
        open = self._pyp_cmd(
            r"((dot|verbatim|code)\b|f[$[({])", words=False
        ).setParseAction(lambda tokens: (True,tokens))
        close = self._pyp_cmd(
            r"(end(dot|verbatim|code)\b|f[$)\]}])", words=False
        ).setParseAction(lambda tokens: (False,tokens))
        open_close = open | close
        for tokens, start, end in open_close.scanString(text):
            tokens = tokens[0]
            #print(tokens)
            is_open = tokens[0]
            prefixed_cmd = tokens[1][0]
            cmd = prefixed_cmd[1:]
            if cmd == "f$":
                is_open = (verbatim_environment == None)
            if is_open:
                if verbatim_environment == None:
                    blocks.append( (text[previous_end:end], False) ) # keep the command itself in previous non-verbatim block
                    verbatim_environment = cmd
                    previous_end = end
                    continue
            # Regarding 'f$', note the 'continue' in the line above
            if not is_open:
                if verbatim_environment == None and cmd != "f$":
                    warnings.warn(f"found '{tokens[0]}' but the respective environment has not been opened")
                if verbatim_environment != None:
                    if cmd.startswith("end"):
                        is_matching_close = cmd[3:] == verbatim_environment
                    else:
                        is_matching_close = cmd.replace("]","[").replace(")","(").replace("}","{") == verbatim_environment[0:2]
                    #
                    if is_matching_close:
                        blocks.append( (text[previous_end:start], True) ) # put the command itself in next non-verbatim block
                        previous_end = start
                        verbatim_environment = None
        if verbatim_environment != None:
            raise RuntimeError(f"environment {verbatim_environment} has never been closed")
        blocks.append((text[previous_end:],False))
        return blocks

    def transform_string(self,text: str,**kwargs):    
        result = ""
        for content, verbatim in self._create_text_blocks(text):
            #print(f"{content=}")
            if verbatim:
                result += content
            else:
                partial_result = self.formatters.transformString(content,**kwargs)
                #print(f"{partial_result=}")
                partial_result = self.transformers.transformString(partial_result,**kwargs)
                result += partial_result
        return result
    
    def search_string(self,text: str,**kwargs):
        """
        See:
            https://pyparsing-docs.readthedocs.io/en/latest/pyparsing.html#pyparsing.ParserElement.search_string
        """
        return self.all.searchString(text,**kwargs)

    def scan_string(self,text: str,**kwargs):
        """
        See:
            https://pyparsing-docs.readthedocs.io/en/latest/pyparsing.html#pyparsing.ParserElement.scan_string
        """
        return self.all.scanString(text,**kwargs)

    def parse_expr(self,text: str, **kwargs):
        """Parses a single doxygen command expression.

        See:
            https://pyparsing-docs.readthedocs.io/en/latest/pyparsing.html#pyparsing.ParserElement.parse_string
        """ 
        return self.all.parseString(text,**kwargs)
