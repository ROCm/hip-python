# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import textwrap

import pyparsing as pyp

import warnings

pyp.ParserElement.setDefaultWhitespaceChars(' \t')

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
    """

    class PythonDocstrings:
        """Base class for Python docstring output that
        defines parse actions for some DoxygenGrammar parsers such as
        ``escaped`` and ``with_word``
        """

        @staticmethod
        def escaped(tokens):
            expr = tokens[0]
            if expr == r"\\n":
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

# for structuring the input

class Node:

    def __init__(self,s,loc,tokens):
        self.s = s
        self.parent = None
        self.children = []
        self.begin = loc
        self.end = None
        self.tokens = tokens

    @property
    def text(self):
        if self.end != None:
            return self.s[self.begin:self.end]
        else:
            raise RuntimeError("'end' must not be `None`")

    @property
    def level(self):
        curr = self
        level = 0
        while curr.parent != None:
            curr = curr.parent
            level += 1
        assert isinstance(self,Root) or level > 0, f"{str(self)}"
        return level
    
    def walk(self,postorder=False):
        if not postorder:
            yield self
        for child in self.children:
            yield from child.walk()
        if postorder:
            yield self
        
class Root(Node):

    def __init__(self,s):
        self.s = s
        self.parent = None
        self.children = []

    def add_details_section(self,start,end):
        self.children.append(
            Section(
                self.s,
                start,
                [ "\\details", self.s[start:end] ]
            )
        )
        self.children[-1].parent = self
        self.children[-1].end = end
        self.children[-1].add_body()
        return self.children[-1]

class TextBlock(Node):
    
    def __init__(self,s,loc,tokens):
        Node.__init__(self,s,loc,tokens)

class Section(Node):
    
    def __init__(self,s,loc,tokens):
        Node.__init__(self,s,loc,tokens)
        self.kind = tokens[0][1:]
        self.end = None

    def add_body(self):
        assert len(self.children) == 0
        self.children.append(
            SectionBody(
                self.s,
                self.begin,
                self.tokens,
            )
        )
        self.children[-1].parent = self
        self.children[-1].end = self.end
        return self.children[-1]
    
    @property
    def body(self):
        result = self.children[0]
        assert isinstance(result,SectionBody)
        return result
    
    def set_body_from_tokens(self):
        body = self.tokens[-1]
        assert isinstance(body,SectionBody)
        self.children.append(body)
        self.children[-1].parent = self

class SectionBody(Node):

    def __init__(self,s,loc,tokens):
        Node.__init__(self,s,loc,tokens)
        self.kind = tokens[0][1:]
        self.end = None

    def add_text_block(self,s,start,tokens,end):
        self.children.append(
            TextBlock(
                s,
                start,
                tokens,
            )
        )
        self.children[-1].end = end
        return self.children[-1]

class VerbatimBlock(Node):
    r"""Verbatim text block expression such
    as \verbatim ... \endverbatim, \code ... \endcode, ...
    """
    
    def __init__(self,s,loc,tokens):
        Node.__init__(self,s,loc,tokens)
        self.kind = tokens[0][1:]

    @property
    def body(self):
        return self.tokens[-2]
    
    @property
    def head(self):
        return self.tokens[:-2]
    
    @property
    def tail(self):
        return self.tokens[-1]

class MathBlock(Node):

    r"""Math block expression such
    as \f[ ... \f], \f{eqnarray} ... \f}, ...
    """
    
    def __init__(self,s,loc,tokens):
        Node.__init__(self,s,loc,tokens)
        self.kind = tokens[0][1:]
        if len(tokens) == 5:
            # '\f{' 'env' '}' '...' '\f}'
            self.env = tokens[2]
    
    @property
    def body(self):
        return self.tokens[-2]
    
    @property
    def head(self):
        return self.tokens[:-2]
    
    @property
    def tail(self):
        return self.tokens[-1]

# Parser

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
        "verbatim_end": [
            "endcode", # yes
            "enddocbookonly", # yes
            "enddot", # yes
            "endhtmlonly", # yes
            "endlatexonly", # yes
            "endmanonly", # yes
            "endmsc", # yes
            "endrtfonly", # yes
            "enduml", # yes
            "endverbatim", # yes
            "endxmlonly", # yes
        ],
        "docbookonly": ["docbookonly"],
        "latexonly": ["latexonly"],
        "manonly": ["manonly"],
        "rtfonly": ["rtfonly"],
        "verbatim": ["verbatim"],
        "xmlonly": ["xmlonly"],
        "no_args": [
            "callergraph",
            "callgraph",
            "else",
            "endcond", # no
            "endif", # no
            "endinternal", # no
            "endparblock", # no
            "endsecreflist", # no
            "hidecallergraph",
            "hidecallgraph",
            "hideinitializer",
            "hiderefby",
            "hiderefs",
            "internal",
            "lineinfo",
            "nosubgrouping",
            "parblock",
            "private",
            "privatesection",
            "protected",
            "protectedsection",
            "public",
            "publicsection",
            "pure",
            "secreflist",
            "showinitializer",
            "showrefby",
            "showrefs",
            "static",
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
        "page_section": ["paragraph", "section", "subsection", "subsubsection"],
        "section_no_args": [
            "alpha", # custom
            "beta", # custom
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
        "dot": ["dot"],
        "msc": ["msc"],
        "with_linkobject": ["copybrief", "copydetails", "copydoc", 
                            "link"],
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
        "startuml": ["startuml"], # yes
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

    def __init__(self,cmd_prefix_chars=r"\\@"):
        self.cmd_prefix_chars = cmd_prefix_chars
        self._construct_grammer()
        # __tree: private instance that is not exposed to the user
        self.__tree = DoxygenGrammar.__new__(DoxygenGrammar)
        self.__tree.cmd_prefix_chars = cmd_prefix_chars
        self.__tree._construct_grammer()

    def _pyp_cmd(self, cmd, words=True):
        if isinstance(cmd, list):
            cmds = cmd
        else:
            cmds = [cmd]
        if words:
            expr = r"["+self.cmd_prefix_chars+"](" + "|".join(cmds) + r")\b"
        else:
            expr = r"["+self.cmd_prefix_chars+"](" + "|".join(cmds) + r")"
        #print(expr)
        return pyp.Regex(expr)

    def _construct_grammer(self):
        """
        Todo:
            Use better filename expression than Word of printables, as this
            does not allow whitespace.
        """
        LBRACE, RBRACE = pyp.Literal("{"), pyp.Literal("}")
        LBPAR, RBPAR = pyp.Literal("["), pyp.Literal("]")
        DQUOT = pyp.Literal('"')
        IDENT = pyp.pyparsing_common.identifier
        INTEGER = pyp.pyparsing_common.integer
        BLANK_LINE = pyp.Regex("\n[ \t]*\n")
        # pyp.Optional(pyp.LineEnd()) + pyp.LineStart()
        UNTIL_LINE_END = pyp.SkipTo(pyp.LineEnd())
        OPT_UNTIL_LINE_END = pyp.Optional(UNTIL_LINE_END,default=None)
        section = pyp.Forward()
        SECTION_TERMINATOR = BLANK_LINE | section | pyp.StringEnd()
        section_body = pyp.SkipTo(
            SECTION_TERMINATOR
        )
        WORD_OF_PRINTABLES = pyp.Word(pyp.printables, pyp.printables)
        OPT_WORD_OF_PRINTABLES = pyp.Optional(WORD_OF_PRINTABLES,default=None)

        # ex: \&
        CHARS = (
            r"&",#"amp"
            r"@",#"at"
            r"\\",#"backslash"
            r"\.",#"chardot"
            r"\$",#"dollar"
            r"=",#"eq"
            r">",#"gt"
            r"#",#"hash"
            r"<",#"lt"
            r"n",#"n"
            r"%",#"perc"
            r"\|",#"pipe"
            r"\"",#"quot"
        )
        escaped = pyp.Regex(r"\\(::|---?|["+ "".join(CHARS) + r"])").setParseAction(
            lambda tk: tk[0][1:] if tk != "\\n" else "\n"
        )
        del CHARS
        self._pyp_cmd(self.kinds["escaped"])
        # ex: \callergraph
        no_args = self._pyp_cmd(self.kinds["no_args"])
        ENDDOCBOOKONLY = self._pyp_cmd("enddocbookonly")
        ENDLATEXONLY   = self._pyp_cmd("endlatexonly")
        ENDMANONLY     = self._pyp_cmd("endmanonly")
        ENDRTFONLY     = self._pyp_cmd("endrtfonly")
        ENDVERBATIM    = self._pyp_cmd("endverbatim")
        ENDXMLONLY     = self._pyp_cmd("endxmlonly")

        docbookonly = self._pyp_cmd("docbookonly") + pyp.SkipTo(ENDDOCBOOKONLY) + ENDDOCBOOKONLY
        latexonly = self._pyp_cmd("latexonly") + pyp.SkipTo(ENDLATEXONLY) + ENDLATEXONLY
        manonly = self._pyp_cmd("manonly") + pyp.SkipTo(ENDMANONLY) + ENDMANONLY
        rtfonly = self._pyp_cmd("rtfonly") + pyp.SkipTo(ENDRTFONLY) + ENDRTFONLY
        verbatim = self._pyp_cmd("verbatim") + pyp.SkipTo(ENDVERBATIM) + ENDVERBATIM
        xmlonly = self._pyp_cmd("xmlonly") + pyp.SkipTo(ENDXMLONLY) + ENDXMLONLY
        verbatim_no_args = docbookonly | latexonly | manonly | rtfonly | verbatim | xmlonly

        verbatim_end = self._pyp_cmd(self.kinds["verbatim_end"])
        # ex: # \addindex (text)
        with_single_line_text = (
            self._pyp_cmd(self.kinds["with_single_line_text"]) + UNTIL_LINE_END
        )
        # \paragraph <paragraph-name> (paragraph title)
        page_section = (
            self._pyp_cmd(self.kinds["page_section"]) + IDENT + UNTIL_LINE_END
        )
        # ex: \arg { item-description }
        section_no_args = (
            self._pyp_cmd(self.kinds["section_no_args"])
            + section_body
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
            + section_body
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

        ENDDOT = self._pyp_cmd("enddot")
        ENDMSC = self._pyp_cmd("endmsc")
        dot = (
            self._pyp_cmd("dot")
            + opt_caption
            + opt_size_indications
            + pyp.SkipTo(ENDDOT) + ENDDOT
            + ENDDOT
        )
        msc = (
            self._pyp_cmd("msc")
            + opt_caption
            + opt_size_indications
            + pyp.SkipTo(ENDMSC) + ENDMSC
            + ENDMSC
        )
        verbatim_with_caption = ENDDOT | ENDMSC
        
        # ex: \copybrief <link-object>
        LINK_OBJECT = pyp.Regex(r"\w+\s*(\(\))?")
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
            + pyp.Optional(LBRACE + INTEGER + RBRACE)
            + WORD_OF_PRINTABLES
        )

        # ex: \cite <label>
        cite = self._pyp_cmd("cite") + WORD_OF_PRINTABLES

        # \code['{'<word>'}']
        # ex: \code{.py}
        ENDCODE = self._pyp_cmd("endcode")
        CODE = self._pyp_cmd("code") + pyp.Optional(LBRACE + pyp.Regex(r"\.\w+") + RBRACE, default=[None,None,None])
        code = CODE + pyp.SkipTo(ENDCODE) + ENDCODE


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
        FBRCLOSE = self._pyp_cmd(r"f\]",words=False)
        # \f[
        FBROPEN = self._pyp_cmd(r"f\[",words=False)
        # \f}
        FCURLYCLOSE = self._pyp_cmd(r"f\}",words=False)
        # \f{environment}{
        FCURLYOPEN = self._pyp_cmd(r"f\{",words=False)
        # \f$
        FDOLLAR = self._pyp_cmd(r"f\$",words=False)
        # \f)
        FRNDCLOSE = self._pyp_cmd(r"f\)",words=False)
        # \f(
        FRNDOPEN = self._pyp_cmd(r"f\(",words=False)

        fdollar = FDOLLAR +  pyp.SkipTo(FDOLLAR) + FDOLLAR
        fbr = FBROPEN +  pyp.SkipTo(FBRCLOSE) + FBRCLOSE
        frnd = FRNDOPEN +  pyp.SkipTo(FRNDCLOSE) + FRNDCLOSE
        fcurly = FCURLYOPEN + IDENT + RBRACE + LBRACE + pyp.SkipTo(FCURLYCLOSE) + FCURLYCLOSE

        # \{
        groupopen = self._pyp_cmd(r"\{",words=False)
        
        # \}
        groupclose = self._pyp_cmd(r"\}",words=False)

        # \file [<name>]
        file = self._pyp_cmd("file") + OPT_WORD_OF_PRINTABLES

        # \fileinfo['{'option'}']
        fileinfo = self._pyp_cmd("code") + pyp.Optional(
            LBRACE + WORD_OF_PRINTABLES + RBRACE
        )

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
        image_options = pyp.Group(pyp.Optional(LBRACE + pyp.delimitedList(IDENT) + RBRACE))
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
            + pyp.Optional(LBRACE + IDENT + RBRACE)
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
            + section_body
        )

        # \param '['dir']' <parameter-name> { parameter description }
        PARAM_DIR = pyp.Regex(r"\[\s*(in|out|(\s*in,\s*out))\s*\]").setParseAction(
            lambda tokens: tokens[0][1:-1].replace(" ","")
        ) # remove [] and whitespace
        PARAM_NAMES = pyp.delimitedList(IDENT)
        param = (
            self._pyp_cmd("param")
            + pyp.Optional(PARAM_DIR,default=None)
            + PARAM_NAMES
            + section_body
        )
        del PARAM_DIR
        del PARAM_NAMES

        # \qualifier <label> | "(text)"
        qualifier = self._pyp_cmd("qualifier") + (
            IDENT | (DQUOT + UNTIL_LINE_END + DQUOT)
        )

        # \retval <return value> { description }
        retval = (
            self._pyp_cmd("retval") + IDENT + section_body
        )

        # \showdate "<format>" [ <date_time> ]
        showdate = self._pyp_cmd("showdate") + pyp.QuotedString('"') + UNTIL_LINE_END

        # \snippet['{'option'}'] <file-name> ( block_id )
        snippet = (
            self._pyp_cmd("snippet")
            + pyp.Optional(LBRACE + IDENT + RBRACE)
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
            self._pyp_cmd("tparam") + IDENT + section_body
        )

        # \vhdlflow [(title for the flow chart)]
        vhdlflow = self._pyp_cmd("vhdlflow") + OPT_UNTIL_LINE_END

        # \xrefitem <key> "heading" "list title" { text }
        xrefitem = (
            self._pyp_cmd("xrefitem")
            + IDENT
            + pyp.QuotedString('"')
            + pyp.QuotedString('"')
            + section_body
        )

        section <<= section_no_args | param | tparam | retval | xrefitem | par | with_exceptionobject
        verbatim = code | verbatim_no_args | verbatim_with_caption | startuml
        math_block = fbr | frnd | fcurly
        formatting = escaped | with_word | fdollar
        other = (
            no_args
            |with_single_line_text
            |cond
            |page_section
            |with_name
            |with_filename
            |with_headerfile_headername
            |with_file_caption
            |with_linkobject
            |with_name_title
            |with_name_text
            |with_filename_blockid
            |with_lineno_filename
            |cite
            |defgroup
            |dir
            |doxyconfig
            |emoji
            |file
            |fileinfo
            |headerfile
            |htmlinclude
            |image
            |include
            |mainpage
            |name
            |overload
            |page
            |qualifier
            |showdate
            |snippet
            |tableofcontents
            |tilde
            |vhdlflow
        )
        all = section | verbatim | math_block | formatting | other
        self.__dict__.update(locals())

    def walk_pyparsers(self):
        for kind in self.kinds:
            yield self.__dict__[kind]

    def parse(self,text: str):
        """Parse a snippet of doxygen documentation.

        Note:
            Limitation: Placing doxygen section commands into verbatim, math, and 
                        code sections will break the parser as it will detect section inside of these environments.
                        If it is really necessary, we advise to change the case of
                        at least one of the command name letters or insert some other characters.
        """
        tree = self.__tree
        tree.section.setParseAction(Section)
        tree.section_body.setParseAction(SectionBody)
        tree.verbatim.setParseAction(VerbatimBlock)
        tree.math_block.setParseAction(MathBlock)
        verbatim_or_math = tree.verbatim|tree.math_block
    
        def scan_for_verbatim_or_math_(section_body: SectionBody,text,text_start,text_end):
            # look for verbatim/code
            section_text = text[text_start:text_end]
            previous_end = 0
            print(f"{section_text=}")
            for tokens, start, end in verbatim_or_math.scanString(section_text):
                print(tokens)
                if start != previous_end:
                    block = section_body.add_text_block(text,text_start+previous_end,tokens,text_start+start)
                    block.parent = section_body
                block = tokens[0]
                assert isinstance(block,(TextBlock,VerbatimBlock,MathBlock))
                block.s = text
                block.begin = text_start + start
                block.end = text_start + end
                block.parent = section_body
                section_body.children.append(block)
                previous_end = end
                print(f"{block=}")
            if not len(section_body.children):
                block = section_body.add_text_block(text,text_start,[section_body.tokens[-1]],text_end)
                block.parent = section_body

        root = Root(text)
        previous_end = 0
        for tokens, start, end in tree.section.scanString(text):
            if start != previous_end:
                # insert fake details section
                section = root.add_details_section(previous_end,start)
                scan_for_verbatim_or_math_(section.body,text,previous_end,end)
            section = tokens[0]
            section.parent = root
            section.end = end
            section.set_body_from_tokens()
            scan_for_verbatim_or_math_(section.body,text,start,end)
            root.children.append(section)
            previous_end = end
        if previous_end < len(text):
            # insert fake details section
            section = root.add_details_section(previous_end,len(text))
            scan_for_verbatim_or_math_(section.body,text,previous_end,len(text))
        return root