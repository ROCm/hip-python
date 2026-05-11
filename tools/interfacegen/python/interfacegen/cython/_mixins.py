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


"""Cython backend — mixins submodule.

Carved out of the historical interfacegen.cython monolith.
Public callers should import from `interfacegen.cython`,
not from this submodule directly.
"""

__author__ = "Advanced Micro Devices, Inc."

import ctypes
import keyword
import logging
import os
import re
import sys
import textwrap
import typing

import clang.cindex
import Cython.Tempita

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults, _doxygen
from ._defaults import *  # noqa: F401,F403
from ._doxygen import *  # noqa: F401,F403

__all__ = [
    'DoxygenMixin',
    'CythonMixin',
    'Node',
]


class DoxygenMixin:
    def __init__(self, doxygen_conv: doxyparser.DoxygenGrammar):
        self.doxygen_conv = doxygen_conv

    @staticmethod
    def _dedent_first_line(text: str) -> str:
        lines = text.splitlines(keepends=True)
        result = lines[0].strip(" \t")
        if len(lines) > 1:
            result += "".join(lines[1:])
        return result

    @staticmethod
    def _render_doxygen_brief(
        sections,
        log_prefix: str = "",
        missing_text: str = "(No short description)",
    ) -> str:
        doxygen_brief: doxyparser.Section = next(
            (sec for sec in sections if sec.kind in ("brief", "short")), None
        )
        if doxygen_brief is not None:
            # clip other sections before the brief, TODO make option
            sections = sections[sections.index(doxygen_brief) + 1 :]
            if len(doxygen_brief[0]) > 1:
                _log.warning(
                    f"{log_prefix}doxygen: more than one text/verbatim/math block in section 'brief'. Ignore others."
                )
            if not isinstance(doxygen_brief.first_block, doxyparser.TextBlock):
                raise RuntimeError(
                    f"{log_prefix}doxygen: expected single text block in section 'brief'"
                )
            return doxygen_brief.first_block.transformed_text.strip() + "\n\n"
        else:
            return f"{missing_text}\n\n"

    @staticmethod
    def _render_doxygen_section_body(section, outer_indent) -> str:
        """Renders the body of a doxygen section."""
        result = ""
        for block in section.blocks:
            if isinstance(block, doxyparser.TextBlock):
                # variants we've seen
                # \note: texttext => firstline == ": texttext"
                # \note texttext
                # \note texttext
                #    texttext
                lines = (
                    block.transformed_text.lstrip(":\n\t ")
                    .rstrip()
                    .splitlines()
                )
                if len(lines):
                    firstline = lines[0]
                    other_lines = lines[1:]
                    if len(other_lines):
                        transformed_text = (
                            firstline
                            + "\n"
                            + textwrap.dedent("\n".join(other_lines))
                        )
                    else:
                        transformed_text = firstline
                    result += (
                        textwrap.indent(transformed_text, outer_indent) + "\n"
                    )
            elif isinstance(block, doxyparser.VerbatimBlock):
                result += f"\n{outer_indent}.. code-block::"
                if block.kind == "code":  # \code { lang } TEXT \endcode
                    if block.tokens == 6:
                        lang = block.tokens[2][1:]
                        result += lang
                result += "\n\n"
                inner_indent = outer_indent + " " * 3
                code = textwrap.dedent(block.code)
                result += textwrap.indent(code, inner_indent) + "\n\n"
            elif isinstance(block, doxyparser.MathBlock):
                inner_indent = outer_indent + " " * 3
                result += f"\n{outer_indent}.. math::\n"
                if block.env is not None:
                    result += "{inner_indent}:nowrap:"
                    result += rf"{inner_indent}\begin{{{block.env}}}\n"
                result += "\n"
                code = textwrap.dedent(block.code)
                result += textwrap.indent(code, inner_indent).rstrip() + "\n"
                if block.env is not None:
                    result += rf"{inner_indent}\end{{{block.env}}}\n"
                result += "\n"
        return result

    def _render_doxygen_simple_section(
        self, section: doxyparser.Section, single_level_indent: str
    ) -> str:
        docstring_addition = "\n"
        if section.kind in ("details", "details*"):
            outer_indent = ""
        else:
            docstring_addition += (
                f"\n{section.kind[0].upper() + section.kind[1:]}:\n"
            )
            outer_indent = single_level_indent
        body = DoxygenMixin._render_doxygen_section_body(section, outer_indent)
        if section.kind in ("see", "sa"):
            docstring_addition += (
                self.doxygen_conv.see_reference.transform_string(body)
            )
        else:
            docstring_addition += body
        return docstring_addition


class CythonMixin(DoxygenMixin):
    def __init__(self):
        global DOXYGEN_CONV
        self.renamer = DEFAULT_RENAMER
        self.sep = "_"
        self.util_types_prefix = ""
        # doxygen parser
        DoxygenMixin.__init__(self, DOXYGEN_CONV)

    @property
    def cython_name(self):
        return self.renamer(self.name)

    @property
    def cython_global_name(self):
        return self.renamer(self.global_name(self.sep))

    def _cython_and_c_name(self, orig_name: str):
        """Returns `<orig_name> "<renamed>"` if `renamer` had an effect, else returns `orig_name`.

        Note:
            For more details see https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#resolving-naming-conflicts-c-name-specifications
        """
        renamed = self.renamer(orig_name)
        if orig_name == renamed:
            return orig_name
        else:
            return f'{renamed} "{orig_name}"'

    def _raw_comment_cleaned(self):

        assert isinstance(self, tree.Node)
        if self.raw_comment is not None:
            cleaned_raw_comment = self.raw_comment_cleaner(self.raw_comment)
            return doxyparser.remove_doxygen_comment_chars(cleaned_raw_comment)
        else:
            return ""

    def _remove_doxygen_comment_chars(self, text: str):
        return doxyparser.remove_doxygen_comment_chars(text)

    def _as_python_comment(self, text: str, comment_chars="#"):
        if text is not None and len(text):
            return "".join(
                [
                    f"{comment_chars} " + ln
                    for ln in text.splitlines(keepends=True)
                ]
            )
        else:
            return ""

    def _raw_comment_as_python_comment(self, comment_chars="#"):

        assert isinstance(self, tree.Node)
        if self.raw_comment is not None:
            comment = self._raw_comment_cleaned()
            return "".join(
                [
                    f"{comment_chars} " + ln
                    for ln in comment.splitlines(keepends=True)
                ]
            )
        else:
            return ""

    def render_c_interface_decl(self):
        """Render a Cython interface for external C code."""
        return None

    def render_python_interface_decl(self, cprefix: str):
        """Render the declaration part for the Python interface."""
        return None

    def render_python_interface_impl(self, cprefix: str, *, module_opts: dict):
        """Render the implementation part for the Python interface."""
        return None

    def render_python_docstring(self, cprefix: str):
        """Converts doxygen comment to a Python docstring using the doxyparser API.

        Note:
            This is the default implementation, the `cython.Function` overwrites
            it to take arguments and return values into account.
        """
        # TODO handle groups; issue detecting addgroup; detecting ingroup is easier

        assert isinstance(self, DoxygenMixin)
        doxyparsetree = self.doxygen_conv.parse_structure(
            self._raw_comment_cleaned()
        )
        sections = list(doxyparsetree.children)
        # brief
        docstring_body = self._render_doxygen_brief(
            sections, log_prefix=f"<{self.render_location()}> "
        )

        # other sections
        single_level_indent = " " * 4
        for section in sections:
            # FIXME warn in such a case or simply ignore?
            # if section.kind in (
            #   "result",
            #   "return",
            #   "returns",
            #   "param"
            # ):
            if section.kind != "brief":
                docstring_body += self._render_doxygen_simple_section(
                    section, single_level_indent
                )

        # Clean result
        docstring_body = self.docstring_cleaner(docstring_body)
        # remove multiple blank lines
        docstring_body = re.sub(
            r"(\n\s*)+\n+", "\n\n", docstring_body
        ).rstrip()
        return f'r"""{docstring_body}\n"""'  # r required if verbatim/code is in body

    @staticmethod
    def to_sphinx_pyobj(expr: str):
        return python_interface_pyobj_role_template.format(name=expr)

    def render_pyi_stub(
        self, cprefix: str, *, override_name: str = None,
        base: str = None,
    ):
        """Render this node as a `.pyi` type-stub fragment.

        Returns a `list[str]` of lines, or `None` if this node should
        be omitted from the stub (most node kinds — only the handful
        of public API entities have an override below).

        Args:
            cprefix:
                The Cython C-level module prefix (e.g. ``"cyhip."``)
                used by the same docstring-rendering helpers that drive
                the .pyx output.
            override_name:
                If set, the stub renders under this name instead of
                ``self.cython_name``. The cuda interop generator uses
                this so a hip node can be re-emitted as its cuda alias
                with the hip signature/docstring intact.
            base:
                If set, class stubs render as ``class Name(<base>):``.
                The cuda interop uses this so each cuda alias inherits
                from the corresponding hip type, matching the
                ``cdef class CUDA_X(hip.X): pass`` shape in the .pyx.
        """
        return None

    def _render_pyi_class_stub(
        self, cprefix: str, override_name: str = None,
        base: str = None,
    ):
        """Shared implementation used by `Record`, `Enum`, `AnonymousEnum`,
        and `FunctionPointer`. Renders ``class Name[(base)]:`` followed
        by the docstring (`render_python_docstring`) and a placeholder
        ``__init__``."""
        name = override_name or getattr(self, "cython_name", None) or self.name
        if not name or not name.isidentifier():
            return None
        try:
            docstring = self.render_python_docstring(cprefix)
        except Exception:
            docstring = None
        head = f"class {name}({base}):" if base else f"class {name}:"
        lines = [head]
        if docstring:
            for dl in docstring.splitlines():
                lines.append(f"    {dl}" if dl else "")
        lines.append("    def __init__(self, *args, **kwargs): ...")
        return lines


Node = CythonMixin  # alias so that it can be used in treefactory


