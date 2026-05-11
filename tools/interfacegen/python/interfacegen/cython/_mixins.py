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


# Mixins


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

    # Leaking-doxygen-tag cleanup. The grammar in doxyparser.py occasionally
    # fails to consume tags like `@retval #FOO_BAR ...` (because IDENT can't
    # match a `#`-prefixed reference) or `@p word` when the doxygen comment
    # was line-wrapped between `@p` and the word. Rather than special-case
    # every grammar gap, we run a post-pass over the assembled docstring to
    # convert any surviving tags to Sphinx-friendly equivalents.
    #
    # Patterns are written against the *post-transform* text — i.e. with
    # references already rewritten to `` `~.NAME` `` (the see_reference /
    # in_text_reference parse actions ran earlier).
    _LEAK_RETVAL_RE = re.compile(
        r"^(?P<indent>[ \t]*)@retval\s+(?P<ref>`[^`\n]+`|[A-Za-z_][\w:#]*)"
        r"[ \t]*(?P<desc>[^\n]*(?:\n(?P=indent)[ \t]+[^\n]*)*)",
        re.MULTILINE,
    )
    _LEAK_SEE_RE = re.compile(
        r"@see\s+(?P<ref>`[^`\n]+`|[A-Za-z_][\w:#]*)"
    )
    _LEAK_BRIEF_RE = re.compile(r"^([ \t]*)@brief[ \t]*", re.MULTILINE)
    _LEAK_WORD_TAG_RE = re.compile(
        r"@(?P<tag>p|a|b|c|e|em)\b[ \t\n]+(?P<word>`[^`\n]+`|[A-Za-z_][\w]*)"
    )
    _LEAK_BLOCK_TAG_RE = re.compile(
        r"^(?P<indent>[ \t]*)@(?P<tag>note|warning|deprecated|since|todo)\b[ \t]*"
        r"(?P<body>[^\n]*(?:\n(?P=indent)[ \t]+[^\n]*)*)",
        re.MULTILINE,
    )
    # HTML tags that doxygen comments embed directly (e.g. hipblas
    # writes `<b> BLAS Level 2 API </b>`). RST doesn't render HTML,
    # so the tags appear verbatim in the rendered docs. Convert to
    # the RST equivalent.
    _LEAK_HTML_BOLD_RE = re.compile(
        r"<\s*b\s*>(?P<text>.*?)<\s*/\s*b\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    _LEAK_HTML_ITALIC_RE = re.compile(
        r"<\s*(?:i|em)\s*>(?P<text>.*?)<\s*/\s*(?:i|em)\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    _LEAK_HTML_CODE_RE = re.compile(
        r"<\s*(?:code|tt)\s*>(?P<text>.*?)<\s*/\s*(?:code|tt)\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    # Bare `<br>` / `<br/>` are paragraph breaks in HTML; in RST a
    # blank line works.
    _LEAK_HTML_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

    @staticmethod
    def _postprocess_leaked_doxygen_tags(text: str) -> str:
        """Rewrite any doxygen tags that survived the structured parser.

        Defensive cleanup — the doxyparser grammar has gaps (e.g. retval's
        IDENT can't match a `#`-prefixed reference, with_word breaks on
        line-wraps between tag and argument). Without this pass those tags
        render verbatim in the Sphinx output as `@retval ...` text.
        """

        def _retval_sub(m):
            indent = m.group("indent")
            ref = m.group("ref").strip("`")
            desc = m.group("desc").strip()
            return f"{indent}* :py:obj:`{ref}`: {desc}"

        def _word_tag_sub(m):
            tag = m.group("tag")
            word = m.group("word").strip("`")
            if tag in ("a", "e", "em"):
                return f"*{word}*"
            if tag == "b":
                return f"**{word}**"
            return f"`{word}`"

        def _block_tag_sub(m):
            indent = m.group("indent")
            tag = m.group("tag")
            body = m.group("body").strip()
            inner = textwrap.indent(body, indent + "    ") if body else ""
            return f"{indent}.. {tag}::\n\n{inner}"

        def _see_sub(m):
            ref = m.group("ref").strip("`")
            return f":py:obj:`{ref}`"

        text = DoxygenMixin._LEAK_BLOCK_TAG_RE.sub(_block_tag_sub, text)
        text = DoxygenMixin._LEAK_RETVAL_RE.sub(_retval_sub, text)
        text = DoxygenMixin._LEAK_SEE_RE.sub(_see_sub, text)
        text = DoxygenMixin._LEAK_BRIEF_RE.sub(r"\1", text)
        text = DoxygenMixin._LEAK_WORD_TAG_RE.sub(_word_tag_sub, text)
        # HTML tags → RST equivalents.
        text = DoxygenMixin._LEAK_HTML_BOLD_RE.sub(
            lambda m: f"**{m.group('text').strip()}**", text
        )
        text = DoxygenMixin._LEAK_HTML_ITALIC_RE.sub(
            lambda m: f"*{m.group('text').strip()}*", text
        )
        text = DoxygenMixin._LEAK_HTML_CODE_RE.sub(
            lambda m: f"``{m.group('text').strip()}``", text
        )
        text = DoxygenMixin._LEAK_HTML_BR_RE.sub("\n\n", text)
        return text

    # Maximum length of an inferred brief before truncation with " [...]".
    # Long sentences read poorly as a heading; sphinx-autoapi truncates
    # very-long briefs on its own anyway.
    _INFERRED_BRIEF_MAX_LEN = 200

    @staticmethod
    def _infer_brief_from_first_text(text: str) -> str:
        """Return a one-line brief drawn from a free-form details body.

        Strategy:
          1. If the first paragraph is a single non-empty line followed by
             a blank line, return that line verbatim.
          2. Otherwise take the first sentence of the first paragraph
             (split on `". "`, `".\\n"`, or end-of-paragraph).
          3. If the resulting sentence exceeds _INFERRED_BRIEF_MAX_LEN,
             truncate and append ` [...]`.

        Returns an empty string when no usable text is available — the
        caller should then fall through to the missing-text placeholder.
        """
        stripped = text.lstrip(":\n\t ").rstrip()
        if not stripped:
            return ""
        # First paragraph (up to the first blank line).
        first_para, _, _ = stripped.partition("\n\n")
        first_para = first_para.strip()
        if not first_para:
            return ""
        # Single-line paragraph → take it as-is.
        if "\n" not in first_para:
            sentence = first_para
        else:
            # Split on first sentence terminator. Avoid splitting on
            # bare "." inside identifiers / abbreviations by requiring
            # whitespace or end-of-line after the period.
            m = re.search(r"\.(?:\s|$)", first_para)
            if m:
                sentence = first_para[: m.end()].strip()
            else:
                sentence = first_para.replace("\n", " ").strip()
        if len(sentence) > DoxygenMixin._INFERRED_BRIEF_MAX_LEN:
            sentence = (
                sentence[: DoxygenMixin._INFERRED_BRIEF_MAX_LEN].rstrip()
                + " [...]"
            )
        return sentence

    # @ingroup ID — survives raw_comment as a literal `@ingroup foo`
    # marker; the doxyparser flattens it into a `with_single_line_text`
    # token rather than a Section, so we scrape the raw comment with
    # a regex (cheaper than introspecting the section tree).
    _INGROUP_RE = re.compile(r"[@\\]ingroup\s+(\w+)")

    @staticmethod
    def _ingroup_names_from_raw(raw_comment: str) -> "list[str]":
        """Return the list of group IDs this comment registers via
        `@ingroup <ID>` (in source order). Empty list if none."""
        if not raw_comment:
            return []
        return DoxygenMixin._INGROUP_RE.findall(raw_comment)

    # @defgroup <ID> <Display Title>  — the doxyparser puts these in
    # the `other` alternation, not as their own Section, so we scrape
    # raw comments with a regex. Skips @addtogroup intentionally per
    # doxygen spec (addtogroup appends; only defgroup defines title).
    _DEFGROUP_RE = re.compile(
        r"[@\\]defgroup\s+(?P<id>\w+)\s+(?P<title>[^\n]+)",
    )
    # @brief inside a (clean) defgroup comment block.
    _BRIEF_LINE_RE = re.compile(r"[@\\](?:brief|short)\s+(?P<text>[^\n]+)")

    @staticmethod
    def _build_group_index(root):
        """Walk a tree.Root and return ``{group_id: (display_title, brief)}``.

        Built from ``@defgroup <ID> <Display Title>`` markers in any
        node's raw_comment. The brief is the explicit ``@brief`` line
        in the same comment if present, else the first sentence of
        the comment body via ``_infer_brief_from_first_text``.
        Memoized on root.

        Per doxygen spec, ``@addtogroup <ID>`` does NOT define a new
        group's display title — it appends to a group defined
        elsewhere — so the index is built solely from ``@defgroup``.
        """
        cached = getattr(root, "_doxygen_group_index", None)
        if cached is not None:
            return cached
        index = {}
        seen_raw = set()
        for node in root.walk(postorder=False):
            raw = getattr(node, "raw_comment", None)
            if not raw or raw in seen_raw:
                continue
            seen_raw.add(raw)
            try:
                cleaner = (
                    node.raw_comment_cleaner
                    if hasattr(node, "raw_comment_cleaner")
                    else (lambda s: s)
                )
                cleaned = doxyparser.remove_doxygen_comment_chars(cleaner(raw))
            except Exception:
                cleaned = raw
            for m in DoxygenMixin._DEFGROUP_RE.finditer(cleaned):
                group_id = m.group("id")
                title = m.group("title").strip() or group_id
                # Compute the brief from the same comment block.
                # Look for @brief first; otherwise take the first
                # sentence after the @defgroup line.
                brief = ""
                brief_m = DoxygenMixin._BRIEF_LINE_RE.search(cleaned)
                if brief_m:
                    brief = brief_m.group("text").strip()
                else:
                    # Body after the defgroup line — slice from the
                    # match end onward, then run the Part 11 helper.
                    body = cleaned[m.end():].lstrip("\n\r\t ")
                    brief = DoxygenMixin._infer_brief_from_first_text(body)
                # Don't overwrite with an empty brief if a non-empty
                # one was already registered elsewhere.
                existing = index.get(group_id)
                if existing is None or (existing[1] == "" and brief):
                    index[group_id] = (title, brief or "")
        setattr(root, "_doxygen_group_index", index)
        return index

    @staticmethod
    def _render_doxygen_brief(
        sections,
        log_prefix: str = "",
        missing_text: str = "(No short description)",
        host_node=None,
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
        # No explicit `\brief` — try to infer one from the first details
        # section's first text block. Many ROCm headers skip `@brief`
        # and just write a free-form description; promoting the first
        # sentence keeps Sphinx's autosummary from showing the bare
        # placeholder.
        for section in sections:
            for block in section.blocks:
                if not isinstance(block, doxyparser.TextBlock):
                    continue
                inferred = DoxygenMixin._infer_brief_from_first_text(
                    block.transformed_text
                )
                if inferred:
                    return inferred + "\n\n"
                break  # first text block per section only
        # Part 12a: explicit @ingroup → @defgroup display title
        # fallback. The symbol's raw comment carries one or more
        # `@ingroup <ID>`; if any ID resolves to a non-empty group
        # brief in the TU-wide index, render it with attribution.
        if host_node is not None and hasattr(host_node, "get_root"):
            try:
                raw = getattr(host_node, "raw_comment", None) or ""
                group_ids = DoxygenMixin._ingroup_names_from_raw(raw)
                if group_ids:
                    idx = DoxygenMixin._build_group_index(host_node.get_root())
                    for gid in group_ids:
                        entry = idx.get(gid)
                        if not entry:
                            _log.debug(
                                "%sdoxygen: @ingroup %s references "
                                "undefined group; skipping",
                                log_prefix,
                                gid,
                            )
                            continue
                        display_title, brief = entry
                        if brief:
                            return f"[group: {display_title}] {brief}\n\n"
            except Exception:
                # Group-fallback is best-effort; never crash on it.
                pass
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
        raw = self.raw_comment
        # libclang sometimes attaches a bare `//! @}` close marker to
        # the first decl after a closed group block. Treat that as no
        # real comment so inheritance can still fire.
        if raw is not None and _raw_comment_is_only_group_bracket(raw):
            raw = None
        if raw is None:
            # Lexical-@{ … @} fallback: inherit the opener comment if
            # this declaration sits inside such a block. Per Part 12b
            # of the docs-polish plan — the dominant missing-brief
            # source (hipblas alone has 989 placeholders).
            idx = _build_inherited_comment_index(self.get_root())
            raw = idx.get(self.cursor.hash)
        if raw is not None:
            # Strip group bracket markers — they're pure structural
            # doxygen syntax, never carry meaning. Apply to ALL raw
            # comments, not just inherited ones, because libclang
            # attaches the `/*! @{ … */` opener directly to the FIRST
            # declaration in a block.
            raw = _strip_group_brackets(raw)
            cleaned_raw_comment = self.raw_comment_cleaner(raw)
            return doxyparser.remove_doxygen_comment_chars(cleaned_raw_comment)
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
            sections,
            log_prefix=f"<{self.render_location()}> ",
            host_node=self if isinstance(self, tree.Node) else None,
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
        # Mop up any doxygen tags the structured parser left behind.
        docstring_body = self._postprocess_leaked_doxygen_tags(docstring_body)
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

