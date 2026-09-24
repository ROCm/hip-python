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

from .. import cparser, cythontemplates, doxyparser, tree
from ..support import cython as support
from ..support.recipes import control

_log = logging.getLogger("interfacegen")
from . import _defaults, _doxygen
from ._defaults import *  # noqa: F401,F403
from ._doxygen import *  # noqa: F401,F403

__all__ = [
    "DoxygenMixin",
    "CythonMixin",
    "Node",
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
    _LEAK_SEE_RE = re.compile(r"@see\s+(?P<ref>`[^`\n]+`|[A-Za-z_][\w:#]*)")
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

        def _norm_ref(ref):
            # Normalise a raw doxygen ref the way the structured path does
            # (see ``_doxygen.reference_``): strip backticks, rewrite
            # ``#``/``::`` separators to ``.``, drop the leading dot, then
            # wrap via the shared role template so leaked refs fuzzy-resolve
            # with a short display name (``~.NAME``) instead of dangling as a
            # bare (non-fuzzy) role.
            ref = ref.strip("`").replace("#", ".").replace("::", ".")
            # Drop any leading role modifiers so a ref that was already
            # rewritten to ``~.NAME`` / ``.NAME`` by an earlier pass is not
            # double-prefixed when re-wrapped through the template.
            return python_interface_pyobj_role_template.format(
                name=ref.lstrip("~.")
            )

        def _retval_sub(m):
            indent = m.group("indent")
            desc = m.group("desc").strip()
            return f"{indent}* {_norm_ref(m.group('ref'))}: {desc}"

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
            return _norm_ref(m.group("ref"))

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
    # `\ingroup` takes a space-separated list of group ids on a single
    # line per the doxygen spec — `\ingroup foo bar baz` adds the
    # current entity to all three groups. Capture the rest of the line
    # and let the caller split on identifier tokens to drop trailing
    # comment delimiters (`*/`) and whitespace.
    _INGROUP_RE = re.compile(r"[@\\]ingroup\s+([^\n]+)")
    _GROUP_ID_RE = re.compile(r"\w+")

    @staticmethod
    def _ingroup_names_from_raw(raw_comment: str) -> "list[str]":
        """Return the list of group IDs this comment registers via
        ``@ingroup <ID> [<ID>...]`` lines (in source order). Empty
        list if none. Multi-id syntax on one line is supported."""
        if not raw_comment:
            return []
        out = []
        for line in DoxygenMixin._INGROUP_RE.findall(raw_comment):
            out.extend(DoxygenMixin._GROUP_ID_RE.findall(line))
        return out

    # @defgroup <ID> <Display Title>  — the doxyparser puts these in
    # the `other` alternation, not as their own Section, so we scrape
    # raw comments with a regex.
    _DEFGROUP_RE = re.compile(
        r"[@\\]defgroup\s+(?P<id>\w+)\s+(?P<title>[^\n]+)",
    )
    # @addtogroup <ID> [<Display Title>]  — primary purpose is to
    # append docs to an existing group, but per doxygen's group
    # priority hierarchy `\addtogroup` is also a *fallback* title
    # source: if no `\defgroup` defines the id, `\addtogroup id title`
    # provides the title. ROCm headers exploit this in roughly 70
    # cases (e.g. `\addtogroup prefixsums Prefix Sums`).  The title
    # group is non-greedy so we don't swallow the trailing `\n` or
    # the `@{` opener that often follows.
    _ADDTOGROUP_TITLED_RE = re.compile(
        r"[@\\]addtogroup\s+(?P<id>\w+)\s+(?P<title>[^\n@\\{]+?)\s*(?:[@\\][{}]|\n|\*/|$)",
    )
    # @brief inside a (clean) defgroup comment block.
    _BRIEF_LINE_RE = re.compile(r"[@\\](?:brief|short)\s+(?P<text>[^\n]+)")

    @staticmethod
    def _build_group_index(root):
        """Walk a tree.Root and return ``{group_id: (display_title, brief)}``.

        Two-pass build per doxygen's group priority hierarchy:

        1. ``\\defgroup <ID> <Display Title>`` — primary source.
           Defines the group's identity (id, title, optional brief).
        2. ``\\addtogroup <ID> <Display Title>`` — fallback source.
           Per the doxygen manual, `\\addtogroup` is primarily for
           appending docs to an existing group, but if the group has
           no `\\defgroup`, the optional title argument provides the
           title. Only filled in when no `\\defgroup` already exists
           for the id.

        ``\\weakgroup`` is the third tier (lowest priority) but ROCm
        headers do not use it, so it's not implemented.

        The brief for either source is the explicit ``@brief`` line
        in the same comment if present, else the first sentence of
        the comment body via ``_infer_brief_from_first_text``.

        Memoized on root.
        """
        cached = getattr(root, "_doxygen_group_index", None)
        if cached is not None:
            return cached
        index = {}
        # Cache cleaned-comment text per raw_comment so we don't run
        # the cleaner twice across the two passes.
        cleaned_by_raw = {}
        for node in root.walk(postorder=False):
            raw = getattr(node, "raw_comment", None)
            if not raw or raw in cleaned_by_raw:
                continue
            try:
                cleaner = (
                    node.raw_comment_cleaner
                    if hasattr(node, "raw_comment_cleaner")
                    else (lambda s: s)
                )
                cleaned_by_raw[raw] = doxyparser.remove_doxygen_comment_chars(
                    cleaner(raw)
                )
            except Exception:
                cleaned_by_raw[raw] = raw

        def _record(group_id, title, cleaned, body_start):
            """Add or refine a group entry."""
            brief = ""
            brief_m = DoxygenMixin._BRIEF_LINE_RE.search(cleaned)
            if brief_m:
                brief = brief_m.group("text").strip()
            else:
                body = cleaned[body_start:].lstrip("\n\r\t ")
                brief = DoxygenMixin._infer_brief_from_first_text(body)
            existing = index.get(group_id)
            if existing is None or (existing[1] == "" and brief):
                index[group_id] = (title, brief or "")

        # Pass 1: \defgroup wins.
        for cleaned in cleaned_by_raw.values():
            for m in DoxygenMixin._DEFGROUP_RE.finditer(cleaned):
                group_id = m.group("id")
                title = m.group("title").strip() or group_id
                _record(group_id, title, cleaned, m.end())
        # Pass 2: \addtogroup id title fills in only when \defgroup
        # didn't already register the id. (Same id may appear in
        # multiple addtogroup comments — first one wins.)
        for cleaned in cleaned_by_raw.values():
            for m in DoxygenMixin._ADDTOGROUP_TITLED_RE.finditer(cleaned):
                group_id = m.group("id")
                if group_id in index:
                    continue
                title = m.group("title").strip()
                if not title:
                    continue
                _record(group_id, title, cleaned, m.end())
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
                    # Annotate the block so `_render_doxygen_section_body`
                    # can elide the same text from the details body and
                    # avoid duplicating the promoted brief.
                    block._promoted_to_brief = inferred
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
    def _render_doxygen_section_body(
        section, outer_indent, transform_references=True
    ) -> str:
        """Renders the body of a doxygen section.

        ``transform_references=False`` skips the inline reference rewrite so a
        caller (``see``/``sa``) can run its own single reference pass without
        double-wrapping ``::``-qualified names.
        """
        result = ""
        for block in section.blocks:
            if isinstance(block, doxyparser.TextBlock):
                # variants we've seen
                # \note: texttext => firstline == ": texttext"
                # \note texttext
                # \note texttext
                #    texttext
                text = (
                    block.get_text(
                        transform_formatting=True,
                        transform_other=True,
                        transform_references=transform_references,
                    )
                    .lstrip(":\n\t ")
                    .rstrip()
                )
                # Part 11 elision — when this block's leading text was
                # promoted as the inferred brief (no explicit `\brief`),
                # strip it here to avoid duplicating the same paragraph
                # in both the brief slot and the details body.
                promoted = getattr(block, "_promoted_to_brief", None)
                if promoted and text.startswith(promoted):
                    text = text[len(promoted) :].lstrip(":\n\t ")
                lines = text.splitlines()
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
                    result += f"{inner_indent}:nowrap:\n"
                    result += f"\n{inner_indent}\\begin{{{block.env}}}\n"
                result += "\n"
                code = textwrap.dedent(block.code)
                result += textwrap.indent(code, inner_indent).rstrip() + "\n"
                if block.env is not None:
                    result += f"{inner_indent}\\end{{{block.env}}}\n"
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
        if section.kind in ("see", "sa"):
            # Render without the inline reference pass, then convert the
            # bare/``::``-qualified name exactly once. Running both passes
            # double-wraps ``::``-qualified names (the literal ``py``/``obj``
            # from the first ``:py:obj:`` output get re-wrapped by the second).
            body = DoxygenMixin._render_doxygen_section_body(
                section, outer_indent, transform_references=False
            )
            docstring_addition += (
                self.doxygen_conv.see_reference.transform_string(body)
            )
        else:
            docstring_addition += DoxygenMixin._render_doxygen_section_body(
                section, outer_indent
            )
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

    # Cython type spellings that should NOT be prefixed with the cy*
    # module name (``cyhip.``, ``cyhiprtc.``, …). These are primitive
    # C / stdint types that exist directly in the Cython compilation
    # context — the cy* module doesn't redefine them, and prepending
    # the prefix would produce broken declarations like
    # ``cyhip.char *``.
    _PRIMITIVE_C_TYPES_FOR_CPREFIX = frozenset(
        {
            "void",
            "char",
            "short",
            "int",
            "long",
            "float",
            "double",
            "signed",
            "unsigned",
            "_Bool",
            "bint",
            "size_t",
            "ssize_t",
            "ptrdiff_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "intptr_t",
            "uintptr_t",
        }
    )

    @staticmethod
    def _add_module_cprefix(c_type: str, cprefix: str) -> str:
        """Prepend ``cprefix`` (the cy* module qualifier) to a Cython
        type spelling, preserving leading ``const`` / ``volatile``
        qualifiers as outermost — but skip the prefix entirely when
        the type's leading identifier is a primitive C type (``int``,
        ``void``, ``char``, ``unsigned``, ``size_t``, ``uint32_t``,
        …). Primitives exist directly in the Cython compilation
        context and are not redefined in the cy* module.

        Used by the with-nogil emitter and the four Python-touching
        ``_analyze_parms`` handlers when the bare cy*-spelling
        (``hipMemcpyKind``, ``dim3``, ``hipStream_t``) collides with
        a same-named Python wrapper / IntEnum class in the high-level
        ``<module>.pyx``, and by the pointer-field property renderer
        (``Field.render_python_property``) for the setter cast target.
        """
        # Strip leading qualifiers (preserving them on the outside).
        qualifiers = ""
        rest = c_type
        for qual in ("const ", "volatile "):
            if rest.startswith(qual):
                qualifiers = qual
                rest = rest[len(qual) :]
                break
        # Inspect the leading identifier (everything before the first
        # space, ``*``, or ``(``).
        head = rest
        for sep in (" ", "*", "("):
            idx = head.find(sep)
            if idx >= 0:
                head = head[:idx]
        if head in CythonMixin._PRIMITIVE_C_TYPES_FOR_CPREFIX:
            return c_type
        return f"{qualifiers}{cprefix}{rest}"

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
            cleaned = doxyparser.remove_doxygen_comment_chars(
                cleaned_raw_comment
            )
            # Resolve transitively any \copydoc / @copydoc directives.
            # Runs LAST (on already-delimiter-stripped text) so the
            # target index — also stored as cleaned text — drops
            # plain doxygen content into plain doxygen content with
            # no nested-comment artifacts. doxygen itself resolves
            # \copydoc at XML-generation time; libclang does not.
            return _resolve_copydoc_in_text(cleaned, self.get_root())
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
        self,
        cprefix: str,
        *,
        override_name: str = None,
        base: str = None,
        module_opts: dict = None,
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

    def _pyi_pointer_base(self, base: str = None):
        """The base a wrapper class stub inherits from.

        Every wrapper the .pyx emits derives from the util package's
        `Pointer` (see `wrapper_class_impl_base_template`), which is
        where `createRef` and the other handle methods live. A stub that
        omits the base drops all of them.
        """
        if base:
            return base
        prefix = getattr(self, "util_types_prefix", "")
        return f"{prefix}Pointer" if prefix else None

    def _render_pyi_class_stub(
        self,
        cprefix: str,
        override_name: str = None,
        base: str = None,
        *,
        body: list = None,
        include_init: bool = True,
    ):
        """Shared implementation used by `Record`, `Enum`, `AnonymousEnum`,
        and `FunctionPointer`. Renders ``class Name[(base)]:`` followed
        by the docstring (`render_python_docstring`), a placeholder
        ``__init__`` and the caller's `body` lines, which arrive
        indented.

        `include_init` is off for enums: a type checker models enum
        construction from the base class, and a ``(*args, **kwargs)``
        placeholder only overrides that with something less precise."""
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
        if include_init:
            lines.append("    def __init__(self, *args, **kwargs): ...")
        lines.extend(body or [])
        return lines


Node = CythonMixin  # alias so that it can be used in treefactory
