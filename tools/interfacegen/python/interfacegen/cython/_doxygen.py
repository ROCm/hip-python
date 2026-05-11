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


"""Cython backend — doxygen submodule.

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
from . import _defaults
from ._defaults import *  # noqa: F401,F403

__all__ = [
    'DOXYGEN_CONV',
    '_INHERITABLE_CURSOR_KINDS',
    '_comment_opens_group',
    '_comment_closes_group',
    '_raw_comment_is_only_group_bracket',
    '_strip_group_brackets',
    '_build_inherited_comment_index',
]

# doxygen parser
DOXYGEN_CONV = doxyparser.DoxygenGrammar()
DOXYGEN_CONV.escaped.set_parse_action(doxyparser.format.PythonDocstrings.escaped)
DOXYGEN_CONV.with_word.set_parse_action(
    doxyparser.format.PythonDocstrings.with_word
)
DOXYGEN_CONV.fdollar.set_parse_action(doxyparser.format.PythonDocstrings.fdollar)
DOXYGEN_CONV.frnd.set_parse_action(doxyparser.format.PythonDocstrings.frnd)


def reference_(tokens):
    global python_interface_pyobj_role_template
    reference: str = tokens[0].replace("#", ".")
    reference = reference.replace("::", ".")
    return python_interface_pyobj_role_template.format(
        name=reference.lstrip(".")
    )


DOXYGEN_CONV.see_reference.set_parse_action(reference_)
DOXYGEN_CONV.in_text_reference.set_parse_action(reference_)


def other_parse_action(tokens):
    cmd = tokens[0][1:]
    if cmd == "ref":
        return f"``{tokens[1]}`` "
    return []  # suppress all others


DOXYGEN_CONV.other.set_parse_action(other_parse_action)


# ---------------------------------------------------------------------------
# Lexical `@{ … @}` group-comment inheritance (docs-polish Part 12b)
# ---------------------------------------------------------------------------
#
# Doxygen lets header authors attach one comment to many declarations via
# `@{ … @}` brackets — anonymous (`/*! @{ \brief Foo */ ... //! @}`) or
# named (`/** \addtogroup foo @{ */ ... /** @} */`). libclang only
# attaches the opener comment to the FIRST enclosed declaration; every
# subsequent declaration's `cursor.raw_comment` is None and the renderer
# falls through to the "(No short description)" placeholder.
#
# Walk the TU's token stream with a comment-stack state machine, keep
# track of the active opener at every source line, and map declarations
# whose own raw_comment is empty to that opener. Functions only —
# typedefs/enums/structs get docs from other code paths and the
# inherited brief usually doesn't apply to them.
#
# Memoized per-Root so the walk runs once per codegen invocation.

_INHERITABLE_CURSOR_KINDS = frozenset({
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
})


def _comment_opens_group(spelling: str) -> bool:
    """True if a comment token's body contains an `@{` or `\\{`
    group-open marker."""
    return "@{" in spelling or r"\{" in spelling


def _comment_closes_group(spelling: str) -> bool:
    """True if a comment token's body contains an `@}` or `\\}`
    group-close marker."""
    return "@}" in spelling or r"\}" in spelling


# Comments whose stripped body matches one of these are bracket markers,
# not real documentation. libclang sometimes attaches a bare `//! @}`
# close to the next decl after a closed group block; treat that as "no
# real raw_comment" so inheritance can still fire.
_GROUP_BRACKET_RE = re.compile(
    r"^\s*(?:/\*[!*]?|//[!/]?)?\s*[@\\][{}]\s*(?:\*+/)?\s*$"
)


def _raw_comment_is_only_group_bracket(raw_comment: str) -> bool:
    """True if `raw_comment` is a bare `@{` / `@}` bracket marker
    with no documentation content."""
    if not raw_comment:
        return False
    return bool(_GROUP_BRACKET_RE.match(raw_comment.strip()))


def _strip_group_brackets(text: str) -> str:
    """Remove `@{` / `@}` (and `\\{` / `\\}`) group bracket markers
    from `text`. They're pure structural doxygen syntax — never
    documentation — so leaving them in any rendered docstring
    produces noise."""
    if not text:
        return text
    text = re.sub(r"[@\\]\{\s*", "", text)
    text = re.sub(r"\s*[@\\]\}", "", text)
    return text


def _build_inherited_comment_index(root):
    """Return ``{cursor.hash: inherited_comment_text}`` for function
    declarations that sit inside a `@{ … @}` block but lack their own
    `cursor.raw_comment`.

    Memoized on ``root`` via ``_inherited_comment_index`` attribute.
    Subsequent calls return the cached dict.

    Algorithm: walk every token in every input file via libclang's
    tokenizer. Track an active-opener stack. On `@{` push the opener
    comment; on `@}` pop. For identifier tokens whose enclosing
    cursor is a function declaration AND whose spelling matches the
    cursor's name (the "duplicate cursor trap" — return-type / param
    / brace tokens all map to the same cursor), inherit the
    stack-top comment if the cursor has no raw_comment of its own.

    Malformed input is logged but does not abort: stray `@}` is
    ignored; unclosed `@{` lets all subsequent decls inherit through
    EOF.
    """
    cached = getattr(root, "_inherited_comment_index", None)
    if cached is not None:
        return cached
    index = {}
    cursor = root.cursor
    tu = cursor.translation_unit
    if tu is None:
        setattr(root, "_inherited_comment_index", index)
        return index

    TokenKind = clang.cindex.TokenKind
    try:
        tokens = list(tu.cursor.get_tokens())
    except Exception:
        setattr(root, "_inherited_comment_index", index)
        return index

    # Track the active group-opener stack PER FILE — `@{` in one
    # header should not silently extend across an `#include` boundary
    # into another header. We segment by `tok.location.file.name`.
    per_file_stack = {}    # filename -> list of opener comment strings
    current_filename = None
    for tok in tokens:
        loc = tok.location
        loc_file = getattr(loc, "file", None)
        filename = loc_file.name if loc_file is not None else None
        # Don't lose state when an enclosing TU re-enters the same file
        # after #include returns; key by filename.
        if filename is None:
            continue
        active_stack = per_file_stack.setdefault(filename, [])
        current_filename = filename

        if tok.kind == TokenKind.COMMENT:
            spelling = tok.spelling
            opens = _comment_opens_group(spelling)
            closes = _comment_closes_group(spelling)
            if opens and closes:
                # Self-closing block — push and immediately pop is a
                # net no-op for inheritance.
                continue
            if opens:
                active_stack.append(spelling)
            elif closes:
                if active_stack:
                    active_stack.pop()
                else:
                    _log.warning(
                        "<%s:%s> doxygen: stray @} with no open "
                        "group; ignoring",
                        filename,
                        getattr(loc, "line", "?"),
                    )
            continue
        if not active_stack:
            continue
        if tok.kind != TokenKind.IDENTIFIER:
            continue
        tok_cursor = clang.cindex.Cursor.from_location(tu, loc)
        if tok_cursor is None or tok_cursor.kind not in _INHERITABLE_CURSOR_KINDS:
            continue
        # Duplicate-cursor trap: only the cursor's name token matches
        # `cursor.spelling`. Return-type and parameter tokens all map
        # to the same cursor but spell differently.
        if tok.spelling != tok_cursor.spelling:
            continue
        if tok_cursor.raw_comment and not _raw_comment_is_only_group_bracket(
            tok_cursor.raw_comment
        ):
            continue
        try:
            key = tok_cursor.hash
        except Exception:
            continue
        # First occurrence wins (the function's name token comes
        # before its parameter list in source order).
        index.setdefault(key, active_stack[-1])

    for filename, stack in per_file_stack.items():
        if stack:
            _log.warning(
                "<%s> doxygen: %d unclosed @{ block(s) at EOF — "
                "subsequent declarations inherit through EOF",
                filename,
                len(stack),
            )

    setattr(root, "_inherited_comment_index", index)
    return index
