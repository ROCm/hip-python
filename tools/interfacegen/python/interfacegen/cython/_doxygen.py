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
    '_COPYDOC_RE',
    '_build_copydoc_target_index',
    '_resolve_copydoc_in_text',
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

    # Collect every distinct file that contains an inheritable cursor.
    # `tu.cursor.get_tokens()` only yields tokens from the main TU
    # file; included headers (where the actual function decls and
    # their @{ markers live) are NOT covered by that iterator. We
    # enumerate the cursor tree and tokenize each file's full extent
    # separately. The per-file end offset is taken from the
    # max-extent of any cursor in that file — libclang rejects
    # `SourceLocation.from_offset` past the file's actual size with
    # silent zero-token returns, so an over-shoot here is fatal.
    files_to_walk = {}    # filename -> (cxfile, max_extent_end_offset)
    for child in cursor.walk_preorder() if hasattr(cursor, "walk_preorder") else []:
        loc_file = getattr(child.location, "file", None)
        if loc_file is None:
            continue
        try:
            end_offset = child.extent.end.offset
        except Exception:
            try:
                end_offset = child.location.offset
            except Exception:
                end_offset = 0
        prev = files_to_walk.get(loc_file.name)
        if prev is None or end_offset > prev[1]:
            files_to_walk[loc_file.name] = (loc_file, end_offset)

    # Aggregate tokens from every file we care about.
    tokens = []
    for fname, (cxfile, max_end) in files_to_walk.items():
        # Try the disk-backed size first (covers includes); fall
        # back to the cursor-derived end-offset for unsaved test
        # files. We never over-shoot because libclang silently
        # returns zero tokens for out-of-bounds extents.
        try:
            with open(fname, "rb") as fh:
                size = len(fh.read())
        except OSError:
            size = max_end
        try:
            start = clang.cindex.SourceLocation.from_offset(tu, cxfile, 0)
            end = clang.cindex.SourceLocation.from_offset(tu, cxfile, size)
            extent = clang.cindex.SourceRange.from_locations(start, end)
            tokens.extend(list(tu.get_tokens(extent=extent)))
        except Exception:
            continue

    # Track the active group-opener stack PER FILE — `@{` in one
    # header should not silently extend across an `#include` boundary
    # into another header. We segment by `tok.location.file.name`.
    per_file_stack = {}      # filename -> list of opener comment strings
    # Track the most-recent non-marker doc comment seen in each file.
    # Used to fold a "doc BEFORE @{" pair (Pattern A) into a single
    # opener content when the @{ comment is itself a bare marker —
    # mirrors libclang's "merge adjacent comments" behavior. Key:
    # filename → (last_comment_spelling, last_comment_token_index)
    # so we can decide whether the prior comment is "close enough" to
    # belong with the opener.
    per_file_last_doc = {}
    current_filename = None
    for tok_index, tok in enumerate(tokens):
        loc = tok.location
        loc_file = getattr(loc, "file", None)
        filename = loc_file.name if loc_file is not None else None
        # Don't lose state when an enclosing TU re-enters the same file
        # after #include returns; key by filename.
        if filename is None:
            continue
        active_stack = per_file_stack.setdefault(filename, [])
        current_filename = filename

        # Pattern-C invalidation: a `;` / `{` / `}` punctuation outside
        # any active group block is a decl/block boundary — any pending
        # doc comment that wasn't claimed by a function decl is now
        # orphaned and must not "leak" forward to the next decl. This
        # is what keeps `int foo(); int bar();` from making `bar`
        # inherit `foo`'s upstream doc comment by accident. Inside an
        # active stack, Pattern A/B owns the bookkeeping and we leave
        # the marker alone.
        if (
            tok.kind == TokenKind.PUNCTUATION
            and not active_stack
            and tok.spelling in (";", "{", "}")
        ):
            per_file_last_doc.pop(filename, None)

        if tok.kind == TokenKind.COMMENT:
            spelling = tok.spelling
            opens = _comment_opens_group(spelling)
            closes = _comment_closes_group(spelling)
            if opens and closes:
                # Self-closing block — push and immediately pop is a
                # net no-op for inheritance.
                per_file_last_doc.pop(filename, None)
                continue
            if opens:
                # Pattern-A fold: if the @{ opener is a bare marker
                # comment (e.g. `/**@{*/` or `///@{` — just the
                # bracket plus surrounding comment delimiters, no
                # description) AND the previous token in this file
                # was a non-marker doc comment (i.e., they're an
                # adjacent doc-then-opener pair), use the prior doc
                # comment's content as the opener content. This
                # mirrors libclang's merge-adjacent-comments
                # behavior at the walker level.
                is_bare_marker = _raw_comment_is_only_group_bracket(spelling)
                prior = per_file_last_doc.get(filename)
                if (
                    is_bare_marker
                    and prior is not None
                    and prior[1] == tok_index - 1
                ):
                    active_stack.append(prior[0])
                else:
                    active_stack.append(spelling)
                per_file_last_doc.pop(filename, None)
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
                per_file_last_doc.pop(filename, None)
            elif active_stack:
                # Pattern-B (doc INSIDE the @{ … @} block). The active
                # opener may be a bare marker like `///@{` or `/**@{*/`
                # carrying no documentation; the real description sits
                # in a separate doc comment placed between the opener
                # and the first decl. Per doxygen's Member Groups
                # feature (with DISTRIBUTE_GROUP_DOC=YES, or implicitly
                # for class members), that doc comment is shared by
                # every member of the lexical block — see the
                # "Member Groups" section of the doxygen manual for the
                # canonical example. Replace the active stack top so
                # subsequent un-attached decls inherit the right text.
                #
                # Most-recent-doc-wins if multiple inside-block doc
                # comments appear before any decl, matching doxygen's
                # "comment immediately preceding declaration" rule.
                active_stack[-1] = spelling
                per_file_last_doc[filename] = (spelling, tok_index)
            else:
                # Bare doc comment outside any active block — remember
                # it in case the immediately-following token opens an
                # @{ group with a bare marker (Pattern A fold above).
                per_file_last_doc[filename] = (spelling, tok_index)
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
        if active_stack:
            # First occurrence wins (the function's name token comes
            # before its parameter list in source order).
            index.setdefault(key, active_stack[-1])
        else:
            # Pattern C — function decl outside any @{ block, with no
            # libclang-attached raw_comment. The most common cause is a
            # `#if(...)` / `#elif(...)` preprocessor guard sitting
            # between the doc comment and the decl: libclang treats the
            # directive as breaking the "comment immediately preceding
            # declaration" association even though doxygen itself
            # happily walks past it. Hipsparse's generic API
            # (`hipsparseCreateSpVec` etc.) is the canonical example.
            #
            # The pending_doc tracker holds the most recent unclaimed
            # doc comment in this file; the boundary-clear above
            # ensures it's invalidated whenever a `;` / `{` / `}`
            # intervenes, so we don't accidentally inherit a comment
            # from across a different decl.
            recovered = per_file_last_doc.get(filename)
            if recovered is not None:
                index.setdefault(key, recovered[0])
                per_file_last_doc.pop(filename, None)

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


# Doxygen accepts both `\copydoc <name>` and `@copydoc <name>`. The
# link-object grammar is `<word>` plus optional `()` (member function
# specifier — meaningless for our C-only inputs but accepted silently).
# Capture the bare identifier; qualified references (`Class::method`,
# overload-disambiguating `nm(args)`) are left as literals — same
# graceful degradation as the unknown-reference case.
_COPYDOC_RE = re.compile(r"[@\\]copydoc\s+(\w+)\s*(?:\(\))?")


def _build_copydoc_target_index(root):
    """Return ``{symbol_name: cleaned_text}`` for every FUNCTION_DECL
    in the TU that has a non-empty libclang-attached
    ``cursor.raw_comment``. The stored text is passed through
    ``_strip_group_brackets`` and
    ``doxyparser.remove_doxygen_comment_chars`` BEFORE caching — so
    substitution into another (already-cleaned) comment never
    produces nested ``/* ... */`` delimiters.

    Memoized on ``root`` via the ``_copydoc_target_index`` attribute.

    Cached values are NOT recursively resolved — ``\\copydoc``
    directives inside a target body are resolved lazily by
    ``_resolve_copydoc_in_text``. Per-call resolution keeps cycle
    detection granular instead of baked into the cache.
    """
    cached = getattr(root, "_copydoc_target_index", None)
    if cached is not None:
        return cached
    index = {}
    cursor = root.cursor
    tu = cursor.translation_unit
    if tu is not None:
        for c in cursor.walk_preorder() if hasattr(cursor, "walk_preorder") else []:
            if c.kind != clang.cindex.CursorKind.FUNCTION_DECL:
                continue
            if not c.spelling or not c.raw_comment:
                continue
            stripped = _strip_group_brackets(c.raw_comment)
            cleaned = doxyparser.remove_doxygen_comment_chars(stripped)
            index.setdefault(c.spelling, cleaned)
    setattr(root, "_copydoc_target_index", index)
    return index


def _resolve_copydoc_in_text(text, root, _seen=None):
    """Substitute every ``\\copydoc <ref>`` directive in ``text``
    with the referenced symbol's cleaned raw_comment. The input
    ``text`` must ALREADY have been passed through
    ``doxyparser.remove_doxygen_comment_chars`` — substitution drops
    plain doxygen content (no ``/* */`` delimiters, no per-line
    ``*`` markers) into plain doxygen content, so no nested-comment
    artifacts arise. Transitive: if a substituted body itself
    contains further ``\\copydoc`` directives those are resolved in
    turn. Cycle-safe via the ``_seen`` accumulator.

    Substitution semantics:

    * The ``\\copydoc <ref>`` directive is replaced with the
      target's cleaned raw_comment text verbatim.
    * Surrounding tags in the same comment (``\\ingroup``,
      ``\\deprecated``, ``\\param``, …) are preserved.
    * Reference to an unknown symbol: leave the directive in place
      as a literal (matches doxygen's behavior — emits "no matching
      definition" but doesn't crash).
    * Reference forming a cycle: stop at the cycle point, leave the
      innermost directive as a literal.

    Called from ``DoxygenMixin._raw_comment_cleaned`` as the final
    step, after ``remove_doxygen_comment_chars`` has stripped the
    source comment's own delimiters. doxygen itself resolves
    ``\\copydoc`` at XML-generation time; libclang does not, so we
    do it here.
    """
    if not text or _COPYDOC_RE.search(text) is None:
        return text  # fast path: no directives
    if _seen is None:
        _seen = set()
    targets = _build_copydoc_target_index(root)

    def _sub(match):
        name = match.group(1)
        if name in _seen:
            return match.group(0)  # cycle — leave verbatim
        target_cleaned = targets.get(name)
        if target_cleaned is None:
            return match.group(0)  # unknown — leave verbatim
        return _resolve_copydoc_in_text(
            target_cleaned, root, _seen | {name},
        )

    return _COPYDOC_RE.sub(_sub, text)
