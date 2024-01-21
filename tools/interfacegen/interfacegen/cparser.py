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

__author__ = "Advanced Micro Devices, Inc."

import clang.cindex
from .typehandler import TypeHandler

def walk_cursors(root: clang.cindex.Cursor, postorder=False):
    """Yields a triple of cursor, level, parents per traversed cursor.

    Yields a triple per cursor that consists of the cursor, its level
    and a stack of parent cursors (in that order).

    Args:
        root (clang.cindex.Cursor): The cursor to do the walk on.
        postorder (bool, optional): Post-order walk. Defaults to False.

    Note:
        Defaults to pre-order walk, i.e. children are yielded after their parent.
    Note:
        The first cursor is the cursor for the translation unit, it has level 0.
    """

    def descend_(cursor, level=0, parent_stack=[]):
        if postorder:
            for child in cursor.get_children():
                yield from descend_(child, level + 1, parent_stack + [cursor])
        yield (cursor, level, parent_stack)  # yield current
        if not postorder:
            for child in cursor.get_children():
                yield from descend_(child, level + 1, parent_stack + [cursor])

    yield from descend_(root)


class CParser:
    """Parser for C APIs."""

    def __init__(
        self, filename: str, append_cflags: list = [], unsaved_files=None
    ):
        """Parse the specified file.

        Args:
            filename (str): Path of the file to parse.
            append_cflags (list[str], optional): Additional flags to append when parsing.
            unsaved_files (optional): List of strings representing source file contents.
        """
        self.filename = filename
        self.append_cflags = append_cflags
        self.translation_unit = None
        self.unsaved_files = unsaved_files

    @property
    def cursor(self):
        assert self.translation_unit != None
        return self.translation_unit.cursor

    def parse(self):
        """Parse the specified file."""
        # print(self._append_cflags)
        self.translation_unit = clang.cindex.TranslationUnit.from_source(
            self.filename,
            args=["-x", "c"] + self.append_cflags,
            options=(
                clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
                | clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD  # keeps the macro defs as "fake" nodes without location
            ),
            unsaved_files=self.unsaved_files,
        )
        return self

    def walk_cursors(self, cursor=None, postorder=False):
        """Yields a tuple per cursor that consists of the cursor's level and the cursor.

        Yields a triple per cursor that consists of the cursor, its level
        and a stack of parent cursors (in that order).

        Args:
            cursor (bool, optional): The cursor to do the walk on, or None if the cparser's root cursor
                                     should be used. Defaults to None, i.e. usage of the cparser's root cursor.
            postorder (bool, optional): Post-order walk. Defaults to False.

        Note:
            Defaults to pre-order walk, i.e. children are yielded after their parent.
        Note:
            The first cursor is the cursor for the translation unit, it has level 0.
        """
        if cursor is None:
            cursor = self.cursor
        yield from walk_cursors(cursor, postorder)

    def render_cursors(self, cursor=None):
        if cursor is None:
            cursor = self.cursor
        result = ""
        for (cursor, level, _) in self.walk_cursors_preorder(cursor):
            indent = "-" * (level)
            result += f"{indent}{str(cursor.kind).replace('CursorKind.','')} '{cursor.spelling}' '{cursor.displayname}' [TYPE-INFO {str(cursor.type.kind).replace('TypeKind.','')} '{cursor.type.spelling}' '{cursor.type.get_canonical().spelling}']"
            if cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                underlying_typedef_type = cursor.underlying_typedef_type
                result += f" [TYPEDEF-INFO '{underlying_typedef_type.spelling}']"
            result += "\n"
        return result

class Analysis:
    """Collection of routines for analyzing the contents of a C translation unit."""

    @staticmethod
    def _type_analysis_part_header():
        return [
            "cursor.type.spelling",
            "Type Layer Kinds",
            "cursor.type.get_canonical().spelling",
            "Canonical Type Layer Kinds",
            "Canonical Type Layer Kinds (Categorized)",
            "Canonical Type Layer Kinds (Categorized, Const)",
            "Canonical Type Layer Kinds (Categorized, Const, Finer)",
        ]

    @staticmethod
    def _type_analysis_part(clang_type: clang.cindex.Type):
        typehandler = TypeHandler(clang_type)
        type_kinds = ",".join(
            [
                str(t)
                for t in typehandler.clang_type_layer_kinds(
                    canonical=False, postorder=False
                )
            ]
        )
        canonical_type_kinds = ",".join(
            [
                str(t)
                for t in typehandler.clang_type_layer_kinds(
                    canonical=True, postorder=False
                )
            ]
        )
        categorized_canonical_type_layer_kinds = ",".join(
            [str(t) for t in typehandler.categorized_type_layer_kinds(postorder=False)]
        )
        categorized_canonical_type_layer_kinds_w_const = ",".join(
            [
                str(t)
                for t in typehandler.categorized_type_layer_kinds(
                    postorder=False, consider_const=True
                )
            ]
        )
        categorized_canonical_type_layer_kinds_finer_w_const = ",".join(
            [
                str(t)
                for t in typehandler.categorized_type_layer_kinds(
                    postorder=False, consider_const=True, subdivide_basic_types=True
                )
            ]
        )

        return [
            f"{clang_type.spelling}",
            f"[{type_kinds}]",
            f"{clang_type.get_canonical().spelling}",
            f"[{canonical_type_kinds}]",
            f"[{categorized_canonical_type_layer_kinds}]",
            f"[{categorized_canonical_type_layer_kinds_w_const}]",
            f"[{categorized_canonical_type_layer_kinds_finer_w_const}]",
        ]

    @staticmethod
    def subtree_as_csv(
        root: clang.cindex.Cursor, spelling: str, maxlevel: int, sep: str = ";"
    ):
        """Render a subtree as CSV table.

        Args:
            root (clang.cindex.Cursor): The cursor to do the walk on.
            spelling (str): The spelling/name of the searched cursor.
            maxlevel (int): Max level of the tree to print.
        """
        result = ""
        header = ["Location"]
        header += [f"Level {l}" for l in range(0, maxlevel + 1)]
        header += ["cursor.spelling"] + Analysis._type_analysis_part_header()

        result += sep.join(header) + "\n"
        activate_printing = False

        def descend_(cursor, level=0):
            nonlocal result
            nonlocal activate_printing
            nonlocal spelling
            nonlocal maxlevel
            nonlocal sep
            if cursor.location is not None:
                if cursor.location.file != None:
                    if spelling == None or cursor.spelling == spelling:
                        activate_printing = True
                    if activate_printing:
                        result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}{sep}"
                        indent = f"{sep}" * (level)
                        result += (
                            f"{indent}{str(cursor.kind).replace('CursorKind.','')}"
                        )
                        result += (maxlevel - level) * f"{sep}"

                        result += f"{sep}{cursor.spelling}"

                        result += sep + sep.join(
                            Analysis._type_analysis_part(cursor.type)
                        )
                        result += "\n"
            for child in cursor.get_children():
                descend_(child, level + 1)
            if cursor.spelling == spelling:
                activate_printing = False

        descend_(root)
        return result

    @staticmethod
    def type_declarations_as_csv(
        root: clang.cindex.Cursor,
        cursor_filter: callable = lambda cursor: True,
        maxlevel: int = 8,
        include_fields=True,
        sep: str = ";",
    ):
        """Renders nodes associated with type declarations.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["Location"]
        header += [f"Level {l}" for l in range(0, maxlevel + 1)]
        header += ["cursor.spelling"] + Analysis._type_analysis_part_header()
        result += sep.join(header) + "\n"
        for (cursor, level, _) in walk_cursors(root):
            if level > maxlevel:
                continue
            if cursor.kind in (
                clang.cindex.CursorKind.TRANSLATION_UNIT,
                clang.cindex.CursorKind.TYPEDEF_DECL,
                clang.cindex.CursorKind.STRUCT_DECL,
                clang.cindex.CursorKind.UNION_DECL,
                clang.cindex.CursorKind.FIELD_DECL,
                clang.cindex.CursorKind.ENUM_DECL,
                clang.cindex.CursorKind.ENUM_CONSTANT_DECL,
            ):
                if not include_fields and (
                    cursor.kind
                    in (
                        clang.cindex.CursorKind.FIELD_DECL,
                        clang.cindex.CursorKind.ENUM_CONSTANT_DECL,
                    )
                ):
                    continue
                if cursor.location is not None:
                    if cursor.location.file != None:
                        if cursor_filter(cursor):
                            result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}{sep}"
                            indent = f"{sep}" * (level)
                            result += (
                                f"{indent}{str(cursor.kind).replace('CursorKind.','')}"
                            )
                            result += (maxlevel - level) * f"{sep}"
                            result += f"{sep}{cursor.spelling}"

                            result += sep + sep.join(
                                Analysis._type_analysis_part(cursor.type)
                            )
                            result += "\n"
        return result

    @staticmethod
    def macros_as_csv(
        root: clang.cindex.Cursor,
        cursor_filter: callable = lambda cursor: True,
        sep: str = ";",
    ):
        """Returns an overview table of macro definitions.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["cursor.spelling", "Tokens (contains macro name and arguments)"]
        result += sep.join(header) + "\n"
        for (cursor, _, parent_stack) in walk_cursors(root):
            if cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
                if cursor_filter(cursor):
                    result += f"{cursor.spelling}"
                    tokens = ",".join(
                        [f"'{tk.spelling}'" for tk in cursor.get_tokens()]
                    )
                    result += f"{sep}[{tokens}]"
                    result += "\n"
        return result

    @staticmethod
    def parameter_and_return_types_as_csv(
        root: clang.cindex.Cursor,
        cursor_filter: callable = lambda cursor: True,
        sep: str = ";",
    ):
        """Returns an overview table of parameter and return values and their types as CSV table.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["Location"]
        header += [
            "Function",
            "Reference Kind",
            "CursorKind",
            "cursor.spelling",
        ] + Analysis._type_analysis_part_header()
        result += sep.join(header) + "\n"
        for (cursor, _, parent_stack) in walk_cursors(root):
            if cursor.kind in (
                clang.cindex.CursorKind.PARM_DECL,
                clang.cindex.CursorKind.FUNCTION_DECL,
            ):
                if cursor.location is not None:
                    if cursor.location.file != None:
                        if cursor_filter(cursor):
                            if cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                                func = parent_stack[1].spelling
                                kind = "Parameter"
                                clang_type = cursor.type
                            else:
                                func = cursor.spelling
                                kind = "Result"
                                clang_type = cursor.result_type

                            result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}"
                            result += f"{sep}{func}{sep}{kind}"
                            result += (
                                f"{sep}{str(cursor.kind).replace('CursorKind.','')}"
                            )
                            result += f"{sep}{cursor.spelling}"

                            result += sep + sep.join(
                                Analysis._type_analysis_part(clang_type)
                            )
                            result += "\n"
        return result
