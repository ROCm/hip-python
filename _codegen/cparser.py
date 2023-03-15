#AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import clang.cindex

class CParser:
    """Parser for C APIs."""

    def __init__(self, filename: str, append_cflags: list[str] = []):
        """Parse the specified file.

        Args:
            filename (str): Path of the file to parse.
            append_cflags (list[str], optional): Additional flags to append when parsing.
        """
        self.filename = filename
        self.append_cflags = append_cflags
        self.translation_unit = None

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
                clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES |
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD # keeps the macro defs as "fake" nodes
            ),
        )
        return self

    def toplevel_cursors(self):
        """Top-levels cursors associated with the current file.

        All cursors found on the highest level of the
        translation unit, e.g. those for C functions.
        """
        return self.cursor.get_children()

    def walk_cursors_preorder(self,cursor=None):
        """Yields a tuple per cursor that consists of the cursor's level and the cursor.
        
        Yields a tuple per cursor that consists of the cursor's level and the cursor (in that order).
        Pre-order walk, i.e. children are yielded after their parent.
        The first cursor is the cursor for the translation unit, it has level 0."""
        if cursor is None:
            cursor = self.cursor
        def descend_(node,level=0):
            yield (level, node)
            for child in node.get_children():
                yield from descend_(child,level+1)
        yield from descend_(cursor)
    
    def walk_cursors_postorder(self,cursor=None):
        """Yields a tuple per cursor that consists of the cursor's level and the cursor.
        
        Yields a tuple per cursor that consists of the cursor's level and the cursor (in that order).
        Post-order walk, i.e. children are yielded before their parent.
        The first cursor is the cursor for the translation unit, it has level 0."""
        if cursor is None:
            cursor = self.cursor
        def descend_(node,level=0):
            for child in node.get_children():
                yield from descend_(child,level+1)
            yield (level, node)
        yield from descend_(self.cursor)
    
    def render_cursors(self,cursor=None):
        if cursor is None:
            cursor = self.cursor
        result = ""
        for (level,cursor) in self.walk_cursors_preorder(cursor):
            indent = "-"*(level)
            result += f"{indent}{str(cursor.kind).replace('CursorKind.','')} {cursor.spelling} {cursor.displayname} [TYPE-INFO {cursor.type.kind}] [PY-INFO {id(cursor)}]"
            if cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                underlying_decl = cursor.underlying_typedef_type.get_declaration()
                result += f" [TYPEDEF-INFO {underlying_decl.kind} {underlying_decl.spelling}]"
            result += "\n"
        return result