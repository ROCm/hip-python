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

import rocm.clang.cindex as ci


def walk_cursors(root: ci.Cursor, postorder=False):
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


def clang_type_kind(clang_type: ci.Type) -> ci.TypeKind:
    """Works around missing entries in the `clang.cindex.TypeKind` enum list.
    """
    if clang_type.spelling == "_Float16":
        return ci.TypeKind.HALF
    else:
        return clang_type.kind

class TypeHandler:
    _INSTANCE = None

    @classmethod
    def get(cls, clang_type: ci.Type):
        if cls._INSTANCE == None:
            cls._INSTANCE = TypeHandler(None)
        cls._INSTANCE.clang_type = clang_type
        return cls._INSTANCE

    @staticmethod
    def match_invalid_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.INVALID

    @staticmethod
    def match_void_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.VOID

    @staticmethod
    def match_bool_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.BOOL

    @staticmethod
    def match_char_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.CHAR_U,
            ci.TypeKind.UCHAR,
            ci.TypeKind.CHAR16,
            ci.TypeKind.CHAR32,
            ci.TypeKind.CHAR_S,
            ci.TypeKind.SCHAR,
            ci.TypeKind.WCHAR,
        )

    @staticmethod
    def match_char8_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.CHAR_U,
            ci.TypeKind.UCHAR,
            ci.TypeKind.CHAR_S,
            ci.TypeKind.SCHAR,
        )

    @staticmethod
    def match_int_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.USHORT,
            ci.TypeKind.UINT,
            ci.TypeKind.ULONG,
            ci.TypeKind.ULONGLONG,
            ci.TypeKind.UINT128,
            ci.TypeKind.SHORT,
            ci.TypeKind.INT,
            ci.TypeKind.LONG,
            ci.TypeKind.LONGLONG,
            ci.TypeKind.INT128,
        )

    @staticmethod
    def match_float_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.FLOAT,
            ci.TypeKind.DOUBLE,
            ci.TypeKind.LONGDOUBLE,
            ci.TypeKind.FLOAT128,
            ci.TypeKind.HALF,
            ci.TypeKind.IBM128,
        )

    @staticmethod
    def match_basic_datatype(type_kind: ci.TypeKind):
        return (
            TypeHandler.match_bool_type(type_kind)
            or TypeHandler.match_char_type(type_kind)
            or TypeHandler.match_int_type(type_kind)
            or TypeHandler.match_float_type(type_kind)
        )

    @staticmethod
    def match_complex_type(type_kind: ci.TypeKind):
        return ci.TypeKind == ci.TypeKind.COMPLEX

    @staticmethod
    def match_other_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.NULLPTR,
            ci.TypeKind.OVERLOAD,
            ci.TypeKind.DEPENDENT,
            ci.TypeKind.OBJCID,
            ci.TypeKind.OBJCCLASS,
            ci.TypeKind.OBJCSEL,
            ci.TypeKind.LVALUEREFERENCE,
            ci.TypeKind.RVALUEREFERENCE,
            ci.TypeKind.OBJCINTERFACE,
            ci.TypeKind.OBJCOBJECTPOINTER,
            ci.TypeKind.AUTO,
            ci.TypeKind.PIPE,
            ci.TypeKind.OCLIMAGE1DRO,
            ci.TypeKind.OCLIMAGE1DARRAYRO,
            ci.TypeKind.OCLIMAGE1DBUFFERRO,
            ci.TypeKind.OCLIMAGE2DRO,
            ci.TypeKind.OCLIMAGE2DARRAYRO,
            ci.TypeKind.OCLIMAGE2DDEPTHRO,
            ci.TypeKind.OCLIMAGE2DARRAYDEPTHRO,
            ci.TypeKind.OCLIMAGE2DMSAARO,
            ci.TypeKind.OCLIMAGE2DARRAYMSAARO,
            ci.TypeKind.OCLIMAGE2DMSAADEPTHRO,
            ci.TypeKind.OCLIMAGE2DARRAYMSAADEPTHRO,
            ci.TypeKind.OCLIMAGE3DRO,
            ci.TypeKind.OCLIMAGE1DWO,
            ci.TypeKind.OCLIMAGE1DARRAYWO,
            ci.TypeKind.OCLIMAGE1DBUFFERWO,
            ci.TypeKind.OCLIMAGE2DWO,
            ci.TypeKind.OCLIMAGE2DARRAYWO,
            ci.TypeKind.OCLIMAGE2DDEPTHWO,
            ci.TypeKind.OCLIMAGE2DARRAYDEPTHWO,
            ci.TypeKind.OCLIMAGE2DMSAAWO,
            ci.TypeKind.OCLIMAGE2DARRAYMSAAWO,
            ci.TypeKind.OCLIMAGE2DMSAADEPTHWO,
            ci.TypeKind.OCLIMAGE2DARRAYMSAADEPTHWO,
            ci.TypeKind.OCLIMAGE3DWO,
            ci.TypeKind.OCLIMAGE1DRW,
            ci.TypeKind.OCLIMAGE1DARRAYRW,
            ci.TypeKind.OCLIMAGE1DBUFFERRW,
            ci.TypeKind.OCLIMAGE2DRW,
            ci.TypeKind.OCLIMAGE2DARRAYRW,
            ci.TypeKind.OCLIMAGE2DDEPTHRW,
            ci.TypeKind.OCLIMAGE2DARRAYDEPTHRW,
            ci.TypeKind.OCLIMAGE2DMSAARW,
            ci.TypeKind.OCLIMAGE2DARRAYMSAARW,
            ci.TypeKind.OCLIMAGE2DMSAADEPTHRW,
            ci.TypeKind.OCLIMAGE2DARRAYMSAADEPTHRW,
            ci.TypeKind.OCLIMAGE3DRW,
            ci.TypeKind.OCLSAMPLER,
            ci.TypeKind.OCLEVENT,
            ci.TypeKind.OCLQUEUE,
            ci.TypeKind.OCLRESERVEID,
            ci.TypeKind.EXTVECTOR,
            ci.TypeKind.ATOMIC,
        )

    @staticmethod
    def match_pointer_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.POINTER,
            ci.TypeKind.BLOCKPOINTER,
            ci.TypeKind.MEMBERPOINTER,
        )

    @staticmethod
    def match_function_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.FUNCTIONNOPROTO,
            ci.TypeKind.FUNCTIONPROTO,
        )

    @staticmethod
    def match_arraylike_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.VECTOR,
            ci.TypeKind.VARIABLEARRAY,
            ci.TypeKind.DEPENDENTSIZEDARRAY,
            ci.TypeKind.CONSTANTARRAY,
            ci.TypeKind.INCOMPLETEARRAY,
        )

    @staticmethod
    def match_record_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.RECORD

    @staticmethod
    def match_enum_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.ENUM

    @staticmethod
    def match_record_or_enum_type(type_kind: ci.TypeKind):
        return type_kind in (
            ci.TypeKind.RECORD,
            ci.TypeKind.ENUM,
        )

    @staticmethod
    def match_elaborated_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.ELABORATED

    @staticmethod
    def match_typedef_type(type_kind: ci.TypeKind):
        return type_kind == ci.TypeKind.TYPEDEF

    def __init__(self, clang_type: ci.Type):
        self.clang_type = clang_type

    def walk_clang_type_layers(self, postorder=False, canonical=False):
        """Walks through the constitutents of a Clang type.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Note:
            Note that this is by default a pre-order walk, e.g., if we have a type `void *`,
            we will obtain first the pointer type and then the `void` type.
        """
        global clang_type_kind

        def descend_(clang_type: ci.Type):
            nonlocal postorder
            type_kind = clang_type_kind(clang_type)
            if TypeHandler.match_invalid_type(type_kind):
                yield clang_type
            elif TypeHandler.match_void_type(
                type_kind
            ) or TypeHandler.match_basic_datatype(type_kind):
                yield clang_type
            elif TypeHandler.match_pointer_type(type_kind):
                if postorder:
                    yield from descend_(clang_type.get_pointee())
                yield clang_type
                if not postorder:
                    yield from descend_(clang_type.get_pointee())
            elif TypeHandler.match_function_type(type_kind):
                yield clang_type
            elif TypeHandler.match_arraylike_type(
                type_kind
            ) or TypeHandler.match_complex_type(type_kind):
                if postorder:
                    yield from descend_(clang_type.get_array_element_type())
                yield clang_type
                if not postorder:
                    yield from descend_(clang_type.get_array_element_type())
            elif TypeHandler.match_record_or_enum_type(type_kind):
                yield clang_type
            elif TypeHandler.match_typedef_type(type_kind):
                underlying_type = clang_type.get_declaration().underlying_typedef_type
                if postorder:
                    yield from descend_(underlying_type)
                yield clang_type
                if not postorder:
                    yield from descend_(underlying_type)
            elif TypeHandler.match_elaborated_type(type_kind):
                named_type = clang_type.get_named_type()
                if postorder:
                    yield from descend_(named_type)
                yield clang_type
                if not postorder:
                    yield from descend_(named_type)
            elif TypeHandler.match_other_type(type_kind):
                yield clang_type
            else:
                raise RuntimeError(f"unknown type kind '{type_kind.spelling}'")

        if canonical:
            yield from descend_(self.clang_type.get_canonical())
        else:
            yield from descend_(self.clang_type)

    def clang_type_layer_kinds(self, postorder=False, canonical=False):
        """Yields the Clang type kinds that constitute this type.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Note:
            Note that this is by default a pre-order walk, e.g., if we have a type `void *`,
            we will obtain first the pointer type and then the `void` type.
        """
        for clang_type in self.walk_clang_type_layers(postorder, canonical):
            yield clang_type.kind


class CParser:
    """Parser for C APIs."""

    _CLANG_RES_DIR = None

    @classmethod
    def set_clang_res_dir(cls, clang_res_dir: str):
        """Set the clang resource dir.
        """
        # TODO check path
        cls._CLANG_RES_DIR = clang_res_dir

    def __init__(self, filename: str, append_cflags: list = [], unsaved_files=None):
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
        if not CParser._CLANG_RES_DIR:
            raise RuntimeError("Clang resource dir is not set.")
        # print(self._append_cflags)
        self.translation_unit = ci.TranslationUnit.from_source(
            self.filename,
            args=["-x", "c","-resource-dir",CParser._CLANG_RES_DIR] + self.append_cflags,
            options=(
                ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD  # keeps the macro defs as "fake" nodes without location
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
