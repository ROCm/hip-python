# AMD_COPYRIGHT

import os
import enum
import textwrap

import clang.cindex

__author__ = "AMD_AUTHOR"

indent = " " * 4


class CApiParser:
    """Parser for C APIs."""

    def __init__(self, filename: str, append_cflags: list[str] = []):
        """Parse the specified file.

        Args:
            filename (str): Path of the file to parse.
            append_cflags (list[str], optional): Additional flags to append when parsing.
        """
        self._filename = filename
        self._append_cflags = append_cflags
        self._cursor = None
        self._is_cursor_valid = False

    @property
    def filename(self):
        """Returns the filename specified at init."""
        return self._filename

    def parse(self):
        """Parse the specified file."""
        # print(self._append_cflags)
        self.translation_unit = clang.cindex.TranslationUnit.from_source(
            self.filename,
            args=["-x", "c"] + self._append_cflags,
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,  # keeps the macro defs as "fake" nodes
        )
        self._cursor = self.translation_unit.cursor
        self._is_cursor_valid = True
        return self

    def toplevel_cursors(self):
        """Top-levels cursors associated with the current file.

        All cursors found on the highest level of the
        translation unit, e.g. those for C functions.
        """
        assert self._is_cursor_valid
        return self._cursor.get_children()


class Node:
    """Wrapper for clang.cindex.Cursor.

    Allows to filter nodes via isinstance and further
    allows to compare cursors from different files
    that may have included the same header files.
    """

    def __init__(self, cursor: clang.cindex.Cursor):
        self._cursor = cursor

    @property
    def name(self):
        return self._cursor.spelling

    @property
    def type(self):
        return self._cursor.type.spelling

    @property
    def raw_comment(self):
        """Returns full (doxygen) comment for this node."""
        return self._cursor.raw_comment
        # todo parse the comment and adjust parameter types.

    @property
    def brief_comment(self):
        """Returns brief (doxygen) comment for this node."""
        return self._cursor.brief_comment

    def __id(self):
        return [
            self._cursor.mangled_name,
            self._cursor.location.file,
            self._cursor.location.line,
            self._cursor.location.column,
            self._cursor.location.offset,
        ]

    def __hash__(self):
        return hash(self.__id())

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.__id() == other.__id()

    # class helpers

    _hip_python_helper_types = []

    @classmethod
    def get_helper_type_name(cls, node):
        assert isinstance(node, Node)
        if node.type not in cls._hip_python_helper_types:
            cls._hip_python_helper_types.append(node.type)
        return (
            f"__hip_python_helper_type_{cls._hip_python_helper_types.index(node.type)}"
        )

    # factory functionality

    @staticmethod
    def _SUBCLASSES():
        # todo: implement registration method
        return (TypedefDecl, MacroDef, FunctionDecl, EnumDecl, UnionStructDecl)

    @staticmethod
    def match(cursor):
        """If the cursor models a node we are interested in."""
        for cls in Node._SUBCLASSES():
            if cls.match(cursor):
                return True
        return False

    @staticmethod
    def from_cursor(cursor):
        for cls in Node._SUBCLASSES():
            if cls.match(cursor):
                return cls(cursor)


class MacroDef(Node):
    """Macro definition. Must evaluate to an integer.
    User must ensure this by filtering out nodes that do not."""

    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION
        return False

    def render_cython_binding(self):
        """Render Cython binding for this C macro definition.

        Returns:
            str: Cython binding for this C macro definition.
        Note:
            Assumes that all macro values are of integer type.
            If that is not valid, filtering on a higher level has to ensure it.
        """
        return f"cdef int {self.name}"


class UserTypeDeclBase(Node):
    """Base class for user-defined/derived type declarations such
    as enum, struct, union declarations that can be nested within
    each other."""

    def walk_nested_user_type_decls_preorder(self):
        """Yields all struct decls (self and nested) in pre-order.

        Yields all struct decls (self and nested) in pre-order,
        i.e. yields every parent before its children."""
        yield self
        for child in self._cursor.get_children():
            if UserTypeDeclBase.match(child.kind):
                yield from UserTypeDeclBase.from_cursor(
                    child
                ).walk_nested_user_type_decls_preorder()

    def walk_nested_user_type_decls_postorder(self):
        """Yields all struct decls (self and nested) in post-order.

        Yields all struct decls (self and nested) in post-order,
        i.e. yields every child before its parent."""
        for child in self._cursor.get_children():
            if UserTypeDeclBase.match(child):
                yield from UserTypeDeclBase.from_cursor(
                    child
                ).walk_nested_user_type_decls_postorder()
        yield self

    # factory functionality

    @staticmethod
    def match(cursor):
        # todo: implement registration method
        if isinstance(cursor, clang.cindex.Cursor):
            return (
                cursor.kind == clang.cindex.CursorKind.STRUCT_DECL
                or cursor.kind == clang.cindex.CursorKind.UNION_DECL
                or cursor.kind == clang.cindex.CursorKind.ENUM_DECL
            )
        return False

    @staticmethod
    def from_cursor(cursor):
        for cls in (EnumDecl, UnionStructDecl):
            if cls.match(cursor):
                return cls(cursor)


class EnumDecl(UserTypeDeclBase):
    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.ENUM_DECL
        return False

    @property
    def is_anonymous(self):
        """If this struct does not have a name."""
        return self._cursor.spelling == ""

    @property
    def name(self):
        """Returns given name or generated name if struct is nested or does not have name."""
        if self.is_anonymous:
            return Node.get_helper_type_name(self)
        return self._cursor.spelling

    def enum_names(self):
        """Yields the enum constants' names."""
        for child in self._cursor.get_children():
            yield child.spelling

    def render_cython_binding(self):
        global indent
        return f"cdef enum {self.name}:\n" + textwrap.indent(
            "\n".join(self.enum_names()), indent
        )


class UnionStructDecl(UserTypeDeclBase):
    """Class for modelling both union and struct declarations."""

    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return (
                cursor.kind == clang.cindex.CursorKind.STRUCT_DECL
                or cursor.kind == clang.cindex.CursorKind.UNION_DECL
            )
        return False

    @property
    def is_union(self):
        """If this is a union."""
        return self._cursor.kind == clang.cindex.CursorKind.UNION_DECL

    @property
    def is_struct(self):
        """If this is a struct."""
        return self._cursor.kind == clang.cindex.CursorKind.STRUCT_DECL

    @property
    def is_anonymous(self):
        """If this struct/union does not have a name."""
        return self._cursor.spelling == ""

    @property
    def is_nested(self):
        """If this represents a struct that is declared in a parent struct."""
        return self._cursor.lexical_parent.kind == clang.cindex.CursorKind.STRUCT_DECL

    @property
    def name(self):
        """Returns given name or generated name if struct is nested or does not have name."""
        if self.is_anonymous or self.is_nested:
            return Node.get_helper_type_name(self)
        return self._cursor.spelling

    def get_fields(self):
        """Returns the fields/members of the struct."""
        return self._cursor.type.get_fields()

    def render_cython_binding(self):
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.

        Note:
            Does not generate any code for struct/union/enum declarations
            nested in this struct/union declaration.
            See `UserTypeDeclBase.walk_nested_user_type_decls_postorder` to obtain
            all struct declarations from a given one.

        See Also:
            get_struct_name
            UserTypeDeclBase.walk_nested_user_type_decls_postorder
        """
        global indent
        type_kind = "struct" if self.is_struct else "union"
        if self.is_nested or self.is_anonymous:
            result = f"ctypedef {type_kind} {self.name}:\n"
        else:
            result = f"cdef {type_kind} {self.name}:\n"
        fields = list(self.get_fields())
        result += textwrap.indent(
            "\n".join([f"{arg.type.spelling} {arg.spelling}" for arg in fields]), indent
        )
        if not len(fields):
            result += f"{indent}pass"
        return result


class TypedefDecl(Node):
    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL
        return False

    @property
    def aliased_type(self):
        first = next(self._cursor.get_children(), None)
        return first

    @property
    def is_aliasing_union_struct(self):
        return UnionStructDecl.match(self.aliased_type)

    @property
    def is_aliasing_enum(self):
        return EnumDecl.match(self.aliased_type)

    def render_cython_binding(self):
        if self.is_aliasing_union_struct:
            typename = UnionStructDecl(self.aliased_type).name
        elif self.is_aliasing_enum:
            typename = EnumDecl(self.aliased_type).name
        else:
            typename = self._cursor.type.spelling
        return f"ctypedef {typename} {self.name}"


class FunctionDecl(Node):
    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
        return False

    @property
    def result_type(self):
        return self._cursor.result_type.spelling

    def argnames(self):
        for arg in self._cursor.get_arguments():
            yield arg.spelling

    def render_c_args(self):
        for arg in self._cursor.get_arguments():
            yield f"{arg.type.spelling} {arg.spelling}"

    def render_c_arglist(self):
        return ",".join(list(self.render_c_args()))

    def render_cython_binding(self, suffix="nogil"):
        return f"{self.result_type} {self.name}({self.render_c_arglist()}) {suffix}"


class HipPlatform(enum.IntEnum):
    AMD = 0
    NVIDIA = 1

    @property
    def cflags(self):
        return ["-D", f"__HIP_PLATFORM_{self.name}__"]


class PackageGenerator:
    """Generate Python/Cython packages for a HIP C interface.

    Generates Python/Cython packages for a HIP C interface
    based on a list of header file names and the name of
    a library to link.
    """

    def __init__(
        self,
        pkg_name: str,
        include_dir: str,
        headers: list,
        dll: str,
        node_filter: callable = lambda x: True,
        platform=HipPlatform.AMD,
    ):
        """Constructor.

        Args:
            pkg_name (str): Name of the package that should be generated. Influences filesnames.
            include_dir (str): Name of the main include dir.
            headers (list): Name of the header files. Absolute paths or w.r.t. to include dir.
            dll (str): Name of the DLL/shared object to link.
            node_filter (callable, optional): Filter for selecting the nodes to include in generated output. Defaults to lambdax:True.
            platform (HipPlatform, optional): The hip platform to use. Defaults to HipPlatform.AMD.
        """
        self._pkg_name = pkg_name
        if not len(headers):
            raise RuntimeError("Argument 'headers' must not be empty")
        self._include_dir = include_dir
        self._headers = headers
        self._platform = platform
        self._apis = {}
        for h in self._headers:
            print(h)
            self._apis[h] = []
            abspath = os.path.join(include_dir, h)
            cflags = platform.cflags + ["-I", f"{include_dir}"]
            parser = CApiParser(abspath, append_cflags=cflags)
            parser.parse()
            for cursor in parser.toplevel_cursors():
                if Node.match(cursor):
                    node = Node.from_cursor(cursor)
                    if node_filter(node):
                        self._apis[h].append(node)
        self._dll = dll
        self._node_filter = node_filter
        # self._node_renamer = lambda name: name # unused, symbol renamer might be better name

    def _apis_from_all_files(self):
        """Yields all APIs from all specified files."""
        for apis_per_file in self._apis.values():
            yield from apis_per_file

    def create_cython_bindings(self):
        """Returns a triple of header, helper and extern part of a Cython bindings file.

        Returns a triple of header, helper and extern part of a Cython bindings file.
        The header part contains any import/cimport statements that are required.
        The helpers part contains any helper types that have been introduced for
        nested enum/struct/union types. The extern part contains Cython declarations
        per C declaration that we want to use.
        """
        global indent
        header_part = []
        helpers_part = []
        extern_part = []
        for filename, nodes in self._apis.items():
            extern_part.append(f'cdef extern from "{filename}":')
            for node in nodes:
                if isinstance(node, FunctionDecl):
                    extern_part.append(
                        textwrap.indent(node.render_cython_binding(), indent)
                    )
                elif isinstance(node, UserTypeDeclBase):
                    for decl_node in node.walk_nested_user_type_decls_postorder():
                        helpers_part.append(decl_node.render_cython_binding())
                    extern_part.append(
                        textwrap.indent(helpers_part.pop(-1), indent)
                    )  # last decl_node is the node itself
                else:
                    extern_part.append(
                        textwrap.indent(node.render_cython_binding(), indent)
                    )
        return (header_part, helpers_part, extern_part)

    def render_cython_bindings(self):
        """Returns the Cython bindings file content for the given headers."""
        (header_part, helpers_part, extern_part) = self.create_cython_bindings()
        nl = "\n"
        return f"""\
#AMD_COPYRIGHT
{nl.join(header_part)}
{nl.join(helpers_part)}
{nl.join(extern_part)}"""
