#AMD_COPYRIGHT

import re
import textwrap
import clang.cindex

from _codegen.cparser import CParser

__author__ = "AMD_AUTHOR"

indent = " "*4
helper_type_prefix = "hip_python_aux_"

class Node:
    """Wrapper for clang.cindex.Cursor.

    Allows to filter nodes via isinstance and further
    allows to compare cursors from different files
    that may have included the same header files.
    """

    def __init__(self, cursor: clang.cindex.Cursor):
        self._cursor = cursor
        self.children = []
        self.parent = None
        self._source_loc = (
            self._cursor.mangled_name,
            self.file,
            self._cursor.location.line,
            self._cursor.location.column,
            self._cursor.location.offset,
        )

    @property
    def orig_name(self):
        return self._cursor.spelling

    def complete_init(self):
        pass

    @property
    def file(self):
        """Returns the filename, or None for macro definitions."""
        if self._cursor.location.file != None:
            return self._cursor.location.file.name
        else:
            return None

    @property
    def name(self):
        """Returns given name or empty string if unnamed entity."""
        return self._cursor.spelling

    @property
    def raw_comment(self):
        """Returns full (doxygen) comment for this node."""
        return self._cursor.raw_comment
        # todo parse the comment and adjust parameter types.

    @property
    def brief_comment(self):
        """Returns brief (doxygen) comment for this node."""
        return self._cursor.brief_comment

    def first_child(self):
        """Returns the first child node."""
        return next(iter(self.children),None)

    def child_cursors(self):
        return self._cursor.get_children()

    def __hash__(self):
        """Hash based on location to allow filtering out 
        duplicates via `set`."""
        return hash(self._source_loc)

    def __eq__(self, other):
        """For a-==-b value comparisons, not a-is-b identity comparison."""
        #print("__eq__")
        if not isinstance(other, Node):
            return False
        #print(self.__source_loc)
        #print(other.__source_loc)
        return self._source_loc == other._source_loc

    # helper routines

    @staticmethod
    def _match(cursor: clang.cindex.Cursor,cursor_kind: clang.cindex.CursorKind):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == cursor_kind
        return False

    # factory functionality
    
    @staticmethod
    def _SUBCLASSES():
        # todo: implement registration method
        return (TypedefDecl, MacroDefinition, FunctionDecl, 
                EnumDecl, UnionDecl, StructDecl, 
                ParmDecl, FieldDecl)

    @staticmethod
    def match(cursor: clang.cindex.Cursor):
        """If the cursor models a node we are interested in."""
        for cls in Node._SUBCLASSES():
            if cls.match(cursor):
                return True
        return False

    @staticmethod
    def from_cursor(cursor: clang.cindex.Cursor):
        for cls in Node._SUBCLASSES():
            if cls.match(cursor):
                return cls(cursor)

class MacroDefinition(Node):
    
    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.MACRO_DEFINITION)
    
    def render_cython_binding(self):
        """Returns a Cython binding for this C macro definition.

        Note:
            Assumes that all macro values are of integer type.
            If that is not valid, filtering on a higher level has to ensure it.
        """
        return f"cdef int {self.name}"

class FieldDecl(Node):
    """Macro definition. Must evaluate to an integer.
    User must ensure this by filtering out nodes that do not."""

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.FIELD_DECL)

    def _render_cython_code(self):
        linked_child = self.first_child()
        #print(linked_child)
        if linked_child == None:
            return f"{self._cursor.type.spelling} {self.name}"
        elif isinstance(linked_child,TypedefDecl):
            return f"{linked_child.name} {self.name}"
        elif isinstance(linked_child,UserTypeDeclBase):
            return f"{linked_child.full_name} {self.name}"
        return f"{self._cursor.type.spelling} {self.name}"
    
    def render_cython_binding(self):
        if self.parent == None:
            return self._render_cython_code()
        return ""

class UserTypeDeclBase(Node):
    """Base class for user-defined/derived type declarations such
    as enum, struct, union declarations that can be nested within
    each other."""

    _helper_type_ctr = 0

    @staticmethod
    def reset_helper_type_ctr():
        UserTypeDeclBase._helper_type_ctr = 0

    def __init__(self, cursor: clang.cindex.Cursor, type_kind: str):
        super().__init__(cursor)
        self._type_kind = type_kind
        self._name = self._cursor.spelling

    def complete_init(self):
        global helper_type_prefix
        if self.is_helper_type:
            self._user_type_id = UserTypeDeclBase._helper_type_ctr
            UserTypeDeclBase._helper_type_ctr += 1
            self._name = f"{helper_type_prefix}{self._type_kind}_{self._user_type_id}"

    @property
    def name(self):
        return self._name

    @property
    def is_unnamed(self):
        """If this struct/union/enum does not have a name."""
        return self._cursor.spelling == ""

    @property
    def is_parent_struct_or_union_decl(self):
        """If this represents a struct/union/enum that is declared in a parent struct/union."""
        return isinstance(self.parent,(StructDecl,UnionDecl))
    
    @property
    def is_parent_typedef_decl(self):
        return isinstance(self.parent,TypedefDecl)
    
    @property
    def is_parent_typedef_decl_with_same_name(self):
        return self.is_parent_typedef_decl and self.parent.name == self.orig_name

    @property
    def is_helper_type(self):
        """Returns if this type is a helper type.

        Note
            A legal C expression `typedef struct Type Type`, where `Type` aliases `struct Type`, 
            would be declared in Cython as `ctypedef Type Type`, which is not legal.
            Hence, we introduce a helper type in this case also.
            More details:
            https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html#styles-of-struct-union-and-enum-declaration
        """
        return (self.is_unnamed 
                or self.is_parent_struct_or_union_decl 
                or self.is_parent_typedef_decl_with_same_name)

    @property
    def full_name(self):
        """Returns elaborated name if the type is a standalone struct
        or in a typedef with the same name"""
        if self.is_helper_type:
            return f"{self.name}"
        else:
            return f"{self._type_kind} {self.name}"

    def _render_cython_binding_head(self):
        """Render the head of the Cython binding.
        Directly renders a ctypedef head if this is a helper type.
        Otherwise, renders a cdef head.
        """
        global indent
        if self.is_helper_type:
            return f"ctypedef {self._type_kind} {self.name}:\n"
        else:
            return f"cdef {self._type_kind} {self.name}:\n"

class EnumDecl(UserTypeDeclBase):
    
    def __init__(self, cursor: clang.cindex.Cursor):
        super().__init__(cursor, "enum")

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.ENUM_DECL)

    def enum_names(self):
        """Yields the enum constants' names."""
        for child in self._cursor.get_children():
            yield f"{child.spelling} = {child.enum_value}"

    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        for child in self._cursor.get_children():
            yield f"{child.spelling} = {child.enum_value}"

    def render_cython_binding(self):
        global indent
        return self._render_cython_binding_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )
    
    def render_python_interface(self):
        global indent
        return f"class {self.name}(enum.IntEnum):\n" + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )

class UnionStructDeclBase(UserTypeDeclBase):
    """Class for modelling both union and struct declarations."""

    def render_cython_binding(self):
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        global indent
        result = self._render_cython_binding_head()
        fields = [child for child in self.children if isinstance(child,FieldDecl)]
        #print(fields)
        result += textwrap.indent(
            "\n".join([field._render_cython_code() for field in fields]), indent
        )
        if not len(fields):
            result += f"{indent}pass"
        return result
    
class StructDecl(UnionStructDeclBase):

    def __init__(self, cursor: clang.cindex.Cursor):
        super().__init__(cursor, "struct")

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.STRUCT_DECL)

class UnionDecl(UnionStructDeclBase):

    def __init__(self, cursor: clang.cindex.Cursor):
        super().__init__(cursor, "union")
    
    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.UNION_DECL)

class TypedefDecl(Node):
    
    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.TYPEDEF_DECL)

    @property
    def is_aliasing_struct_decl(self):
        user_type_decl = self.first_child()
        return isinstance(user_type_decl,StructDecl)
    
    @property
    def is_aliasing_union_decl(self):
        user_type_decl = self.first_child()
        return isinstance(user_type_decl,UnionDecl)
    
    @property
    def is_aliasing_enum_decl(self):
        user_type_decl = self.first_child()
        return isinstance(user_type_decl,EnumDecl)

    def render_cython_binding(self):
        user_type_decl = self.first_child()
        if user_type_decl != None:
            return f"ctypedef {user_type_decl.name} {self.name}"
        else:
            return f"ctypedef {self._cursor.underlying_typedef_type.spelling} {self.name}"
        
    def render_python_interface(self):
        global indent
        if self.is_aliasing_enum_decl:
            enum_decl = self.first_child()
            return f"class {self.name}(enum.IntEnum):\n" + textwrap.indent(
                "\n".join(enum_decl._render_cython_enums()), indent
            )

class ParmDecl(Node):
    """Macro definition. Must evaluate to an integer.
    User must ensure this by filtering out nodes that do not."""

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.PARM_DECL)

    def _render_cython_code(self):
        return f"{self._cursor.type.spelling} {self._cursor.spelling}"

class FunctionDecl(Node):
    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
        return False

    @property
    def result_type(self):
        return self._cursor.result_type.spelling

    def arguments(self):
        for arg in self.children:
            if isinstance(arg,ParmDecl):
                yield arg

    def argument_names(self):
        for arg in self.arguments():
            yield arg.name

    def render_cython_binding(self, suffix="nogil"):
        comment = (self.raw_comment+"\n") if self.raw_comment != None else ""
        comment = ["# " + l for l in comment.splitlines(keepends=True)]
        result = "".join(comment)
        parm_decls = ",".join([arg._render_cython_code() for arg in self.arguments()])
        result += f"{self.result_type} {self.name}({parm_decls}) {suffix}"
        return result
    
    p_comment = re.compile(r"^(\s*\*|\s*\/)+\s*")

    def comment_as_docstring(self):
        comment: str = self.raw_comment
        result = []
        if comment != None:
            for line1 in comment.splitlines(keepends=True):
                line = self.p_comment.sub("",line1,count=2)
                result.append(line)
        return f'"""{"".join(result).rstrip()}\n"""'

    def render_python_interface(self):
        global indent
        result = []
        parm_names = ",".join([arg.name for arg in self.arguments()])
        result.append(f"def {self.name}({parm_names})")
        result.append(textwrap.indent(self.comment_as_docstring(),indent))
        result.append(f"{indent}pass")
        return "\n".join(result)
    
def _link_nodes(nodes):
    """Double-link nodes that reference each other."""
    for i,node in enumerate(nodes):
        for child_cursor in node.child_cursors():
            dummy = Node(child_cursor)
            #print(f"DUMMY: {node.name}->{dummy.name} {str([node.name for node in nodes])}")
            for node2 in nodes[:i]:
                if dummy == node2:
                    node.children.append(node2)
                    node2.parent = node # post-order walk puts children before parent
                    #print(f"MATCH {node2.name} {node2.parent.name}")

def _filter_nodes(nodes,node_filter=lambda node: True):
    """Yields only nodes that are required on the top level.
    
    Yields only nodes that are required on the top level.
    This includes all top level nodes plus helper type declarations.
    """
    for node in nodes:
        #if node.parent != None: print(f"HASPARENT {node.__class__} {node.name} {node.parent.name}")
        if (
             not isinstance(node,ParmDecl)
             and (not isinstance(node,FieldDecl) or node.parent == None)
           ):
            if node_filter(node):
                yield node

def create_nodes(cparser: CParser, node_filter=lambda node: True):
    """Yields tree of nodes that reference each other."""
    UserTypeDeclBase.reset_helper_type_ctr()
    nodes = []
    for (_,cursor) in cparser.walk_cursors_postorder():
        if Node.match(cursor):
            node = Node.from_cursor(cursor)
            if node not in nodes:
                nodes.append(node)
    _link_nodes(nodes)
    for node in nodes:
        node.complete_init()
    return list(_filter_nodes(nodes, node_filter))