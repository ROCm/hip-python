#AMD_COPYRIGHT

import enum
import re
import textwrap
import clang.cindex

from _codegen.cparser import CParser

__author__ = "AMD_AUTHOR"

indent = " "*4
helper_type_prefix = "hip_python_aux_"

def create_nodes(cparser: CParser,
                 node_filter=lambda node: True):
    """Generate a tree/graph out of the CParser's cursor
    that puts nodes into a parent-child relationship."""
    ElaboratedTypeDeclBase.reset_helper_type_ctr()
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

def _link_nodes(nodes):
    """Double-link nodes that reference each other."""
    for i,node in enumerate(nodes):
        # connect references with their type
        if isinstance(node,TypeRef) and node.is_defined_in_translation_unit:
            for node2 in nodes:#[:i]:
                if node.match_declaration(node2):
                    node.definition = node2
                    break
            if node.definition == None:
                raise RuntimeError(f"No matching type definition found for type reference '{node.name}'")
        # Parent-child relationship
        for child_cursor in node.child_cursors():
            dummy = Node(child_cursor)
            #print(f"DUMMY: {node.name}->{dummy.name} {str([node.name for node in nodes])}")
            for node2 in nodes[:i]:
                if dummy == node2: # see: Node.__eq__
                    node.children.append(node2)
                    node2.parent = node # post-order walk puts children before parent
                    #print(f"MATCH {node2.name} {node2.parent.name}")
                    break

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

class Node:
    """Wrapper for clang.cindex.Cursor.

    Allows to filter nodes via isinstance and further
    allows to compare cursors from different files
    that may have included the same header files.
    """

    def __init__(self, cursor: clang.cindex.Cursor):
        self.cursor = cursor
        self.children = []
        self.parent = None
        self._source_loc = (
            self.cursor.mangled_name,
            self.file,
            self.cursor.location.line,
            self.cursor.location.column,
            self.cursor.location.offset,
        )

    @property
    def orig_name(self):
        return self.cursor.spelling

    def complete_init(self):
        pass

    @property
    def file(self):
        """Returns the filename, or None for macro definitions."""
        if self.cursor.location.file != None:
            return self.cursor.location.file.name
        else:
            return None

    @property
    def name(self):
        """Returns given name or empty string if unnamed entity."""
        return self.cursor.spelling

    @property
    def raw_comment(self):
        """Returns full (doxygen) comment for this node."""
        return self.cursor.raw_comment
        # todo parse the comment and adjust parameter types.

    @property
    def brief_comment(self):
        """Returns brief (doxygen) comment for this node."""
        return self.cursor.brief_comment

    def first_child(self,kind=None):
        """Returns the first child node."""
        if kind == None:
            return next(iter(self.children),None)
        else:
            for child in self.children:
                if isinstance(child,kind):
                    return child
            return None

    def child_cursors(self):
        return self.cursor.get_children()

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
    
    @staticmethod
    def _SUBCLASSES():
        return (TypedefDecl, MacroDefinition, FunctionDecl, 
                EnumDecl, UnionDecl, StructDecl, 
                ParmDecl, FieldDecl, TypeRef)

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
            
    # Rendering
    def render_cython_c_binding(self):
        return None
    
    # Rendering
    def render_python_interface(self):
        return None

class MacroDefinition(Node):
    
    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.MACRO_DEFINITION)
    
    def render_cython_c_binding(self,macro_type: callable = lambda node: "int"):
        """Returns a Cython binding for this C macro definition.

        Note:
            Assumes that all macro values are of integer type.
            If that is not valid, filtering on a higher level has to ensure it.
        """
        return f"cdef {macro_type(self)} {self.name}"

class FieldDecl(Node):
    """Macro definition. Must evaluate to an integer.
    User must ensure this by filtering out nodes that do not."""

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.FIELD_DECL)

    def _render_cython_code(self):
        """
        Note:
            Not if the first child of a 
        """
        linked_child = self.first_child(kind = ElaboratedTypeDeclBase)
        #print(linked_child)
        if linked_child != None:
            return f"{linked_child.full_name} {self.name}"
        return f"{self.cursor.type.spelling} {self.name}"
    
    def render_cython_c_binding(self):
        if self.parent == None:
            return self._render_cython_code()
        return ""

class ElaboratedTypeDeclBase(Node):
    """Base class for user-defined/derived type declarations such
    as enum, struct, union declarations that can be nested within
    each other."""

    # TODO put topmost parent's name into helper type name to allow to filter according to name.
    # Will allow to move away from global counter, make type easier to identify.

    _helper_type_ctr = 0
 
    @staticmethod
    def reset_helper_type_ctr():
        ElaboratedTypeDeclBase._helper_type_ctr = 0

    def __init__(self, cursor: clang.cindex.Cursor, type_kind: str):
        super().__init__(cursor)
        self._type_kind = type_kind
        self._name = self.cursor.spelling

    def complete_init(self):
        global helper_type_prefix
        if self.is_helper_type:
            self._user_type_id = ElaboratedTypeDeclBase._helper_type_ctr
            ElaboratedTypeDeclBase._helper_type_ctr += 1
            self._name = f"{helper_type_prefix}{self._type_kind}_{self._user_type_id}"

    @property
    def name(self):
        return self._name

    @property
    def is_unnamed(self):
        """If this struct/union/enum does not have a name."""
        return self.cursor.spelling == ""

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

    def _render_cython_c_binding_head(self):
        """Render the head of the Cython binding.
        Directly renders a ctypedef head if this is a helper type.
        Otherwise, renders a cdef head.
        """
        global indent
        if self.is_helper_type:
            return f"ctypedef {self._type_kind} {self.name}:\n"
        else:
            return f"cdef {self._type_kind} {self.name}:\n"

class EnumDecl(ElaboratedTypeDeclBase):
    
    def __init__(self, cursor: clang.cindex.Cursor):
        super().__init__(cursor, "enum")

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.ENUM_DECL)

    def enum_names(self):
        """Yields the enum constants' names."""
        for child in self.cursor.get_children():
            yield f"{child.spelling} = {child.enum_value}"

    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        for child in self.cursor.get_children():
            yield f"{child.spelling} = {child.enum_value}"

    def render_cython_c_binding(self):
        global indent
        return self._render_cython_c_binding_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )
    
    def render_python_interface(self):
        global indent
        return f"class {self.name}(enum.IntEnum):\n" + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )

class UnionStructDeclBase(ElaboratedTypeDeclBase):
    """Class for modelling both union and struct declarations."""

    def render_cython_c_binding(self):
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        global indent
        result = self._render_cython_c_binding_head()
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
        return self.first_child(kind=StructDecl) != None
    
    @property
    def is_aliasing_union_decl(self):
        return self.first_child(kind=UnionDecl) != None
    
    @property
    def is_aliasing_enum_decl(self):
        return self.first_child(kind=EnumDecl) != None

    def render_cython_c_binding(self):
        user_type_decl = self.first_child(kind=ElaboratedTypeDeclBase)
        if user_type_decl != None:
            assert isinstance(user_type_decl,ElaboratedTypeDeclBase)
            #print(f"{user_type_decl.name} vs {self.cursor.underlying_typedef_type.spelling}")
            # if ( user_type_decl.is_helper_type
            #      and self.cursor.underlying_typedef_type.kind == clang.cindex.TypeKind.RECORD
            # ):
            #     cython_type = user_type_decl.name
            # else:
            p_elaborated_type = re.compile(rf"(struct|union|enum)\s+{user_type_decl.orig_name}")
            cython_type = p_elaborated_type.sub(user_type_decl.name,self.cursor.underlying_typedef_type.spelling)
        else:
            cython_type = self.cursor.underlying_typedef_type.spelling
        return f"ctypedef {cython_type} {self.name}"
        
    def render_python_interface(self):
        global indent
        if self.is_aliasing_enum_decl:
            enum_decl = self.first_child()
            return f"class {self.name}(enum.IntEnum):\n" + textwrap.indent(
                "\n".join(enum_decl._render_cython_enums()), indent
            )

class ParmDecl(Node):
    """Node representing function parameter declaration in
    signature of a function."""
    
    class Intent(enum.IntEnum):
        IN = 0
        OUT = 1
        INOUT = 2

    class Rank(enum.IntEnum):
        SCALAR = 0
        BUFFER = 1

    # TODO: Per function, per arg, we need intent, rank, Python type, output type
    # Store in table?

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.PARM_DECL)

    @property
    def is_of_user_defined_type(self):
        return next((cursor for cursor in self.cursor.get_children()
            if cursor.kind == clang.cindex.CursorKind.TYPE_REF),
            None) != None

    @property
    def is_pointer(self):
        #canonical_type = self.cursor.type.get_canonical()
        #print(f"{self.parent.name}:{self.name}:{self.cursor.type.spelling}:{canonical_type.spelling}:{canonical_type.kind}")
        return False

    @property
    def has_default_value(self):
        return next((cursor for cursor in self.cursor.get_children()
            if cursor.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR),
            None) != None

    @property
    def default_value(self):
        unexposed_expr_cursor = next(iter(self.cursor.get_children()),
            self.cursor.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR,None)
        if unexposed_expr_cursor != None:
            return unexposed_expr_cursor.spelling
        else:
            return ""

    def _render_cython_code(self):
        return f"{self.cursor.type.spelling} {self.cursor.spelling}"

class TypeRef(Node):
     
    def __init__(self, cursor: clang.cindex.Cursor):
        """
        Note:
            Pointers to undefined types, e.g. struct UndefinedStruct*,
            result in a type reference that has no definition. In this case,
            `clang.cindex.Cursor.get_definition(self)` returns None.
        """
        super().__init__(cursor)
        self._definition = None
        if self.is_defined_in_translation_unit:
            self._dummy_declaration_node = Node(self.cursor.get_definition())
        else:
            self._dummy_declaration_node = None

    @staticmethod
    def match(cursor):
        return Node._match(cursor,clang.cindex.CursorKind.TYPE_REF)

    def match_declaration(self,other: Node):
        return self._dummy_declaration_node == other

    @property
    def is_defined_in_translation_unit(self):
        return not self.cursor.get_definition() is None

    @property
    def definition(self):
        return self._definition

    @definition.setter
    def definition(self,node: Node):
        self._definition = node

class FunctionDecl(Node):

    cython_funptr_name_template = "{name}_funptr"

    @staticmethod
    def match(cursor):
        if isinstance(cursor, clang.cindex.Cursor):
            return cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
        return False

    @property
    def result_type(self):
        return self.cursor.result_type.spelling
    
    @property
    def cython_funptr_name(self):
        return FunctionDecl.cython_funptr_name_template.format(name=self.name)

    def arguments(self):
        for arg in self.children:
            if isinstance(arg,ParmDecl):
                yield arg

    def argument_names(self):
        for arg in self.arguments():
            yield arg.name

    def argument_types(self):
        for arg in self.arguments():
            yield arg.cursor.type.spelling

    def _raw_comment_stripped(self):
        """Returns raw comment without the C comment chars."""
        comment: str = self.raw_comment
        p_comment = re.compile(r"^(\s*\*|\s*\/)+\s*")
        result = []
        if comment != None:
            for line1 in comment.splitlines(keepends=True):
                line = p_comment.sub("",line1,count=2)
                result.append(line)
        return "".join(result)

    def _raw_comment_as_python_comment(self):
        if self.raw_comment != None:
            comment = self._raw_comment_stripped()
            return "".join(["# " + l for l in comment.splitlines(keepends=True)])
        else:
            return ""
        
    def _raw_comment_as_docstring(self):
        return f'"""{"".join(self._raw_comment_stripped()).rstrip()}\n"""'

    def render_cython_c_binding(self, modifiers="nogil"):
        parm_decls = ",".join([arg._render_cython_code() for arg in self.arguments()])
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
{self.result_type} {self.name}({parm_decls}) {modifiers}
"""
    
    def render_cython_lazy_loader_decl(self, modifiers="nogil"):
        parm_decls = ",".join([arg._render_cython_code() for arg in self.arguments()])
        #return ""
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
cdef {self.result_type} {self.name}({parm_decls}) {modifiers}
"""

    def render_cython_lazy_loader_def(self,lib_handle: str="__lib_handle", modifiers="nogil"):
        parm_decls = ",".join([arg._render_cython_code() for arg in self.arguments()])
        parm_types = ",".join(self.argument_types())
        parm_names = ",".join(self.argument_names())
        #return ""
        return f"""\
cdef void* {self.cython_funptr_name}
{self.render_cython_lazy_loader_decl(modifiers).strip()}:
    global {lib_handle}
    global {self.cython_funptr_name}
    if {self.cython_funptr_name} == NULL:
        with gil:
            {self.cython_funptr_name} = loader.load_symbol({lib_handle}, "{self.name}")
    return (<{self.result_type} (*)({parm_types})> {self.cython_funptr_name})({parm_names})
"""

    def _render_python_function_buffer_arg(self,arg: ParmDecl):
        # TODO
        #return f"cast_to_void_ptr"
        pass

    def render_python_interface(self,
                                parm_intent: callable = lambda node: ParmDecl.Intent.IN,
                                parm_rank: callable = lambda node: ParmDecl.Rank.SCALAR):
        global indent
        
        if False: # TODO
            result = []
            # signature
            in_inout_args = [arg.name for arg in self.arguments() 
                            if parm_intent(arg) in [ParmDecl.Intent.IN,ParmDecl.Intent.INOUT]]
            result.append(f"def {self.name}({','.join(in_inout_args)}):")
            # doc string
            result.append(textwrap.indent(self._raw_comment_as_docstring(),indent))
            # body - convert frontend to backend types
            for arg in self.arguments():
                if parm_intent(arg) == ParmDecl.Intent.OUT:
                    result.append(f"{indent}{arg.name} = None")
                if arg.is_pointer: # or pointer like # or double pointer
                    if arg.is_of_user_defined_type:
                        if parm_rank(arg) == ParmDecl.Rank.SCALAR:
                            pass
                    else: # basic data type
                        if parm_rank(arg) == ParmDecl.Rank.SCALAR:
                            pass
                        else: # self.param_rank(arg) == Rank.BUFFER:
                            pass
            # call backend
            # convert backend to front end types
            result.append(f"{indent}pass")
            return "\n".join(result)
        return ""
