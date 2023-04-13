#AMD_COPYRIGHT

import collections
import re
import sys
import textwrap
import enum
import clang.cindex

from _codegen.cparser import walk_cursors, TypeHandler

__author__ = "AMD_AUTHOR"

indent = " "*4
helper_type_prefix = "hip_python_aux_"

class Warnings(enum.IntEnum):
    IGNORE = 0
    WARN = 1
    ERROR = 2

class CythonBackend:

    def __init__(self, 
                 translation_unit: clang.cindex.TranslationUnit,
                 warnings = Warnings.WARN,
                 ):
        self.root_cursor = translation_unit.cursor
        self.warnings = warnings

    def generate_nodes(self):
        """Generate a tree/graph out of the CParser's cursor
        that puts nodes into a parent-child relationship."""

        def first_child_cursors_of_kinds_(cursor: clang.cindex.Cursor,kinds: tuple[clang.cindex.CursorKind]):
            """Returns the first typeref child or None."""
            return next((child_cursor for child_cursor in cursor.get_children()
                        if child_cursor.kind in kinds), None)

        def lookup_referenced_type_(typeref_cursor: clang.cindex.Cursor):
            nonlocal root
            if typeref_cursor is not None:
                return root.lookup_type(typeref_cursor.type.get_canonical().spelling,
                                        typeref_cursor.type.spelling)
            return None

        def handle_top_level_cursor_(cursor: clang.cindex.Cursor,root: Root):
            if cursor.kind in  (clang.cindex.CursorKind.STRUCT_DECL,
                                clang.cindex.CursorKind.UNION_DECL,
                                clang.cindex.CursorKind.ENUM_DECL, ):
                handle_record_enum_or_field_cursor_(cursor,root)
            elif cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                node = Typedef(cursor,root)
                descend_into_child_cursors_(node)
                type_decl_cursor = first_child_cursors_of_kinds_(cursor,(clang.cindex.CursorKind.STRUCT_DECL,
                                                                 clang.cindex.CursorKind.UNION_DECL,
                                                                 clang.cindex.CursorKind.ENUM_DECL,))
                if type_decl_cursor is None:
                    typeref_cursor = first_child_cursors_of_kinds_(cursor,(clang.cindex.CursorKind.TYPE_REF,))
                    node.typeref = lookup_referenced_type_(typeref_cursor)
                    root.append(node)
                elif not len(type_decl_cursor.spelling): # found anonymous struct/union/enum child
                                                         # in case of anon enum
                    # replace the original node with the given one
                    anon_type_decl = lookup_referenced_type_(type_decl_cursor)
                    assert anon_type_decl != None and isinstance(anon_type_decl,(Enum,Record))
                    type_decl = handle_anon_typedef_child_cursor_(type_decl_cursor,node)
                    root.insert(anon_type_decl.index,type_decl)
                    root.remove(anon_type_decl)
                    # do not append typedef node
                elif type_decl_cursor.spelling != cursor.spelling: # child with different name
                    # update, append typedef node
                    node.typeref = lookup_referenced_type_(type_decl_cursor)
                    root.append(node)
                else: # child with same name
                    pass # do not append typedef node
            elif cursor.kind == clang.cindex.CursorKind.VAR_DECL:
                if self.warnings in (Warnings.WARN, Warnings.ERROR):
                    msg = f"VAR_DECL cursor '{cursor.spelling}' not handled (not implemented)"
                    if self.warnings == Warnings.WARN:
                        print(f"WARN: {msg}'",file=sys.stderr)
                    else:
                        print(f"ERROR: {msg}'",file=sys.stderr)
                        sys.exit(2)
            elif cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
                root.append(MacroDefinition(cursor,root))
            elif cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                typeref_cursor = first_child_cursors_of_kinds_(cursor,(clang.cindex.CursorKind.TYPE_REF,))
                typeref = lookup_referenced_type_(typeref_cursor)
                node = Function(cursor,root,typeref=typeref)
                root.append(node)
                descend_into_child_cursors_(node)

        def handle_anon_typedef_child_cursor_(cursor: clang.cindex.Cursor, parent: Typedef):
            root = parent.get_root()
            parent_cursor = parent.cursor
            assert (cursor.type.get_canonical().spelling == parent_cursor.type.get_canonical().spelling)
            assert (cursor.spelling == '' and cursor.type.spelling == parent_cursor.type.spelling)
            name = parent_cursor.spelling
            if cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
                node = Struct(cursor, root, name,
                              use_ctypedef=True)
            elif cursor.kind == clang.cindex.CursorKind.UNION_DECL:
                node = Union(cursor, root, name,
                             use_ctypedef=True)
            else:
                assert cursor.kind == clang.cindex.CursorKind.ENUM_DECL
                node = Enum(cursor, root, name,
                            use_ctypedef=True)
            return node

        def handle_record_enum_or_field_cursor_(cursor: clang.cindex.Cursor,parent: Node):
            is_anonymous = cursor.spelling == ""
            root = parent.get_root()
            if cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
                if is_anonymous:
                    node = NestedAnonStruct(cursor,root,parent)
                else:
                    node = Struct(cursor,root)
                descend_into_child_cursors_(node)
                root.append(node)
                parent.anon_type_decls.append(node)
            elif cursor.kind == clang.cindex.CursorKind.UNION_DECL:
                if is_anonymous:
                    node = NestedAnonUnion(cursor,root,parent)
                else:
                    node = Union(cursor,root)
                descend_into_child_cursors_(node)
                root.append(node)
                parent.anon_type_decls.append(node)
            elif cursor.kind == clang.cindex.CursorKind.ENUM_DECL:
                if is_anonymous and not isinstance(parent,Root): # anon enum is allowed
                    node = NestedAnonEnum(cursor,root,parent)
                else:
                    node = Enum(cursor,root)
                descend_into_child_cursors_(node)
                root.append(node)
                parent.anon_type_decls.append(node)
            #
            elif cursor.kind == clang.cindex.CursorKind.FIELD_DECL:
                typeref_cursor = first_child_cursors_of_kinds_(cursor,(clang.cindex.CursorKind.TYPE_REF,
                                                                       clang.cindex.CursorKind.STRUCT_DECL,
                                                                       clang.cindex.CursorKind.UNION_DECL,
                                                                       clang.cindex.CursorKind.ENUM_DECL,))
                typeref = lookup_referenced_type_(typeref_cursor)
                # TODO check that typeref is only None if no type is involved
                node = Field(cursor,parent,typeref=typeref)
                parent.append(node)

        def descend_(cursor,parent=None):
            assert isinstance(parent,Node)
            assert parent.cursor is not None
            parent_cursor = parent.cursor
            if parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
                handle_top_level_cursor_(cursor,parent)
            elif parent_cursor.kind in (
                clang.cindex.CursorKind.STRUCT_DECL,
                clang.cindex.CursorKind.UNION_DECL,
            ):
                handle_record_enum_or_field_cursor_(cursor,parent)
            elif parent_cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                if cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                    typeref_cursor = first_child_cursors_of_kinds_(cursor,(clang.cindex.CursorKind.TYPE_REF,
                                                                   clang.cindex.CursorKind.STRUCT_DECL,
                                                                   clang.cindex.CursorKind.UNION_DECL,
                                                                   clang.cindex.CursorKind.ENUM_DECL,))
                    typeref = lookup_referenced_type_(typeref_cursor)
                    node = Parm(cursor,parent,typeref=typeref)
                    parent.append(node)

        def descend_into_child_cursors_(node: Node):
            for child_cursor in node.cursor.get_children():
                descend_(child_cursor,node)

        root = Root(self.root_cursor)
        descend_into_child_cursors_(root)
        return root

class Node:

    def __init__(
        self, 
        cursor: clang.cindex.Cursor,
        parent,
    ):
        assert parent is None or isinstance(parent,Node)
        self.cursor = cursor
        self.parent = parent
        self.child_nodes = []
        self.anon_type_decls = []

    def append(self,node):
        assert isinstance(node,Node)
        self.child_nodes.append(node)

    def remove(self,node):
        assert isinstance(node,Node)
        self.child_nodes.remove(node)

    def insert(self,pos: int,node):
        assert isinstance(node,Node)
        self.child_nodes.insert(pos,node)

    @property
    def name(self):
        return self.cursor.spelling
    
    @property
    def file(self):
        """Returns the filename, or None for macro definitions."""
        if self.cursor.location.file != None:
            return self.cursor.location.file.name
        else:
            return None
    
    def get_root(self):
        curr = self
        while curr.parent != None:
            curr = curr.parent
        assert isinstance(curr,Root)
        return curr

    def _index(self,cls = None):
        """Index with respect to parent, considers only nodes as specified by `cls`.
        
        Determines the position of the given node
        it is parent's `child_nodes` list.

        Args:
            cls:    A class or a tuple of classes to include in the counting. Defaults to the Node type, i.e. 
                    all children are considered.
        """
        assert self.parent != None
        if cls == None:
            cls = Node
        num = 0
        for child in self.parent.child_nodes:
            if isinstance(child,cls):
                if child == self:
                    return num
                num += 1
        raise RuntimeError("Node must be present in parent's `child_nodes` list")

    @property
    def index(self):
        assert self.parent != None
        return self._index()

class Root(Node):
    
    def __init__(
        self, 
        cursor: clang.cindex.Cursor,
    ):
        Node.__init__(self,cursor,None)
        self.types = collections.OrderedDict()

    def lookup_all_types(self,canonical_typename: str) -> list:
        return self.types.get(canonical_typename,[])

    def lookup_type(self,
                    canonical_typename: str,
                    typename: str):
        """Lookup Type instances with the given canonical and non-canonical name.

        Lookup Type instances with the given canonical type spelling 
        `canonical_typename` and non-canonical type spelling `typename`.

        Note:
            Employs a two-step strategy to ensure that types are looked up correctly
            for references to combined typedef declarations such as
            
                `typedef struct same_name {/*...*/} same_name;`

            which, if parsed with libclang, result in

            * A libclang STRUCT_DECL cursor with type spelling "struct same_name" and canonical type spelling "struct same_name"
            * A libclang TYPEDEF_DECL cursor with type spelling "same_name" (!) and canonical type spelling "struct same_name"

            However, in Cython, we must specify a `cdef struct same_name` and cannot specify an
            additional `ctypedef struct same_name same_name`.
            (See the Cython guide on "Interfacing with External C Code".)
            Hence for such constructs only a single type node is emitted.

            To take the above case (reference to typedef declaration) into account, this method therefore performs two lookups:
            
            * First, it performs a lookup with the cursor's canonical typename ("struct same_name") 
                and the cursor's typename ("same_name") 
            * Second, it performs a lookup with the cursor's canonical typename ("struct same_name") 
                and the cursor's canonical typename ("struct same_name") instead of the cursor's typename.
            
            In the above scenario, the second lookup would then find the node that represents the `cdef struct same_name` node.
            
            Note that only in the above scenario, the second lookup will find a node. In all other scenarios, the 
            second lookup will return None.
        """
        for node in self.lookup_all_types(canonical_typename):
            if node.cursor.type.spelling in (typename,canonical_typename):
                return node
        return None

    def _canonical_typename(self,node: Node):
        return node.cursor.type.get_canonical().spelling

    def append(self,node):
        assert isinstance(node,Node)
        if isinstance(node,Type):
            canonical_typename = self._canonical_typename(node)
            if not canonical_typename in self.types:
                self.types[canonical_typename] = []
            self.types[canonical_typename].append(node)
        self.child_nodes.append(node)

    def remove(self,node):
        assert isinstance(node,Node)
        if isinstance(node,Type):
            canonical_typename = self._canonical_typename(node)
            if canonical_typename in self.types:
                assert node in self.types[canonical_typename]
                self.types[canonical_typename].remove(node)
        self.child_nodes.remove(node)

    def insert(self,pos: int,node):
        assert isinstance(node,Node)
        if isinstance(node,Type):
            canonical_typename = self._canonical_typename(node)
            if not canonical_typename in self.types:
                self.types[canonical_typename] = []
            self.types[canonical_typename].append(node)
        self.child_nodes.insert(pos,node)

class MacroDefinition(Node):

    def __init__(self, 
                 cursor: clang.cindex.Cursor, 
                 parent: Node):
        Node.__init__(self,cursor,parent)
        self.type = "int"
       
    def render_cython_c_binding(self):
        """Returns a Cython binding for this C macro definition.
        """
        return f"cdef {self.type} {self.cursor.spelling}"

class TypeHandlerMixin:

    def __init__(self,clang_type,typeref = None):
        self.typeref = typeref
        self._cached_typename = None
        self._clang_type = clang_type

    @staticmethod
    def canonical_typename(clang_type,searched_canonical_typename,repl_typename = None):
        """Returns a Cython-compatible typename for the given Clang type.

        If `record_enum_name` is provided, replaces elaborated C type names, e.g. `struct Foo`, 
        and anonymous types by `record_enum_name`.
        Otherwise, simply returns the spelling of `clang_type.get_canonical()`.

        Args:
            forced_record_enum_name (str): A forced typename for the struct, union, or enum part of the 
                                        canonical Clang typename.
        """
        result = clang_type.get_canonical().spelling
        if repl_typename == None:
            return result
        else:
            assert type(repl_typename) == str and repl_typename.isidentifier(), repl_typename
            typehandler = TypeHandler(clang_type)
            for clang_type_layer in typehandler.walk_clang_type_layers(canonical=True):
                if clang_type_layer.get_canonical().spelling == searched_canonical_typename: 
                    result = result.replace(
                        searched_canonical_typename,
                        repl_typename
                    )
            return result

    @property
    def typename(self):
        if self._cached_typename == None:
            if self.typeref != None:
                searched_typename = self.typeref.cursor.type.get_canonical().spelling
                repl_typename = self.typeref.name # TODO Set search pattern and subst here
            else:
                searched_typename = None
                repl_typename = None
            self._cached_typename = TypeHandlerMixin.canonical_typename(self._clang_type,searched_typename,repl_typename)
        return self._cached_typename
    
    def clang_type_kinds(self,
                         postorder=False,
                         canonical=False):
        return TypeHandler(self._clang_type).clang_type_kinds(
            postorder=postorder,canonical=canonical)
    
    def categorized_type_kinds(self,
                               postorder=False,
                               consider_const=False,
                               subdivide_basic_types: bool = False):
        return TypeHandler(self._clang_type).clang_type_kinds(
                postorder=postorder,
                consider_const=consider_const,
                subdivide_basic_types=subdivide_basic_types)

class Intent(enum.IntEnum):
    IN = 0
    OUT = 1
    INOUT = 2
    CREATE = 3 # OUT result that is also created

class Rank(enum.IntEnum):
    SCALAR = 0
    BUFFER = 1

class Field(Node,TypeHandlerMixin):
    
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref = None,
    ):
        Node.__init__(self,cursor,parent)
        TypeHandlerMixin.__init__(self,self.cursor.type,typeref)

    def cython_repr(self):
        return f"{self.typename} {self.name}"

class Type(Node):
    """Indicates that this node represents a type."""

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        name = None,
    ):
        Node.__init__(self,cursor,parent)
        self._name = name

    @property
    def name(self):
        if self._name == None:
            return Node.name.fget(self)
        else:
            return self._name

class Record(Type):
    
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        name = None,
        use_ctypedef: bool = False,
    ):
        Type.__init__(self,cursor,parent,name)
        self._use_ctypedef: bool = use_ctypedef

    @property
    def fields(self):
        for child in self.child_nodes:
            if isinstance(child,Field):
                yield child

    @property
    def c_record_kind(self) -> str:
        if self.cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
            return "struct"
        else:
            return "union"

    def _render_cython_c_binding_head(self) -> str:
        cython_def_kind = "ctypedef" if self._use_ctypedef else "cdef"
        return f"{cython_def_kind} {self.c_record_kind} {self.name}:\n"
    
    def render_cython_c_binding(self):
        """Render Cython binding for this struct/union declaration.

        Renders a Cython binding for this struct/union declaration, does
        not render declarations for nested types.

        Returns:
            str: Cython C-binding representation of this struct declaration.
        """
        global indent
        result = self._render_cython_c_binding_head()
        fields = list(self.fields)
        if len(fields):
            result += textwrap.indent(
                "\n".join([field.cython_repr() for field in fields]), indent
            )
        else:
            result += f"{indent}pass"
        return result

class Struct(Record):
    pass

class Union(Record):
    pass

class Enum(Type):
     
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        name = None,
        use_ctypedef: bool = False
    ):
        Type.__init__(self,cursor,parent,name=name)
        self._use_ctypedef: bool = use_ctypedef

    @property
    def is_anonymous(self):
        return not len(self.cursor.spelling)

    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        for child_cursor in self.cursor.get_children():
            yield f"{child_cursor.spelling}"

    def _render_cython_c_binding_head(self) -> str:
        cython_def_kind = "ctypedef" if self._use_ctypedef else "cdef"
        return f"{cython_def_kind} enum{'' if self.is_anonymous else ' '+self.name}:\n"

    def render_cython_c_binding(self):
        global indent
        return self._render_cython_c_binding_head() + textwrap.indent(
            "\n".join(self._render_cython_enums()), indent
        )
    
    def _render_python_enums(self,prefix: str):
        """Yields the enum constants' names."""
        for child_cursor in self.cursor.get_children():
            yield f"{child_cursor.spelling} = {prefix}{child_cursor.spelling}"
    
    def render_python_interface(self,prefix: str):
        """Renders an enum.IntEnum class.

        Note:
            Does not create an enum.IntEnum class but only exposes the enum constants 
            from the Cython package corresponding to the prefix if the 
            Enum is anonymous.
        """
        global indent
        if self.is_anonymous:
            return "\n".join(self._render_python_enums(prefix))
        else:
            return f"class {self.name}(enum.IntEnum):\n" + textwrap.indent(
                "\n".join(self._render_python_enums(prefix)), indent
            )

class NestedType:

    PREFIX = "" # Prefix to allow make the name unique or more distinguishable if same or similar name exists

    def __init__(self,orig_parent: Node):
        self.orig_parent: Node = orig_parent

    def _orig_index(self,cls = None):
        """Index with respect to the original parent, considers only nodes as specified by `cls`.
        
        Determines the position of the given node
        it is original parent's `child_nodes` list.

        Args:
            cls:    A class or a tuple of classes to include in the counting. Defaults to the Node type, i.e. 
                    all children are considered.
        """
        assert self.parent != None
        if cls == None:
            cls = Node
        num = 0
        for child in self.orig_parent.anon_type_decls:
            if isinstance(child,cls):
                if child == self:
                    return num
                num += 1
        raise RuntimeError("Node must be present in orig_parents's `child_nodes` list")

    def _nested_type_name(self):
        curr = self
        name_parts = []
        while isinstance(curr,NestedType):
            name_parts.append(curr.name_part)
            curr = curr.orig_parent
        assert isinstance(curr,(Struct,Union,Enum,Root))
        if not isinstance(curr,Root):
            name_parts.append(curr.name)
        sep = "_"
        return f"{NestedType.PREFIX}{sep.join(reversed(name_parts))}"

class NestedAnonStruct(Struct,NestedType):
    
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        orig_parent,
    ):
        Struct.__init__(self,cursor,parent)
        NestedType.__init__(self,orig_parent)

    @property
    def anon_struct_index(self):
        return self._orig_index(NestedAnonStruct)
    
    @property
    def name_part(self):
        return f"anon_struct_{self.anon_struct_index}"
    
    @property
    def name(self):
        if self._name == None:
            return self._nested_type_name()
        else:
            return self._name

class NestedAnonUnion(Union,NestedType):
    
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        orig_parent,
    ):
        Union.__init__(self,cursor,parent)
        NestedType.__init__(self,orig_parent)

    @property
    def anon_union_index(self):
        return self._orig_index(NestedAnonUnion)
    
    @property
    def name_part(self):
        return f"anon_union_{self.anon_union_index}"
    
    @property
    def name(self):
        if self._name == None:
            return self._nested_type_name()
        else:
            return self._name
    
class NestedAnonEnum(Enum,NestedType):

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        orig_parent,
    ):
        Enum.__init__(self,cursor,parent)
        NestedType.__init__(self,orig_parent)

    def _render_cython_enums(self):
        """Yields the enum constants' names."""
        for child_cursor in self.cursor.get_children():
            yield f"{child_cursor.spelling} = {child_cursor.enum_value}"

    @property
    def anon_enum_index(self):
        return self._orig_index(NestedAnonEnum)
    
    @property
    def name_part(self):
        return f"anon_enum_{self.anon_enum_index}"
    
    @property
    def name(self):
        if self._name == None:
            return self._nested_type_name()
        else:
            return self._name

class Typedef(Type,TypeHandlerMixin):
    
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref = None,
    ):
        Type.__init__(self,cursor,parent)
        TypeHandlerMixin.__init__(self,self.cursor.type,typeref)

    def render_cython_c_binding(self):
        """Returns a Cython binding for this Typedef.
        """
        return f"ctypedef {self.typename} {self.cursor.spelling}"

class Parm(Node,TypeHandlerMixin):

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref = None,
    ):
        Node.__init__(self,cursor,parent)
        TypeHandlerMixin.__init__(self,self.cursor.type,typeref)
        self.rank = Rank.SCALAR
        self.intent = Intent.IN

    def cython_repr(self):
        return f"{self.typename} {self.name}"

class Function(Node,TypeHandlerMixin):
    
    CYTHON_FUNPTR_NAME_TEMPLATE = "{name}_funptr"

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        typeref = None, # result_type
    ):
        Node.__init__(self,cursor,parent)
        TypeHandlerMixin.__init__(self,self.cursor.result_type,typeref)

    @property
    def parms(self):
        for child in self.child_nodes:
            if isinstance(child,Parm):
                yield child

    def parm_names(self):
        for parm in self.parms:
            assert isinstance(parm,Parm)
            yield parm.name

    def parm_types(self):
        for parm in self.parms:
            assert isinstance(parm,Parm)
            yield parm.typename

    @property
    def raw_comment(self):
        """Returns full (doxygen) comment for this node."""
        return self.cursor.raw_comment
        # todo parse the comment and adjust parameter types.

    @property
    def brief_comment(self):
        """Returns brief (doxygen) comment for this node."""
        return self.cursor.brief_comment

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
        parm_decls = ",".join([arg.cython_repr() for arg in self.parms])
        return f"""\
 {self._raw_comment_as_python_comment().rstrip()}
 {self.typename} {self.name}({parm_decls}) {modifiers}
 """

    @property
    def cython_funptr_name(self):
        return Function.CYTHON_FUNPTR_NAME_TEMPLATE.format(name=self.name)

    def render_cython_lazy_loader_decl(self, modifiers="nogil"):
        parm_decls = ",".join([parm.cython_repr() for parm in self.parms])
        #return ""
        return f"""\
{self._raw_comment_as_python_comment().rstrip()}
cdef {self.typename} {self.name}({parm_decls}) {modifiers}
    """

    def render_cython_lazy_loader_def(self,lib_handle: str="__lib_handle", modifiers="nogil"):
        #parm_decls = ",".join([arg.cython_repr() for arg in self.parms])
        parm_types = ",".join(self.parm_types())
        parm_names = ",".join(self.parm_names())
        #return ""
        return f"""\
cdef void* {self.cython_funptr_name}
{self.render_cython_lazy_loader_decl(modifiers).strip()}:
global {lib_handle}
global {self.cython_funptr_name}
if {self.cython_funptr_name} == NULL:
    with gil:
        {self.cython_funptr_name} = loader.load_symbol({lib_handle}, "{self.name}")
return (<{self.typename} (*)({parm_types})> {self.cython_funptr_name})({parm_names})
"""