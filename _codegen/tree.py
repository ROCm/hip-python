# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import collections
import re
import sys
import clang.cindex

from . import control
from . import cparser
from . import cython

indent = " " * 4

__MacroDefinitionMixins = (cython.MacroDefinitionMixin,)
__FieldMixins = (cython.FieldMixin,)
__StructMixins = (cython.StructMixin,)
__UnionMixins = (cython.UnionMixin,)
__EnumMixins = (cython.EnumMixin,)
__TypedefMixins = (cython.TypedefMixin,)
__TypedefedFunctionPointerMixins = (cython.TypedefedFunctionPointerMixin,)
__AnonymousFunctionPointerMixins = (cython.AnonymousFunctionPointerMixin,)
__ParmMixins = (cython.ParmMixin,)
__FunctionMixin = (cython.FunctionMixin,)


class Node:
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        assert parent is None or isinstance(parent, Node)
        self.cursor = cursor
        self.parent = parent
        self.child_nodes = []

    def append(self, node):
        assert isinstance(node, Node)
        if isinstance(node, Type):
            self.get_root().append_type(node)
        self.child_nodes.append(node)

    def remove(self, node):
        assert isinstance(node, Node)
        if isinstance(node, Type):
            self.get_root().remove_type(node)
        self.child_nodes.remove(node)

    def insert(self, pos: int, node):
        assert isinstance(node, Node)
        if isinstance(node, Type):
            self.get_root().append_type(node)
        self.child_nodes.insert(pos, node)

    @property
    def name(self):
        return self.cursor.spelling

    def global_name(self, sep: str = None):
        """Returns node's name with respect to its parents.

        Args:
            sep (`str`):  A separator to use for joining the individual names. If None is passed, the list is returned.
                          Defaults to None.
        """
        assert isinstance(self, (Node))
        curr = self
        name_parts = []
        while not isinstance(curr, Root):
            name_parts.append(curr.name)
            curr = curr.parent
        if sep == None:
            return name_parts
        else:
            return f"{sep.join(reversed(name_parts))}"

    @property
    def is_cursor_anonymous(self):
        """If the cursor is anonymous.
        Note:
            Always use the raw cursor as 'name' might be overwritten.
        """
        assert isinstance(self, Node)
        return len(self.cursor.spelling) == 0

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
        assert isinstance(curr, Root)
        return curr

    def _index(self, cls=None):
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
            if isinstance(child, cls):
                if child == self:
                    return num
                num += 1
        raise RuntimeError("Node must be present in parent's `child_nodes` list")

    @property
    def index(self):
        assert self.parent != None
        return self._index()

    def walk(self, postorder=True):
        if postorder:
            for child in self.child_nodes:
                yield from child.walk()
        yield self
        if not postorder:
            for child in self.child_nodes:
                yield from child.walk()


class Root(Node):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
    ):
        Node.__init__(self, cursor, None)
        self.types = collections.OrderedDict()

    def lookup_all_types(self, canonical_typename: str) -> list:
        return self.types.get(canonical_typename, [])

    def lookup_type(self, canonical_typename: str, typename: str):
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
            if node.cursor.type.spelling in (typename, canonical_typename):
                return node
        return None

    def lookup_type_from_cursor(self, cursor: clang.cindex.Cursor):
        """Lookup Type instances via the cursor's canonical and non-canonical typename.

        See:
            Root.lookup_type
        """
        if cursor is not None:
            result = self.lookup_type(
                cursor.type.get_canonical().spelling, cursor.type.spelling
            )
            return result
        return None

    def _canonical_typename(self, node: Node):
        return node.cursor.type.get_canonical().spelling

    def append_type(self, node):
        canonical_typename = self._canonical_typename(node)
        if not canonical_typename in self.types:
            self.types[canonical_typename] = []
        self.types[canonical_typename].append(node)

    def remove_type(self, node):
        canonical_typename = self._canonical_typename(node)
        if canonical_typename in self.types:
            assert node in self.types[canonical_typename]
            self.types[canonical_typename].remove(node)


class MacroDefinition(Node, *__MacroDefinitionMixins):
    def __init__(self, cursor: clang.cindex.Cursor, parent: Node):
        Node.__init__(self, cursor, parent)


class Typed:
    def __init__(self, clang_type: clang.cindex.Type, typeref=None):
        self.typeref: Node = typeref
        self._clang_type: clang.cindex.Type = clang_type
        self._type_handler = cparser.TypeHandler(clang_type)

    @staticmethod
    def canonical_typename(
        typehandler: cparser.TypeHandler,
        searched_canonical_typename,
        repl_typename=None,
    ):
        """Returns a Cython-compatible typename for the given Clang type.

        If `record_enum_name` is provided, replaces elaborated C type names, e.g. `struct Foo`,
        and anonymous types by `record_enum_name`.
        Otherwise, simply returns the spelling of `clang_type.get_canonical()`.

        Args:
            forced_record_enum_name (str): A forced typename for the struct, union, or enum part of the
                                           canonical Clang typename.
        """
        canonical_type_to_modify = typehandler.clang_type.get_canonical().spelling
        if repl_typename == None:
            return canonical_type_to_modify
        else:
            assert (
                type(repl_typename) == str and repl_typename.isidentifier()
            ), repl_typename
            for clang_type_layer in typehandler.walk_clang_type_layers(
                postorder=True,  # must be post-order to go from inside to outside
                canonical=True,
            ):
                layer_canonical_type_spelling = (
                    clang_type_layer.get_canonical().spelling
                )
                if layer_canonical_type_spelling.startswith(
                    searched_canonical_typename
                ) or layer_canonical_type_spelling.endswith(  # pointer with optional trailing modifiers
                    searched_canonical_typename
                ):  # other (canonical!) type with optional preceding modifiers
                    start_incl = canonical_type_to_modify.index(
                        layer_canonical_type_spelling
                    )
                    end_excl = len(layer_canonical_type_spelling)
                    if start_incl > 0:
                        preceding = canonical_type_to_modify[0 : start_incl - 1]
                    else:
                        preceding = ""
                    return f"{preceding}{repl_typename}{canonical_type_to_modify[start_incl+end_excl:]}"
            raise RuntimeError(
                f"typename '{searched_canonical_typename}' is no part of '{canonical_type_to_modify}'"
            )

    def global_typename(self, sep: str, renamer: callable = lambda name: name):
        if sep == None:
            raise ValueError("sep may not be None")
        if self.typeref is not None:
            searched_typename = self.typeref.cursor.type.get_canonical().spelling
            repl_typename = renamer(self.typeref.global_name(sep))
        else:
            searched_typename = None
            repl_typename = None
        return Typed.canonical_typename(
            self._type_handler, searched_typename, repl_typename
        )

    def typename(self, renamer: callable = lambda name: name):
        if self.typeref is not None:
            searched_typename = self.typeref.cursor.type.get_canonical().spelling
            repl_typename = renamer(self.typeref.name)
        else:
            searched_typename = None
            repl_typename = None
        return Typed.canonical_typename(
            self._type_handler, searched_typename, repl_typename
        )

    def clang_type_layer_kinds(self, postorder=False, canonical=False):
        return self._type_handler.clang_type_layer_kinds(
            postorder=postorder, canonical=canonical
        )

    def categorized_type_layer_kinds(
        self, postorder=False, consider_const=False, subdivide_basic_types: bool = False
    ):
        return self._type_handler.categorized_type_layer_kinds(
            postorder=postorder,
            consider_const=consider_const,
            subdivide_basic_types=subdivide_basic_types,
        )

    def const_qualifiers(self,postorder=False,canonical=False):
        """Yields a flag per type layer that constitute this type if 
        the layer is const qualified.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Yields:
            bool: Per type layer, yields a flag indicating if ``const`` is specified for this layer.
        """
        return self._type_handler.const_qualifiers(
            postorder=postorder,canonical=canonical)

    def get_rank(
        self,
        constant_array: bool = True,
        incomplete_array: bool = True,
        pointer: bool = True,
    ):
        """Array rank of the type.
        
        Counts layers of the type that can be interpreted as array dimension.
        By default constant arrays, incomplete arrays or pointers are counted
        as array dimension. Stops counting as soon as it finds anything else.

        Args:
            const_array (bool, optional): Consider const arrays. Defaults to True.
            incomplete_array (bool, optional): Consider incomplete arrays. Defaults to True.
            pointer (bool, optional): Consider pointers as array dimensions. Defaults to True.

        Returns:
            int: Rank of the array, with respect to the options.
        """
        from clang.cindex import TypeKind

        TypeCategory = cparser.TypeHandler.TypeCategory
        rank_counted: int = 0
        for kind in self._type_handler.clang_type_layer_kinds(canonical=True):
            if constant_array and kind == TypeKind.CONSTANTARRAY:
                rank_counted += 1
            elif incomplete_array and kind == TypeKind.INCOMPLETEARRAY:
                rank_counted += 1
            elif pointer and kind == TypeKind.POINTER:
                rank_counted += 1
            else:
                break
        return rank_counted

    def has_rank(
        self,
        rank: int,
        constant_array: bool = True,
        incomplete_array: bool = True,
        pointer: bool = True,
    ):
        """If the type has the given rank.
        
        See:
            get_rank

        Returns:
            bool: If the rank matches the input.
        """
        return self.get_rank(constant_array,incomplete_array,pointer) == rank
            
    def get_pointer_degree(self) -> int:
        """Returns number of outer type layers which are of TypeKind.POINTER.
        """
        return self.get_rank(constant_array=False,incomplete_array=False,pointer=True)

    def _is_pointer_to_kind(self,
                            type_kind,
                            degree: int = 1):
        """If this is a pointerof the given ``degree`` to the given type kind.
        Note:
            Does not check for any const modifiers.
        """
        if isinstance(type_kind,tuple):
            type_kinds = type_kind
        else:
            assert isinstance(type_kind,clang.cindex.TypeKind)
            type_kinds = (type_kind,)

        if list(self._type_handler.clang_type_layer_kinds(canonical=True))[-1] in type_kinds:
            return self.get_pointer_degree() == degree
        
    def _is_pointer_to_category(self,
                               type_category,
                               degree: int = 1,
                               subdivide_categories:bool =False):
        """If this is a pointerof the given ``degree`` to the given type category.
        Note:
            Does not check for any const modifiers.
        """
        if isinstance(type_category,tuple):
            type_categories = type_category
        else:
            assert isinstance(type_category,self._type_handler.TypeCategory)
            type_categories = (type_category,)

        if list(self._type_handler.categorized_type_layer_kinds(subdivide_basic_types=subdivide_categories))[-1] in type_categories:
            return self.get_pointer_degree() == degree

    @property
    def is_void(self):
        """If this is a void type."""
        from clang.cindex import TypeKind

        return (
            next(self._type_handler.clang_type_layer_kinds(canonical=True))
            == TypeKind.VOID
        )

    def is_pointer_to_void(self,degree: int = 1):
        """If this is a record (struct, union) pointer of the given degree.
        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self._is_pointer_to_kind(TypeKind.VOID,degree)
    
    def is_pointer_to_char(self,degree: int = 1):
        """If this is a record (struct, union) pointer of the given degree.
        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self._is_pointer_to_kind(TypeKind.CHAR_S,degree)
    
    def is_pointer_to_basic_type(self,degree: int = 1):
        """If this is a record (struct, union) pointer of the given degree.
        Note:
            Does not check for any const modifiers.
        """
        TypeCategory = cparser.TypeHandler.TypeCategory

        return self._is_pointer_to_category(TypeCategory.BASIC,degree)
    
    def is_pointer_to_record(self,degree: int = 1):
        """If this is a void pointer of the given degree.
        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self._is_pointer_to_kind(TypeKind.RECORD,degree)
    
    def is_pointer_to_enum(self,degree: int = 1):
        """If this is a enum pointer of the given degree.
        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self._is_pointer_to_kind(TypeKind.ENUM,degree)

    @property
    def is_any_pointer(self):
        """If this is any form of pointer, i.e. the outer most type layer must be a pointer."""
        from clang.cindex import TypeKind

        return self.get_rank(constant_array=False,incomplete_array=False,pointer=True) > 0

    @property
    def is_any_array(self):
        """If this is any form of array."""
        TypeCategory = cparser.TypeHandler.TypeCategory
        return (
            next(self._type_handler.categorized_type_layer_kinds())
            == TypeCategory.ARRAY
        )

    @property
    def is_record(self):
        """If this is a record (struct, union)."""
        from clang.cindex import TypeKind

        return (
            next(self._type_handler.clang_type_layer_kinds(canonical=True))
            == TypeKind.RECORD
        )

    @property
    def is_record_constantarray(self):
        """If this is a struct or union array."""
        # TODO multi-dim arrays
        from clang.cindex import TypeKind

        return list(self._type_handler.clang_type_layer_kinds(canonical=True)) == [
            TypeKind.CONSTANTARRAY,
            TypeKind.RECORD,
        ]

    @property
    def is_enum(self):
        """If this is an enum."""
        from clang.cindex import TypeKind

        return (
            next(self._type_handler.clang_type_layer_kinds(canonical=True))
            == TypeKind.ENUM
        )

    @property
    def is_enum_constantarray(self):
        """If this is an enum array."""
        # TODO multi-dim arrays
        from clang.cindex import TypeKind

        return list(self._type_handler.clang_type_layer_kinds(canonical=True)) == [
            TypeKind.CONSTANTARRAY,
            TypeKind.ENUM,
        ]

    @property
    def is_basic_type(self):
        """If this is a pointer to a struct or enum."""
        TypeCategory = cparser.TypeHandler.TypeCategory
        return list(self._type_handler.categorized_type_layer_kinds()) in [
            [TypeCategory.BASIC],
        ]
    
    @property
    def is_basic_type_constarray(self):
        """If this is a pointer to a struct or enum."""
        # TODO multi-dim arrays
        from clang.cindex import TypeKind

        TypeCategory = cparser.TypeHandler.TypeCategory
        kinds = list(self._type_handler.clang_type_layer_kinds(canonical=True))
        if len(kinds) == 2:
            categories = list(self._type_handler.categorized_type_layer_kinds())
            return (
                kinds[0] == TypeKind.CONSTANTARRAY
                and categories[1] == TypeCategory.BASIC
            )
        return False

    @property
    def is_char_incompletearray(self):
        """If this is an incomplete array of chars."""
        from clang.cindex import TypeKind

        return list(self._type_handler.clang_type_layer_kinds(canonical=True)) in [
            [TypeKind.INCOMPLETEARRAY, TypeKind.CHAR_S],
        ]

    @property
    def is_scalar(self):
        """If the type is a scalar of basic, record, or enum type.

        Returns:
            bool: If the type is a scalar of basic, record, or enum type.
        """
        return self.is_basic_type or self.is_record or self.is_enum

    @property
    def is_double_pointer_to_non_const_type(self):
        """If the type is something like ``void**``, ``char **``, ...

        Returns:
            bool: If the type is something like ``void**``, ``char **``, ...
        """
        TypeCategory = self._type_handler.TypeCategory
        categories_w_const = list(
            self.categorized_type_layer_kinds(consider_const=True)
        )
        return categories_w_const[0:2] in (
            [TypeCategory.POINTER, TypeCategory.POINTER],
        ) and categories_w_const[2] in (
            TypeCategory.VOID,
            TypeCategory.BASIC,
            TypeCategory.RECORD,
            TypeCategory.ENUM,
        )


class Field(Node, Typed, *__FieldMixins):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Node.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)


class Type(Node):
    """Indicates that this node represents a type."""

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Node.__init__(self, cursor, parent)
        self._name = None

    def overwrite_name(self, name):
        self._name = name

    @property
    def is_anonymous(self):
        """If this type is anonymous, i.e. the
        cursor's spelling is anonymous while the `_name` member has
        not been overwritten."""
        return self.is_cursor_anonymous and self._name == None

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
        from_typedef_with_anon_child: bool = False,
    ):
        Type.__init__(self, cursor, parent)
        self._from_typedef_with_anon_child: bool = from_typedef_with_anon_child

    @property
    def fields(self):
        """Fields specified for this type."""
        for child in self.child_nodes:
            if isinstance(child, Field):
                yield child

    @property
    def is_incomplete(self):
        """If the type does not have any fields."""
        return next(self.fields, None) == None


class Struct(Record, *__StructMixins):
    pass


class Union(Record, *__UnionMixins):
    pass


class Enum(Type, *__EnumMixins):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        from_typedef_with_anon_child: bool = False,
    ):
        Type.__init__(self, cursor, parent)
        self._from_typedef_with_anon_child: bool = from_typedef_with_anon_child

    @property
    def is_incomplete(self):
        """If the type does not have any fields."""
        for child_cursor in self.cursor.get_children():
            if child_cursor.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL:
                return False
        return True


class Nested:
    """A marker for nested struct/union/enum types."""

    pass


class NestedStruct(Struct, Nested):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Struct.__init__(self, cursor, parent)

    @property
    def struct_index(self):
        return self._index(NestedStruct)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"struct_{self.struct_index}"
            else:
                return Struct.name.fget(self)
        else:
            return self._name


class NestedUnion(Union, Nested):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Union.__init__(self, cursor, parent)

    @property
    def union_index(self):
        return self._index(NestedUnion)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"union_{self.union_index}"
            else:
                return Union.name.fget(self)
        else:
            return self._name


class NestedEnum(Enum, Nested):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Enum.__init__(self, cursor, parent)

    @property
    def enum_index(self):
        return self._index(NestedEnum)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"enum_{self.enum_index}"
            else:
                return Enum.name.fget(self)
        else:
            return self._name


class Typedef(Type, Typed, *__TypedefMixins):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Type.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)


class FunctionPointer(Type):  # TODO handle result type
    def __init__(
        self,
        cursor: clang.cindex.Cursor,  # TYPEDEF_DECL
        parent: Node,
        result_type: clang.cindex.Type,
    ):
        Type.__init__(self, cursor, parent)
        self._result_type = result_type
        result_typeref = parent.get_root().lookup_type(
            result_type.get_canonical().spelling, result_type.spelling
        )
        self._canonical_result_typename = Typed.canonical_typename(
            cparser.TypeHandler(result_type),
            result_type.get_canonical().spelling,
            result_typeref.name if result_typeref is not None else None,
        )

    @property
    def canonical_result_typename(self):
        """
        Note:
            The canonical result type name must be named and cannot be anonymous.
            Hence, the `name` of the typeref can be used and it doesnt't make sense
            to introduce a `global_canonical_result_typename`.
        """
        return self._canonical_result_typename

    @property
    def parms(self):
        for child in self.child_nodes:
            if isinstance(child, Parm):
                yield child

    def parm_types(self, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.typename(renamer)

    def global_parm_types(self, sep=None, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.global_typename(sep, renamer)


class TypedefedFunctionPointer(FunctionPointer, *__TypedefedFunctionPointerMixins):
    @staticmethod
    def match(clang_type: clang.cindex.Type):
        return list(cparser.TypeHandler(clang_type).clang_type_layer_kinds()) == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.POINTER,
            clang.cindex.TypeKind.FUNCTIONPROTO,
        ]

    def __init__(self, cursor: clang.cindex.Cursor, parent: Node):  # TYPEDEF_DECL
        result_type = cursor.underlying_typedef_type.get_pointee().get_result()
        FunctionPointer.__init__(self, cursor, parent, result_type)


class AnonymousFunctionPointer(
    FunctionPointer, Nested, *__AnonymousFunctionPointerMixins
):
    @staticmethod
    def match(clang_type: clang.cindex.Type):
        return list(cparser.TypeHandler(clang_type).clang_type_layer_kinds()) == [
            clang.cindex.TypeKind.POINTER,
            clang.cindex.TypeKind.FUNCTIONPROTO,
        ]

    def __init__(
        self, cursor: clang.cindex.Cursor, parent: Node  # PARM_DECL, FIELD_DECL
    ):
        result_type = cursor.type.get_pointee().get_result()
        FunctionPointer.__init__(self, cursor, parent, result_type)

    @property
    def anon_funptr_index(self):
        return self._orig_index(AnonymousFunctionPointer)

    @property
    def name(self):
        return f"anon_funptr_{self.anon_funptr_index}"


class Parm(Node, Typed, *__ParmMixins):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Node.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)


class Function(Node, Typed, *__FunctionMixin):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
        typeref=None,  # result_type
    ):
        Node.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.result_type, typeref)

    @property
    def parms(self):
        for child in self.child_nodes:
            if isinstance(child, Parm):
                yield child

    def parm_names(self, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield renamer(parm.name)

    def global_parm_types(self, sep=None, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.global_typename(sep, renamer)

    def parm_types(self, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.typename(renamer)

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
                line = p_comment.sub("", line1, count=2)
                result.append(line)
        return "".join(result)


def from_libclang_translation_unit(
    translation_unit: clang.cindex.TranslationUnit, warnings=control.Warnings.WARN
) -> Root:
    """Create a tree from a libclang translation unit."""

    def first_child_cursors_of_kinds_(
        cursor: clang.cindex.Cursor, kinds: tuple[clang.cindex.CursorKind]
    ):
        """Returns the first typeref child or None."""
        return next(
            (
                child_cursor
                for child_cursor in cursor.get_children()
                if child_cursor.kind in kinds
            ),
            None,
        )

    structure_types = {
        clang.cindex.CursorKind.STRUCT_DECL: Struct,
        clang.cindex.CursorKind.UNION_DECL: Union,
        clang.cindex.CursorKind.ENUM_DECL: Enum,
    }
    nested_structure_types = {
        clang.cindex.CursorKind.STRUCT_DECL: NestedStruct,
        clang.cindex.CursorKind.UNION_DECL: NestedUnion,
        clang.cindex.CursorKind.ENUM_DECL: NestedEnum,
    }

    def handle_top_level_cursor_(cursor: clang.cindex.Cursor, root: Root):
        """Handle cursors whose parent is the cursor of kind TRANSLATION_UNIT."""
        nonlocal structure_types
        if cursor.kind in structure_types.keys():
            handle_nested_record_or_enum_cursor_(cursor, root)
        elif cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
            handle_typedef_cursor_(cursor, root)
        elif cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            if warnings in (control.Warnings.WARN, control.Warnings.ERROR):
                msg = (
                    f"VAR_DECL cursor '{cursor.spelling}' not handled (not implemented)"
                )
                if warnings == control.Warnings.WARN:
                    print(f"WARN: {msg}'", file=sys.stderr)
                else:
                    print(f"ERROR: {msg}'", file=sys.stderr)
                    sys.exit(2)
        elif cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
            root.append(MacroDefinition(cursor, root))
        elif cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            typeref_cursor = first_child_cursors_of_kinds_(
                cursor, (clang.cindex.CursorKind.TYPE_REF,)
            )
            typeref = root.lookup_type_from_cursor(typeref_cursor)
            node = Function(cursor, root, typeref=typeref)
            descend_into_child_cursors_(node)
            root.append(node)

    def handle_typedef_cursor_(cursor: clang.cindex.Cursor, root: Root):
        """Handle typedef cursors with respect to their children and type.

        Checks if the typedef has any STRUCT_DECL, UNION_DECL, ENUM_DECL, or TYPE_REF child cursor, which
        all indicate that there is already a node in the Root's child_nodes list for the inner type
        due to libclang's way of constructing the parse tree.

        In case of the former three, three different cases have to be handled:

        1. The inner type is anonymous.
            * In this case, a previously inserted anonymous Struct/-Union/-Enum node has to
            be replaced by Struct/Union/Enum node that uses a `ctypedef struct <name>`/...
            instead of `cdef struct <name>`/...  when rendering Cython code,
            where `<name>` is the spelling of the `TYPEDEF_DECL` cursor.
        2. Inner type and typedef name are the same.
            * In this case, no Typedef node is inserted as only `cdef struct <name>`/...  needs to be specified
              in the rendered Cython code.
        3. Inner type and typedef name differ.
            * In this case a Typedef case is inserted that specifies a previously added
                Struct/Union/Enum as typeref argument.

        In case none of the listed four child cursors could be found,
        the routine checks if the cursor's type might be a typedefed function pointer.
        In this case, no Typedef node but a `TypedefedFunctionPointer` is inserted.
        """
        node = Typedef(cursor, root)
        type_decl_cursor = first_child_cursors_of_kinds_(
            cursor,
            (
                clang.cindex.CursorKind.STRUCT_DECL,
                clang.cindex.CursorKind.UNION_DECL,
                clang.cindex.CursorKind.ENUM_DECL,
            ),
        )
        if type_decl_cursor is None:
            typeref_cursor = first_child_cursors_of_kinds_(
                cursor, (clang.cindex.CursorKind.TYPE_REF,)
            )
            node.typeref = root.lookup_type_from_cursor(typeref_cursor)
            if node.typeref == None:
                if TypedefedFunctionPointer.match(cursor.type):
                    node = TypedefedFunctionPointer(
                        cursor, root
                    )  # note that var `node`` is reassigned here
            descend_into_child_cursors_(node)  # post-order walk,
            root.append(node)
        elif not len(
            type_decl_cursor.spelling
        ):  # found anonymous struct/union/enum child
            # in case of anon enum
            # replace the original node with the given one
            anon_type_decl = root.lookup_type_from_cursor(type_decl_cursor)
            assert anon_type_decl != None and isinstance(anon_type_decl, (Enum, Record))
            type_decl = handle_anon_typedef_child_cursor_(type_decl_cursor, node)
            descend_into_child_cursors_(node)  # post-order walk
            root.insert(anon_type_decl.index, type_decl)
            root.remove(anon_type_decl)
            # do not append typedef node
        elif type_decl_cursor.spelling != cursor.spelling:  # child with different name
            # update, append typedef node
            node.typeref = root.lookup_type_from_cursor(type_decl_cursor)
            descend_into_child_cursors_(node)  # post-order walk
            root.append(node)
        else:  # child with same name
            descend_into_child_cursors_(node)  # post-order walk
            pass  # do not append typedef node

    def handle_anon_typedef_child_cursor_(cursor: clang.cindex.Cursor, parent: Typedef):
        """Handle a TYPEDEF_DECL cursors' anonymous STRUCT_DECL/UNION_DECL/ENUM_DECL child cursor."""
        nonlocal structure_types

        root = parent.get_root()
        parent_cursor = parent.cursor
        assert (
            cursor.type.get_canonical().spelling
            == parent_cursor.type.get_canonical().spelling
        )
        assert (
            cursor.spelling == ""
            and cursor.type.spelling == parent_cursor.type.spelling
        )
        if cursor.kind in structure_types:
            cls = structure_types[cursor.kind]
            node = cls(cursor, root, from_typedef_with_anon_child=True)
            node.overwrite_name(parent_cursor.spelling)
        else:
            raise RuntimeError(
                "expected cursor of kind 'STRUCT_DECL', 'UNION_DECL', or 'ENUM_DECL'"
            )

        return node

    def handle_nested_record_or_enum_cursor_(cursor: clang.cindex.Cursor, parent: Node):
        """Handle a STRUCT_DECL/UNION_DECL cursor's STRUCT_DECL/UNION_DECL/ENUM_DECL child cursor.
        Other cursors are ignored.
        """
        nonlocal structure_types
        nonlocal nested_structure_types

        is_anonymous = cursor.spelling == ""
        root = parent.get_root()

        if cursor.kind in structure_types:
            cls = structure_types[cursor.kind]
            cls_nested = nested_structure_types[cursor.kind]
            if is_anonymous:
                node = cls_nested(cursor, parent)
            else:
                node = cls(cursor, parent)
            descend_into_child_cursors_(node)
            parent.append(node)

    def handle_param_or_field_decl_cursor_(cursor: clang.cindex.Cursor, parent: Node):
        """Handle PARAM_DECL/FIELD_DECL cursors.

        First check if the cursor's type is anonymous function pointer.
        In this case emit an additional AnonymousFunctionPointer node.
        If there are further PARAM_DECL children of the given cursor, it visits
        them first before emitting an `AnonymousFunctionPointer` node.
        This guarantees that nested anoymous pointers are processed
        before constructing the `AnonymousFunctionPointer` node for the parent cursor.
        """
        assert cursor.kind in (
            clang.cindex.CursorKind.PARM_DECL,
            clang.cindex.CursorKind.FIELD_DECL,
        )
        if AnonymousFunctionPointer.match(cursor.type):
            root = parent.get_root()
            typeref = AnonymousFunctionPointer(cursor, parent)
            descend_into_child_cursors_(typeref)  # post-order walk
            root.append(typeref)
        else:
            typeref_cursor = first_child_cursors_of_kinds_(
                cursor,
                (
                    clang.cindex.CursorKind.TYPE_REF,
                    clang.cindex.CursorKind.STRUCT_DECL,
                    clang.cindex.CursorKind.UNION_DECL,
                    clang.cindex.CursorKind.ENUM_DECL,
                ),
            )
            # TODO check that typeref is only None if no type is involved
            root = parent.get_root()
            typeref = root.lookup_type_from_cursor(typeref_cursor)
        if cursor.kind == clang.cindex.CursorKind.PARM_DECL:
            node = Parm(cursor, parent, typeref=typeref)
        else:
            node = Field(cursor, parent, typeref=typeref)
        parent.append(node)

    def descend_(cursor, parent=None):
        assert isinstance(parent, Node)
        assert parent.cursor is not None
        parent_cursor = parent.cursor
        if parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            handle_top_level_cursor_(cursor, parent)
        elif parent_cursor.kind in (
            clang.cindex.CursorKind.STRUCT_DECL,
            clang.cindex.CursorKind.UNION_DECL,
        ):
            handle_nested_record_or_enum_cursor_(cursor, parent)
        #
        if cursor.kind in (
            clang.cindex.CursorKind.PARM_DECL,
            clang.cindex.CursorKind.FIELD_DECL,
        ):
            handle_param_or_field_decl_cursor_(cursor, parent)

    def descend_into_child_cursors_(node: Node):
        for child_cursor in node.cursor.get_children():
            descend_(child_cursor, node)

    root = Root(translation_unit.cursor)
    descend_into_child_cursors_(root)
    return root
