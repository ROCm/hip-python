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

import collections

import logging

_log = logging.getLogger("interfacegen")

import clang.cindex

from . import cparser

indent = " " * 4

# TODO dynamically create a tree module for backend in (cython, fortran)
# and make it available via __init__ package


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
        _log.debug(
            f"<{self.render_location()}>: NEW {self.__class__.__name__} from {self.cursor.kind} '{self.cursor.spelling}'"
        )

    @staticmethod
    def render_cursor_location(cursor):
        return f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}"

    def render_location(self):
        return self.render_cursor_location(self.cursor)

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
        assert isinstance(self, Node)
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

    @property
    def raw_comment(self):
        """Returns full (doxygen) comment for this node."""
        return self.cursor.raw_comment

    @property
    def brief_comment(self):
        """Returns brief (doxygen) comment for this node."""
        return self.cursor.brief_comment

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

    def has_record_for_type(self, node):
        canonical_typename = self._canonical_typename(node)
        return node in self.types.get(canonical_typename, [])

    def append_type(self, node):
        canonical_typename = self._canonical_typename(node)
        if not canonical_typename in self.types:
            self.types[canonical_typename] = []
        _log.debug(
            f" append_type: {type(node)} for canonical typename '{self._canonical_typename(node)}' from {node.cursor.kind} '{node.cursor.spelling}' ({node.render_location()})"
        )
        self.types[canonical_typename].append(node)

    def remove_type(self, node):
        """Removes a type from the registry.

        If the type is a parent of other types, removes those types too.

        Example 1:

        `typedef union {  } mytype;`

        will have clang produce an anoymous top-level union cursor plus a top-level
        typedef cursor with spelling `mytype`, even though there
        is no way to access the union cursor.

        For languages such as Cython, the union thus must be removed or renamed.
        This routine therefore allows to remove such types.

        `typedef union { struct { ... } field; } mytype;`

        will have clang produce an anoymous top-level union cursor plus a top-level
        typedef cursor with spelling `mytype`.
        Additionally clang

        Args:
            node (_type_)
        """
        canonical_typename = self._canonical_typename(node)
        if canonical_typename in self.types:
            assert node in self.types[canonical_typename]
            _log.debug(
                f" remove_type: {type(node)} for canonical typename '{self._canonical_typename(node)}' from {node.cursor.kind} '{node.cursor.spelling}' ({node.render_location()})"
            )
            self.types[canonical_typename].remove(node)


class MacroDefinition(Node):
    def __init__(self, cursor: clang.cindex.Cursor, parent: Node):
        Node.__init__(self, cursor, parent)


class Typed:
    def __init__(self, clang_type: clang.cindex.Type, typeref=None):
        self.typeref: Node = typeref
        self._clang_type: clang.cindex.Type = clang_type
        self.typehandler = cparser.TypeHandler(clang_type)

    @staticmethod
    def canonical_typename(
        typehandler: cparser.TypeHandler,
        searched_canonical_typename,
        repl_typename=None,
    ):
        """Returns a Cython-compatible typename for the given Clang type.

        If `repl_typename` is provided, replaces elaborated C type names, e.g. `struct Foo`,
        and anonymous types by `repl_typename`.
        Otherwise, simply returns the spelling of `clang_type.get_canonical()`.

        Args:
            repl_typename (str): A forced typename for the struct, union, or enum part of the
                                 canonical Clang typename.
        """

        # FIXME(interfacegen.cython.tree.canonical_typename,0,docharri) Revise method; may not be robust as "name" in "name_" would be regarded das match, better do regex search with word boundaries
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
                    searched_canonical_typename  # pointer with optional trailing modifiers
                ) or layer_canonical_type_spelling.endswith(
                    searched_canonical_typename  # other (canonical!) type with optional preceding modifiers
                ):
                    assert (
                        layer_canonical_type_spelling in canonical_type_to_modify
                    ), f"Types (searched typename, canonical type, canonical type of layer): '{searched_canonical_typename}', '{canonical_type_to_modify}', '{layer_canonical_type_spelling}'"
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

    def global_typename(
        self,
        sep: str,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
    ):
        """Returns a global typename based on types in the Root nodes type registry and
        a backend-specific renaming function provided by the user.

        Args:
            sep (str): A separator for connecting a nested types name with its parent name and it ancestors' name in order to
                       derive a global name.
            renamer (_type_, optional): A renaming function. Defaults to identity.
            use_canonical (bool, optional): If the canonical name should be preferred.

        Raises:
            ValueError: If the separator is not a string.

        Returns:
            str: A global typename for this type.
        """
        use_canonical = (
            prefer_canonical
            and self.is_innermost_canonical_type_layer_of_basic_type_or_void
        )
        if not isinstance(sep, str):
            raise ValueError("argument 'sep' must be a string.")
        if self.typeref is not None and not use_canonical:
            # print(f"[pre] {type(self.typeref)} <{self.typeref.render_location()}>")
            searched_typename = self.typeref.cursor.type.get_canonical().spelling
            repl_typename = renamer(self.typeref.global_name(sep))
            # print(f"[post] {type(self.typeref)} <{self.typeref.render_location()}>")
        else:
            searched_typename = None
            repl_typename = None
        return renamer(
            Typed.canonical_typename(self.typehandler, searched_typename, repl_typename)
        )

    def typename(
        self, renamer: callable = lambda name: name, prefer_canonical: bool = False
    ):
        use_canonical = (
            prefer_canonical
            and self.is_innermost_canonical_type_layer_of_basic_type_or_void
        )
        if self.typeref is not None and not use_canonical:
            searched_typename = self.typeref.cursor.type.get_canonical().spelling
            repl_typename = renamer(self.typeref.name)
        else:
            searched_typename = None
            repl_typename = None
        return renamer(
            Typed.canonical_typename(self.typehandler, searched_typename, repl_typename)
        )

    def clang_type_layer_kinds(self, postorder=False, canonical=False):
        return self.typehandler.clang_type_layer_kinds(
            postorder=postorder, canonical=canonical
        )

    def categorized_type_layer_kinds(
        self, postorder=False, consider_const=False, subdivide_basic_types: bool = False
    ):
        return self.typehandler.categorized_type_layer_kinds(
            postorder=postorder,
            consider_const=consider_const,
            subdivide_basic_types=subdivide_basic_types,
        )

    def const_qualifiers(self, postorder=False, canonical=False):
        """Yields a flag per type layer that constitute this type if
        the layer is const qualified.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Yields:
            bool: Per type layer, yields a flag indicating if ``const`` is specified for this layer.
        """
        return self.typehandler.const_qualifiers(
            postorder=postorder, canonical=canonical
        )

    @property
    def has_typeref(self):
        """If this type is referencing any other typedef, record, or enum.

        Returns:
            bool: If this type is referencing any other typedef, record, or enum.
        """
        return self.typeref != None

    def lookup_innermost_type(self):
        from . import tree

        curr = self
        while isinstance(curr, Typed) and curr.has_typeref:
            curr = curr.typeref
        return curr

    def get_pointer_degree(self, incomplete_array=False) -> int:
        """Returns number of outer type layers which are of TypeKind.POINTER.
        Args:
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.
        """
        return self.typehandler.get_pointer_degree(incomplete_array=incomplete_array)

    @property
    def is_void(self):
        """If this is a void type."""
        from clang.cindex import TypeKind

        return (
            next(self.typehandler.clang_type_layer_kinds(canonical=True))
            == TypeKind.VOID
        )

    def is_pointer_to_void(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a void pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.VOID, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_char(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a char pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.CHAR_S, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_basic_type(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a pointer to a basic datatype of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        TypeCategory = cparser.TypeHandler.TypeCategory

        return self.typehandler.is_pointer_to_category(
            TypeCategory.BASIC, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_constantarray_of_basic_type(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a pointer to a constant array of basic type of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        TypeCategory = cparser.TypeHandler.TypeCategory

        pointer_degree = self.get_pointer_degree(incomplete_array)
        if self.typehandler.compare_pointer_degree(pointer_degree, degree):
            (success, _) = self.typehandler.create_from_layer(
                pointer_degree, canonical=True
            ).is_constantarray_of_kind_or_category(type_category=TypeCategory.BASIC)
            return success
        return False

    def is_pointer_to_record(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a void pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.RECORD, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_enum(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a enum pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.ENUM, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_function_proto(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a void pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.FUNCTIONPROTO, degree, incomplete_array=incomplete_array
        )

    def is_pointer_to_function_proto(
        self,
        degree: int = 1,
        incomplete_array: bool = False,
    ):
        """If this is a void pointer of the given degree.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        from clang.cindex import TypeKind

        return self.typehandler.is_pointer_to_kind(
            TypeKind.FUNCTIONPROTO, degree, incomplete_array=incomplete_array
        )

    @property
    def is_any_pointer(self):
        """If this is any form of pointer, i.e. the outer most type layer must be a pointer."""
        from clang.cindex import TypeKind

        return (
            self.typehandler.get_rank(
                constant_array=False, incomplete_array=False, pointer=True
            )
            > 0
        )

    @property
    def is_any_array(self):
        """If this is any form of array."""
        TypeCategory = cparser.TypeHandler.TypeCategory
        return (
            next(self.typehandler.categorized_type_layer_kinds()) == TypeCategory.ARRAY
        )

    @property
    def is_record(self):
        """If this is a record (struct, union)."""
        from clang.cindex import TypeKind

        return (
            next(self.typehandler.clang_type_layer_kinds(canonical=True))
            == TypeKind.RECORD
        )

    @property
    def is_record_constantarray(self):
        """If this is a struct or union array."""
        # TODO multi-dim arrays
        from clang.cindex import TypeKind

        return list(self.typehandler.clang_type_layer_kinds(canonical=True)) == [
            TypeKind.CONSTANTARRAY,
            TypeKind.RECORD,
        ]

    @property
    def is_enum(self):
        """If this is an enum."""
        from clang.cindex import TypeKind

        return (
            next(self.typehandler.clang_type_layer_kinds(canonical=True))
            == TypeKind.ENUM
        )

    @property
    def is_enum_constantarray(self):
        """If this is an enum array."""
        # TODO multi-dim arrays
        from clang.cindex import TypeKind

        return list(self.typehandler.clang_type_layer_kinds(canonical=True)) == [
            TypeKind.CONSTANTARRAY,
            TypeKind.ENUM,
        ]

    @property
    def is_basic_type(self):
        """If this is a pointer to a struct or enum."""
        TypeCategory = cparser.TypeHandler.TypeCategory
        return list(self.typehandler.categorized_type_layer_kinds()) in [
            [TypeCategory.BASIC],
        ]

    def is_basic_type_constantarray(self, rank=-1):
        """If this is a constant array of a basic datatype.

        Args:
            rank (int, optional):
                Check for a specific rank by providing a positive value.
                Check for all ranks greater than or equal to``abs(rank)`` by providing
                a negative value Defaults to -1.
        """
        assert rank != 0, "Rank must not equal 0"
        (result, dims) = self.typehandler.is_constantarray_of_kind_or_category(
            type_category=cparser.TypeHandler.TypeCategory.BASIC
        )
        return result and (rank == dims or (rank < 0 and -dims <= rank))

    @property
    def is_char_incompletearray(self):
        """If this is an incomplete array of chars."""
        from clang.cindex import TypeKind

        return list(self.typehandler.clang_type_layer_kinds(canonical=True)) in [
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
        TypeCategory = self.typehandler.TypeCategory
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

    @property
    def is_innermost_canonical_type_layer_of_basic_type_or_void(self):
        """If the innermost type layer is of basic type or void type."""
        return self.typehandler.is_innermost_canonical_type_layer_of_basic_type_or_void


class Field(Node, Typed):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Node.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)


class Type(Node):
    """Indicates that this node represents a type.

    Note:
        'Type' is not the same as 'Typed'.
    """

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        self._name = None
        Node.__init__(self, cursor, parent)

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
        _log.debug(
            f"<{self.render_location()}> walk fields of {self.__class__.__name__} {self.global_name('_')},{self.cursor.kind=},{self.cursor.spelling=}"
        )
        for child in self.child_nodes:
            if isinstance(child, Field):
                assert child.cursor.kind == clang.cindex.CursorKind.FIELD_DECL
                yield child

    @property
    def is_incomplete(self):
        """If the type does not have any fields."""
        return next(self.fields, None) == None

    @property
    def is_opague(self):
        """Same as 'is_incomplete'."""
        return self.is_incomplete


class Struct(Record):
    def __init__(self, *args, **kwargs):
        Record.__init__(self, *args, **kwargs)


class Union(Record):
    def __init__(self, *args, **kwargs):
        Record.__init__(self, *args, **kwargs)


class Enum(Type):
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


class Anonymous:
    """A marker for nested struct/union/enum types."""

    pass


class AnonymousStruct(Struct, Anonymous):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Struct.__init__(self, cursor, parent)

    @property
    def struct_index(self):
        return self._index(AnonymousStruct)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"struct_{self.struct_index}"
            else:
                return Struct.name.fget(self)
        else:
            return self._name


class AnonymousUnion(Union, Anonymous):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Union.__init__(self, cursor, parent)

    @property
    def union_index(self):
        return self._index(AnonymousUnion)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"union_{self.union_index}"
            else:
                return Union.name.fget(self)
        else:
            return self._name


class AnonymousEnum(Enum, Anonymous):
    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent,
    ):
        Enum.__init__(self, cursor, parent)

    @property
    def enum_index(self):
        return self._index(AnonymousEnum)

    @property
    def name(self):
        if self._name == None:
            if self.is_cursor_anonymous:
                return f"enum_{self.enum_index}"
            else:
                return Enum.name.fget(self)
        else:
            return self._name


class Typedef(Type, Typed):
    @staticmethod
    def match_typedefed_enum(clang_type: clang.cindex.Type):
        """If the type is a typedef of an enum."""
        return list(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()) == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.ELABORATED,
            clang.cindex.TypeKind.ENUM,
        ]

    @staticmethod
    def match_typedefed_record(clang_type: clang.cindex.Type):
        """If the type is a typedef of an record (struct or union)."""
        return list(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()) == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.ELABORATED,
            clang.cindex.TypeKind.RECORD,
        ]

    @staticmethod
    def match_typedefed_record_or_enum(clang_type: clang.cindex.Type):
        """If the type is a typedef of an record (struct or union)."""
        return list(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds())[
            :2
        ] == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.ELABORATED,
        ]

    @staticmethod
    def match_typedefed_basic_type(clang_type: clang.cindex.Type):
        """If the type is a typedef of a basic type."""
        typehandler = cparser.TypeHandler.get(clang_type)
        return (
            next(typehandler.clang_type_layer_kinds()) == clang.cindex.TypeKind.TYPEDEF
            and next(typehandler.categorized_type_layer_kinds())
            == cparser.TypeHandler.TypeCategory.BASIC
        )

    @staticmethod
    def match_typedefed_void(clang_type: clang.cindex.Type):
        """If the type is a typedef of a basic type."""
        return (
            next(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds())
            == clang.cindex.TypeKind.TYPEDEF
            and next(cparser.TypeHandler.get(clang_type).categorized_type_layer_kinds())
            == cparser.TypeHandler.TypeCategory.VOID
        )

    @staticmethod
    def match_typedefed_pointer(clang_type: clang.cindex.Type):
        """If the type is a typedef of a pointer type of arbitrary degree."""
        return list(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds())[
            :2
        ] == [clang.cindex.TypeKind.TYPEDEF, clang.cindex.TypeKind.POINTER]

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Type.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)


class ConstantArray(Type, Typed):
    @staticmethod
    def match_typedefed_constantarray_of_basic_type(clang_type: clang.cindex.Type):
        """If the type is a typedef of a basic type."""
        typehandler = cparser.TypeHandler.get(clang_type)
        if next(typehandler.clang_type_layer_kinds()) == clang.cindex.TypeKind.TYPEDEF:
            (success, _) = typehandler.is_constantarray_of_kind_or_category(
                type_category=cparser.TypeHandler.TypeCategory.BASIC
            )
            return success
        return False

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Type.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)
        self.element_type, self.shape = self._get_element_type_and_shape()
        self.dim = len(self.shape)

    def _get_element_type_and_shape(self):
        """Returns element type and the array shape. Uses canonical type."""
        array_shape = []
        for layer_type in self.typehandler.walk_clang_type_layers(
            postorder=True, canonical=True
        ):
            if layer_type.kind == clang.cindex.TypeKind.CONSTANTARRAY:
                array_shape.append(layer_type.get_array_size())
            else:
                element_type = layer_type
        return (element_type, array_shape)


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

    def global_parm_types(
        self,
        sep=None,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
    ):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.global_typename(sep, renamer, prefer_canonical)


class TypedefedFunctionPointer(FunctionPointer):
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


class AnonymousFunctionPointer(FunctionPointer, Anonymous):
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
        return self._index(AnonymousFunctionPointer)

    @property
    def name(self):
        return f"anon_funptr_{self.anon_funptr_index}"


class Parm(Node, Typed):
    unnamed_parm_template = "arg{parm_index}"

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,
        typeref=None,
    ):
        Node.__init__(self, cursor, parent)
        Typed.__init__(self, self.cursor.type, typeref)

    @property
    def parm_index(self):
        """Index of the parameter in the argument list."""
        assert self.parent != None
        return self._index(cls=Parm)

    @property
    def name(self):
        """Returns a generic name in case the parameter
        has not been given a name.
        """
        given_name = Node.name.fget(self)
        if not len(given_name):
            return Parm.unnamed_parm_template.format(parm_index=self.parm_index)
        return given_name


class Function(Node, Typed):
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

    def get_parm(self, index: int):
        """Return the parameter at the given index."""
        if index >= 0:
            cur = 0
            for parm in self.parms:
                if cur == index:
                    return parm
                cur += 1
        raise IndexError(f"Index {index} is out of bounds.")

    def parm_names(self, renamer: callable = lambda name: name):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield renamer(parm.name)

    def global_parm_types(
        self,
        sep=None,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
    ):
        for parm in self.parms:
            assert isinstance(parm, Parm)
            yield parm.global_typename(sep, renamer, prefer_canonical)