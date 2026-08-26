# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

import clang.cindex

from . import cparser

_log = logging.getLogger("interfacegen")

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
        if sep is None:
            return name_parts
        else:
            return f"{sep.join(reversed(name_parts))}"

    @property
    def is_cursor_anonymous(self):
        """If the cursor is anonymous.

        Note:
            Always use the raw cursor as 'name' might be overwritten.

            libclang exposes both ``cursor.spelling`` and
            ``cursor.is_anonymous()``. For some anonymous types
            (e.g. an unnamed nested ``struct {…};`` member of a typedef'd
            union) libclang fills ``spelling`` with a synthetic
            ``"struct (anonymous at /path:N:M)"`` placeholder rather than an
            empty string. Only ``is_anonymous()`` is reliable in that case;
            the empty-spelling check still catches older shapes.

            From libclang 17 onward, the inner type of a
            ``typedef struct/union/enum {...} foo_t;`` no longer
            reports ``cursor.spelling == ""`` — the typedef name is
            inlined into the inner cursor's spelling. Detect that
            case via ``cursor.type.spelling``: a tagged inner type
            always carries its kind keyword (``enum foo``,
            ``struct foo``, ``union foo``); a typedef'd anonymous
            inner type carries just the typedef name.
        """
        assert isinstance(self, Node)
        if hasattr(self.cursor, "is_anonymous"):
            try:
                if self.cursor.is_anonymous():
                    return True
            except Exception:
                pass
        if len(self.cursor.spelling) == 0:
            return True
        # libclang 17+ shim: detect anonymous-typedef inner types whose
        # spelling now carries the typedef name.
        import clang.cindex

        kind = getattr(self.cursor, "kind", None)
        prefix = {
            clang.cindex.CursorKind.ENUM_DECL: "enum ",
            clang.cindex.CursorKind.STRUCT_DECL: "struct ",
            clang.cindex.CursorKind.UNION_DECL: "union ",
        }.get(kind)
        if prefix is not None:
            return not self.cursor.type.spelling.startswith(prefix)
        return False

    @property
    def file(self):
        """Returns the filename, or None for macro definitions."""
        if self.cursor.location.file is not None:
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
        while curr.parent is not None:
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
        assert self.parent is not None
        if cls is None:
            cls = Node
        num = 0
        for child in self.parent.child_nodes:
            if isinstance(child, cls):
                if child == self:
                    return num
                num += 1
        raise RuntimeError(
            "Node must be present in parent's `child_nodes` list"
        )

    @property
    def index(self):
        assert self.parent is not None
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
        if canonical_typename not in self.types:
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

    def render_type(
        self,
        sep: str,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
        local_name_only: bool = False,
    ):
        """Return a structured ``RenderedType`` for this typed entity.

        Walks the Clang type layer hierarchy through ``TypeHandler``
        instead of tokenizing canonical spellings. See
        ``interfacegen.typerender.render`` for substitution semantics
        (typeref-based ``struct Foo``/``enum Bar``/``union Baz`` rewrite,
        per-pointer qualifier preservation, incomplete-array → pointer
        decay).

        The typedef maps the backend attached to this node (if any) travel
        with it rather than through every caller's argument list, the way
        ``renamer`` does — every renderer call in the Cython backend wants the
        same module-wide maps.
        """
        from . import typerender

        return typerender.render(
            self,
            sep,
            renamer=renamer,
            prefer_canonical=prefer_canonical,
            local_name_only=local_name_only,
            typedef_aliases=getattr(self, "typedef_aliases", None),
            typedef_specs=getattr(self, "typedef_specs", None),
        )

    def fixed_width_typedef(self):
        """The ``(spelling, (signed, bits))`` this type's leaf pins, or None.

        The one place to ask what a declaration promises about width, rather
        than what clang canonicalized it to on the host the generator ran on.

        It lives here rather than on ``cparser.TypeHandler`` because the latter
        is built from a clang type alone and cannot see the per-module typedef
        maps — an accessor there would silently miss a library typedef such as
        ``hoff_t``, and ``cparser`` should stay a faithful mirror of libclang.
        """
        from . import typerender

        return typerender.fixed_width_typedef(
            self.typehandler.clang_type,
            getattr(self, "typedef_aliases", None),
            getattr(self, "typedef_specs", None),
        )

    def global_typename(
        self,
        sep: str,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
    ):
        """Return the Cython type spelling with elaborated tags substituted.

        Thin wrapper around :pymeth:`render_type` for callers that want a
        finished string. Uses the typeref's ``global_name(sep)`` as the
        substitution identifier.
        """
        if not isinstance(sep, str):
            raise ValueError("argument 'sep' must be a string.")
        return self.render_type(
            sep, renamer=renamer, prefer_canonical=prefer_canonical
        ).global_decl()

    def typename(
        self,
        renamer: callable = lambda name: name,
        prefer_canonical: bool = False,
    ):
        """Like :pymeth:`global_typename` but uses the typeref's local
        ``name`` (not its global path) as the substitution identifier."""
        return self.render_type(
            sep="_",
            renamer=renamer,
            prefer_canonical=prefer_canonical,
            local_name_only=True,
        ).global_decl()

    def clang_type_layer_kinds(self, postorder=False, canonical=False):
        return self.typehandler.clang_type_layer_kinds(
            postorder=postorder, canonical=canonical
        )

    def categorized_type_layer_kinds(
        self,
        postorder=False,
        consider_const=False,
        subdivide_basic_types: bool = False,
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
    def is_const_qualified(self) -> bool:
        """If the type of this typed entity is const qualified."""
        return self.typehandler.clang_type.is_const_qualified()

    @property
    def has_typeref(self):
        """If the type of this typed entity is referencing any other typedef,
        record, or enum.

        Returns:
            bool: If this type is referencing any other typedef, record, or enum.
        """
        return self.typeref is not None

    def lookup_innermost_type(self):
        curr = self
        while isinstance(curr, Typed) and curr.has_typeref:
            curr = curr.typeref
        return curr

    def get_pointer_degree(self, incomplete_array=False) -> int:
        """Returns number of outer type layers which are of TypeKind.POINTER.
        Args:
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.
        """
        return self.typehandler.get_pointer_degree(
            incomplete_array=incomplete_array
        )

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
            ).is_constantarray_of_kind_or_category(
                type_category=TypeCategory.BASIC
            )
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

    @property
    def is_any_pointer(self):
        """If this is any form of pointer, i.e. the outer most type layer must be a pointer."""
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
            next(self.typehandler.categorized_type_layer_kinds())
            == TypeCategory.ARRAY
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

        return list(
            self.typehandler.clang_type_layer_kinds(canonical=True)
        ) == [
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

        return list(
            self.typehandler.clang_type_layer_kinds(canonical=True)
        ) == [
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

    def char_constantarray_extent(self):
        """The declared extent of a rank-1 constant array of chars.

        Returns:
            int|None:
                The number of elements, or None if this is not a
                one-dimensional constant array with a char element type.

        Note:
            Backends must bound the read of such a field at this extent.
            A ``char[N]`` decays to ``char*`` in most languages, so a
            terminator-based conversion reads past the field whenever the
            data fills the array without a NUL.

        Note:
            All char flavours count, signed and unsigned alike: the C type
            says nothing about whether the payload is text or opaque bytes,
            and both need the same bound.
        """
        from clang.cindex import TypeKind

        (result, dims) = self.typehandler.is_constantarray_of_kind_or_category(
            type_kind=(
                TypeKind.CHAR_S,
                TypeKind.CHAR_U,
                TypeKind.SCHAR,
                TypeKind.UCHAR,
            )
        )
        if not result or dims != 1:
            return None
        for layer_type in self.typehandler.walk_clang_type_layers(
            postorder=True, canonical=True
        ):
            if layer_type.kind == TypeKind.CONSTANTARRAY:
                return layer_type.get_array_size()
        return None

    @property
    def is_char_incompletearray(self):
        """If this is an incomplete array of chars."""
        from clang.cindex import TypeKind

        return list(
            self.typehandler.clang_type_layer_kinds(canonical=True)
        ) in [
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
        return (
            self.typehandler.is_innermost_canonical_type_layer_of_basic_type_or_void
        )

    @property
    def has_innermost_type_layer_const_modifier(self):
        """If the innermost type layer has a const modifier."""
        return next(
            self.typehandler.const_qualifiers(postorder=True, canonical=True)
        )


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
        return self.is_cursor_anonymous and self._name is None

    @property
    def name(self):
        if self._name is None:
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
        return next(self.fields, None) is None

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
        if self._name is None:
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
        if self._name is None:
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
        if self._name is None:
            if self.is_cursor_anonymous:
                return f"enum_{self.enum_index}"
            else:
                return Enum.name.fget(self)
        else:
            return self._name


class Typedef(Type, Typed):
    @staticmethod
    def match_typedefed_enum(clang_type: clang.cindex.Type):
        """If the type is a typedef of an enum.

        libclang ≤16 reports the layer chain as
        ``[TYPEDEF, ELABORATED, ENUM]``; libclang ≥17 collapses the
        ``ELABORATED`` wrapper for typedefs that resolve to a record/enum
        and reports ``[TYPEDEF, ENUM]`` directly. Accept both shapes.
        """
        layers = list(
            cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()
        )
        if not layers or layers[0] != clang.cindex.TypeKind.TYPEDEF:
            return False
        return layers[1:] in (
            [clang.cindex.TypeKind.ELABORATED, clang.cindex.TypeKind.ENUM],
            [clang.cindex.TypeKind.ENUM],
        )

    @staticmethod
    def match_typedefed_record(clang_type: clang.cindex.Type):
        """If the type is a typedef of an record (struct or union).

        Accepts both the libclang ≤16 ``[TYPEDEF, ELABORATED, RECORD]``
        shape and the libclang ≥17 ``[TYPEDEF, RECORD]`` shape.
        """
        layers = list(
            cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()
        )
        if not layers or layers[0] != clang.cindex.TypeKind.TYPEDEF:
            return False
        return layers[1:] in (
            [clang.cindex.TypeKind.ELABORATED, clang.cindex.TypeKind.RECORD],
            [clang.cindex.TypeKind.RECORD],
        )

    @staticmethod
    def match_typedefed_record_or_enum(clang_type: clang.cindex.Type):
        """If the type is a typedef of a record (struct or union) or enum.

        Accepts both the libclang ≤16 ``[TYPEDEF, ELABORATED, …]`` shape
        and the libclang ≥17 collapsed ``[TYPEDEF, RECORD]`` /
        ``[TYPEDEF, ENUM]`` shapes — the second layer can be any of
        ELABORATED, RECORD, or ENUM.
        """
        layers = list(
            cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()
        )[:2]
        if len(layers) < 2 or layers[0] != clang.cindex.TypeKind.TYPEDEF:
            return False
        return layers[1] in (
            clang.cindex.TypeKind.ELABORATED,  # libclang ≤16
            clang.cindex.TypeKind.RECORD,  # libclang ≥17 (struct/union)
            clang.cindex.TypeKind.ENUM,  # libclang ≥17
        )

    @staticmethod
    def match_typedefed_basic_type(clang_type: clang.cindex.Type):
        """If the type is a typedef of a basic type."""
        typehandler = cparser.TypeHandler.get(clang_type)
        return (
            next(typehandler.clang_type_layer_kinds())
            == clang.cindex.TypeKind.TYPEDEF
            and next(typehandler.categorized_type_layer_kinds())
            == cparser.TypeHandler.TypeCategory.BASIC
        )

    @staticmethod
    def match_typedefed_void(clang_type: clang.cindex.Type):
        """If the type is a typedef of a basic type."""
        return (
            next(cparser.TypeHandler.get(clang_type).clang_type_layer_kinds())
            == clang.cindex.TypeKind.TYPEDEF
            and next(
                cparser.TypeHandler.get(
                    clang_type
                ).categorized_type_layer_kinds()
            )
            == cparser.TypeHandler.TypeCategory.VOID
        )

    @staticmethod
    def match_typedefed_pointer(clang_type: clang.cindex.Type):
        """If the type is a typedef of a pointer type of arbitrary degree."""
        return list(
            cparser.TypeHandler.get(clang_type).clang_type_layer_kinds()
        )[:2] == [clang.cindex.TypeKind.TYPEDEF, clang.cindex.TypeKind.POINTER]

    @staticmethod
    def match_typedefed_typedef(clang_type: clang.cindex.Type):
        """If the type is a typedef whose immediate underlying is itself
        a typedef.

        Catches the chain ``typedef A B;`` where ``A`` is itself
        ``typedef X A;`` for some upstream ``X``. Layer shape is
        ``[TYPEDEF, TYPEDEF, ...]``.

        libclang 17+ interposes an ``ELABORATED`` layer when the
        immediate underlying type is a *named* typedef, so ``B``'s
        canonical layer walk reports ``[TYPEDEF, ELABORATED, TYPEDEF,
        ...]`` rather than the older ``[TYPEDEF, TYPEDEF, ...]``. Skip
        those passthrough ``ELABORATED`` layers before inspecting the
        first two kinds so the chain still matches across libclang
        versions (same approach as
        ``generic.opaque_typedef_is_handle._is_typedef_to_pointer``).

        Note this matcher MUST be checked BEFORE
        ``match_typedefed_basic_type`` /
        ``match_typedefed_record_or_enum`` etc. in
        ``treefactory.handle_typedef_cursor_``: those matchers
        classify by the *resolved* leaf category, so a typedef chain
        bottoming out in (say) ``int`` would also match
        ``match_typedefed_basic_type`` and skip the chain handling.
        Routing chains through this matcher keeps the typeref link
        to the underlying typedef intact, so the renderer
        substitutes the alias name (e.g. ``hsa_ext_module_t``)
        instead of the canonical spelling (e.g.
        ``struct BrigModuleHeader *`` — which Cython rejects).

        HSA's ``typedef BrigModule_t hsa_ext_module_t;`` (declared in
        ``hsa_ext_finalize.h`` after
        ``typedef struct BrigModuleHeader* BrigModule_t;``) is the
        motivating case.
        """
        kinds = [
            k
            for k in cparser.TypeHandler.get(
                clang_type
            ).clang_type_layer_kinds()
            if k != clang.cindex.TypeKind.ELABORATED
        ]
        return kinds[:2] == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.TYPEDEF,
        ]

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
    def match_typedefed_constantarray_of_basic_type(
        clang_type: clang.cindex.Type,
    ):
        """If the type is a typedef of a basic type."""
        typehandler = cparser.TypeHandler.get(clang_type)
        if (
            next(typehandler.clang_type_layer_kinds())
            == clang.cindex.TypeKind.TYPEDEF
        ):
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
        from . import typerender

        self._canonical_result_typename = typerender.render_clang_type(
            result_type,
            typeref=result_typeref,
            local_name_only=True,
        ).global_decl()

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
        return list(
            cparser.TypeHandler(clang_type).clang_type_layer_kinds()
        ) == [
            clang.cindex.TypeKind.TYPEDEF,
            clang.cindex.TypeKind.POINTER,
            clang.cindex.TypeKind.FUNCTIONPROTO,
        ]

    def __init__(
        self, cursor: clang.cindex.Cursor, parent: Node
    ):  # TYPEDEF_DECL
        result_type = cursor.underlying_typedef_type.get_pointee().get_result()
        FunctionPointer.__init__(self, cursor, parent, result_type)


class AnonymousFunctionPointer(FunctionPointer, Anonymous):
    @staticmethod
    def match(clang_type: clang.cindex.Type):
        return list(
            cparser.TypeHandler(clang_type).clang_type_layer_kinds()
        ) == [
            clang.cindex.TypeKind.POINTER,
            clang.cindex.TypeKind.FUNCTIONPROTO,
        ]

    def __init__(
        self,
        cursor: clang.cindex.Cursor,
        parent: Node,  # PARM_DECL, FIELD_DECL
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
        assert self.parent is not None
        return self._index(cls=Parm)

    @property
    def name(self):
        """Returns a generic name in case the parameter
        has not been given a name.
        """
        given_name = Node.name.fget(self)
        if not len(given_name):
            return Parm.unnamed_parm_template.format(
                parm_index=self.parm_index
            )
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
