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

import enum

import clang.cindex


class TypeHandler:

    _INSTANCE = None

    @classmethod
    def get(cls, clang_type: clang.cindex.Type):
        if cls._INSTANCE is None:
            cls._INSTANCE = TypeHandler(None)
        cls._INSTANCE.clang_type = clang_type
        return cls._INSTANCE

    class TypeCategory(enum.IntEnum):
        INVALID = -2
        UNCATEGORIZED = -1
        VOID = 0
        BASIC = 10  # basic datatype
        BOOL = BASIC + 1
        CHAR = BASIC + 2
        INT = BASIC + 3
        FLOAT = BASIC + 4
        RECORD = 30
        ENUM = RECORD + 1
        POINTER = RECORD + 2
        ARRAY = RECORD + 3
        COMPLEX = RECORD + 4
        FUNCTION = RECORD + 5
        # const variants
        CONST_VOID = VOID + 100
        CONST_BASIC = BASIC + 100  # basic datatype
        CONST_BOOL = BOOL + 100
        CONST_CHAR = CHAR + 100
        CONST_INT = INT + 100
        CONST_FLOAT = FLOAT + 100
        CONST_RECORD = RECORD + 100
        CONST_ENUM = ENUM + 100
        CONST_POINTER = POINTER + 100
        CONST_ARRAY = ARRAY + 100
        CONST_COMPLEX = COMPLEX + 100
        CONST_FUNCTION = FUNCTION + 100

        @property
        def is_const(self):
            return self.value >= TypeHandler.TypeCategory.CONST_VOID.value

        @property
        def is_basic(self):
            value = self.value
            if self.is_const:
                value -= TypeHandler.TypeCategory.CONST_VOID.value
            return (
                value >= TypeHandler.TypeCategory.BASIC.value
                and value < TypeHandler.TypeCategory.RECORD.value
            )

    @staticmethod
    def get_type_kind(obj):
        if isinstance(obj, clang.cindex.TypeKind):
            return obj
        elif isinstance(obj, clang.cindex.Type):
            return obj.kind
        elif isinstance(obj, clang.cindex.Cursor):
            return obj.type.kind
        else:
            raise RuntimeError("expected obj of type ")

    @staticmethod
    def match_invalid_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.INVALID

    @staticmethod
    def match_void_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.VOID

    @staticmethod
    def match_bool_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.BOOL

    @staticmethod
    def match_char_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.CHAR_U,
            clang.cindex.TypeKind.UCHAR,
            clang.cindex.TypeKind.CHAR16,
            clang.cindex.TypeKind.CHAR32,
            clang.cindex.TypeKind.CHAR_S,
            clang.cindex.TypeKind.SCHAR,
            clang.cindex.TypeKind.WCHAR,
        )

    @staticmethod
    def match_char8_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.CHAR_U,
            clang.cindex.TypeKind.UCHAR,
            clang.cindex.TypeKind.CHAR_S,
            clang.cindex.TypeKind.SCHAR,
        )

    @staticmethod
    def match_int_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.USHORT,
            clang.cindex.TypeKind.UINT,
            clang.cindex.TypeKind.ULONG,
            clang.cindex.TypeKind.ULONGLONG,
            clang.cindex.TypeKind.UINT128,
            clang.cindex.TypeKind.SHORT,
            clang.cindex.TypeKind.INT,
            clang.cindex.TypeKind.LONG,
            clang.cindex.TypeKind.LONGLONG,
            clang.cindex.TypeKind.INT128,
        )

    @staticmethod
    def match_float_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.FLOAT,
            clang.cindex.TypeKind.DOUBLE,
            clang.cindex.TypeKind.LONGDOUBLE,
            clang.cindex.TypeKind.FLOAT128,
            clang.cindex.TypeKind.HALF,
            clang.cindex.TypeKind.IBM128,
        )

    @staticmethod
    def match_basic_datatype(type_kind: clang.cindex.TypeKind):
        return (
            TypeHandler.match_bool_type(type_kind)
            or TypeHandler.match_char_type(type_kind)
            or TypeHandler.match_int_type(type_kind)
            or TypeHandler.match_float_type(type_kind)
        )

    @staticmethod
    def match_complex_type(type_kind: clang.cindex.TypeKind):
        return clang.cindex.TypeKind == clang.cindex.TypeKind.COMPLEX

    @staticmethod
    def match_other_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.NULLPTR,
            clang.cindex.TypeKind.OVERLOAD,
            clang.cindex.TypeKind.DEPENDENT,
            clang.cindex.TypeKind.OBJCID,
            clang.cindex.TypeKind.OBJCCLASS,
            clang.cindex.TypeKind.OBJCSEL,
            clang.cindex.TypeKind.LVALUEREFERENCE,
            clang.cindex.TypeKind.RVALUEREFERENCE,
            clang.cindex.TypeKind.OBJCINTERFACE,
            clang.cindex.TypeKind.OBJCOBJECTPOINTER,
            clang.cindex.TypeKind.AUTO,
            clang.cindex.TypeKind.PIPE,
            clang.cindex.TypeKind.OCLIMAGE1DRO,
            clang.cindex.TypeKind.OCLIMAGE1DARRAYRO,
            clang.cindex.TypeKind.OCLIMAGE1DBUFFERRO,
            clang.cindex.TypeKind.OCLIMAGE2DRO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYRO,
            clang.cindex.TypeKind.OCLIMAGE2DDEPTHRO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYDEPTHRO,
            clang.cindex.TypeKind.OCLIMAGE2DMSAARO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAARO,
            clang.cindex.TypeKind.OCLIMAGE2DMSAADEPTHRO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAADEPTHRO,
            clang.cindex.TypeKind.OCLIMAGE3DRO,
            clang.cindex.TypeKind.OCLIMAGE1DWO,
            clang.cindex.TypeKind.OCLIMAGE1DARRAYWO,
            clang.cindex.TypeKind.OCLIMAGE1DBUFFERWO,
            clang.cindex.TypeKind.OCLIMAGE2DWO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYWO,
            clang.cindex.TypeKind.OCLIMAGE2DDEPTHWO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYDEPTHWO,
            clang.cindex.TypeKind.OCLIMAGE2DMSAAWO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAAWO,
            clang.cindex.TypeKind.OCLIMAGE2DMSAADEPTHWO,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAADEPTHWO,
            clang.cindex.TypeKind.OCLIMAGE3DWO,
            clang.cindex.TypeKind.OCLIMAGE1DRW,
            clang.cindex.TypeKind.OCLIMAGE1DARRAYRW,
            clang.cindex.TypeKind.OCLIMAGE1DBUFFERRW,
            clang.cindex.TypeKind.OCLIMAGE2DRW,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYRW,
            clang.cindex.TypeKind.OCLIMAGE2DDEPTHRW,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYDEPTHRW,
            clang.cindex.TypeKind.OCLIMAGE2DMSAARW,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAARW,
            clang.cindex.TypeKind.OCLIMAGE2DMSAADEPTHRW,
            clang.cindex.TypeKind.OCLIMAGE2DARRAYMSAADEPTHRW,
            clang.cindex.TypeKind.OCLIMAGE3DRW,
            clang.cindex.TypeKind.OCLSAMPLER,
            clang.cindex.TypeKind.OCLEVENT,
            clang.cindex.TypeKind.OCLQUEUE,
            clang.cindex.TypeKind.OCLRESERVEID,
            clang.cindex.TypeKind.EXTVECTOR,
            clang.cindex.TypeKind.ATOMIC,
        )

    @staticmethod
    def match_pointer_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.POINTER,  # ATT
            clang.cindex.TypeKind.BLOCKPOINTER,  # ATT
            clang.cindex.TypeKind.MEMBERPOINTER,  # ATT
        )

    @staticmethod
    def match_function_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.FUNCTIONNOPROTO,  # ATT
            clang.cindex.TypeKind.FUNCTIONPROTO,  # ATT
        )

    @staticmethod
    def match_arraylike_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.VECTOR,  # ATT
            clang.cindex.TypeKind.VARIABLEARRAY,  # ATT
            clang.cindex.TypeKind.DEPENDENTSIZEDARRAY,  # ATT
            clang.cindex.TypeKind.CONSTANTARRAY,  # ATT
            clang.cindex.TypeKind.INCOMPLETEARRAY,  # ATT
        )

    @staticmethod
    def match_record_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.RECORD

    @staticmethod
    def match_enum_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.ENUM

    @staticmethod
    def match_record_or_enum_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.RECORD,  # ATT
            clang.cindex.TypeKind.ENUM,  # ATT
        )

    @staticmethod
    def match_elaborated_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.ELABORATED  # ATT

    @staticmethod
    def match_typedef_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.TYPEDEF

    @staticmethod
    def categorize_clang_type_kind(
        type_kind: clang.cindex.TypeKind,
        is_const: bool = False,
        subdivide_basic_types: bool = False,
    ):
        """
        is_const (bool): If the type is const qualified, a special type category is returned.
                                   If you do no want this behaviour, just pass False. Defaults to False.
        subdivide_basic_types (bool,optional): If basic datatypes should be further categorized into
                                               the categories: bool, char, int, float. Defaults to false
        """
        if TypeHandler.match_invalid_type(type_kind):
            result = TypeHandler.TypeCategory.INVALID
        elif TypeHandler.match_void_type(type_kind):
            result = TypeHandler.TypeCategory.VOID
        elif not subdivide_basic_types and TypeHandler.match_basic_datatype(
            type_kind
        ):
            result = TypeHandler.TypeCategory.BASIC
        elif TypeHandler.match_bool_type(type_kind):
            result = TypeHandler.TypeCategory.BOOL
        elif TypeHandler.match_char_type(type_kind):
            result = TypeHandler.TypeCategory.CHAR
        elif TypeHandler.match_int_type(type_kind):
            result = TypeHandler.TypeCategory.INT
        elif TypeHandler.match_float_type(type_kind):
            result = TypeHandler.TypeCategory.FLOAT
        elif TypeHandler.match_record_type(type_kind):
            result = TypeHandler.TypeCategory.RECORD
        elif TypeHandler.match_enum_type(type_kind):
            result = TypeHandler.TypeCategory.ENUM
        elif TypeHandler.match_pointer_type(type_kind):
            result = TypeHandler.TypeCategory.POINTER
        elif TypeHandler.match_arraylike_type(type_kind):
            result = TypeHandler.TypeCategory.ARRAY
        elif TypeHandler.match_complex_type(type_kind):
            result = TypeHandler.TypeCategory.COMPLEX
        elif TypeHandler.match_function_type(type_kind):
            result = TypeHandler.TypeCategory.FUNCTION
        else:
            raise ValueError(
                f"type kind '{type_kind}' could not be Categorized"
            )
        if is_const:
            return TypeHandler.TypeCategory(
                result.value + TypeHandler.TypeCategory.CONST_VOID.value
            )
        else:
            return result

    def __init__(self, clang_type: clang.cindex.Type):
        self.clang_type = clang_type

    # flake8: noqa: C901
    # TODO break function apart to reduce complexity
    def walk_clang_type_layers(self, postorder=False, canonical=False):
        """Walks through the constitutents of a Clang type.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Note:
            Note that this is by default a pre-order walk, e.g., if we have a type `void *`,
            we will obtain first the pointer type and then the `void` type.
        """

        def descend_(clang_type: clang.cindex.TypeKind):
            nonlocal postorder
            type_kind = clang_type.kind
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
                underlying_type = (
                    clang_type.get_declaration().underlying_typedef_type
                )
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
                raise RuntimeError(
                    f"handling types of kind '{type_kind.spelling}' not implemented"
                )
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

    def create_from_layer(self, layer: int, canonical: bool = False):
        """Create a new TypeHandler instance from the specified layer.

        Args:
            layer (int): If you specify '-1', you get the last layer.
            canonical (bool, optional): Use the canonical type for the walk.

        Raises:
            ValueError: If the specified layer is smaller '-1',

        Returns:
            _type_: _description_
        """
        layers = list(self.walk_clang_type_layers(canonical=canonical))
        if layer >= len(layers):
            raise ValueError(
                "argument 'layer' was chosen larger than the number of available layers"
            )
        elif layer < -1:
            raise ValueError(
                "argument 'layer' must be chosen greater than '-1'"
            )
        return TypeHandler(layers[layer])

    def const_qualifiers(self, postorder=False, canonical=False):
        """Yields a flag per type layer that constitute this type if
        the layer is const qualified.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Yields:
            bool: Per type layer, yields a flag indicating if ``const`` is specified for this layer.
        """
        for clang_type in self.walk_clang_type_layers(postorder, canonical):
            yield clang_type.is_const_qualified()

    def categorized_type_layer_kinds(
        self,
        postorder=False,
        consider_const=False,
        subdivide_basic_types: bool = False,
    ):
        """Yields the Clang type kinds that constitute this type.
        Always uses the canonical type.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            consider_const (bool,optional): If the categories should explicitly consider const-qualified types. Defaults to False.
            subdivide_basic_types (bool,optional): If basic datatypes should be further categorized into
                                                   the categories: bool, char, int, float. Defaults to False.

        Note:
            Note that this is by default a pre-order walk, e.g., if we have a type `void *`,
            we will obtain first the pointer type and then the `void` type.
        """
        for clang_type in self.walk_clang_type_layers(
            postorder, canonical=True
        ):
            yield TypeHandler.categorize_clang_type_kind(
                clang_type.kind,
                is_const=consider_const and clang_type.is_const_qualified(),
                subdivide_basic_types=subdivide_basic_types,
            )

    @property
    def is_invalid_type(self):
        return TypeHandler.match_invalid_type(self.clang_type.kind)

    @property
    def is_void_type(self):
        return TypeHandler.match_void_type(self.clang_type.kind)

    @property
    def is_bool_type(self):
        return TypeHandler.match_bool_type(self.clang_type.kind)

    @property
    def is_char_type(self):
        return TypeHandler.match_char_type(self.clang_type.kind)

    @property
    def is_char8_type(self):
        return TypeHandler.match_char8_type(self.clang_type.kind)

    @property
    def is_int_type(self):
        return TypeHandler.match_int_type(self.clang_type.kind)

    @property
    def is_float_type(self):
        return TypeHandler.match_float_type(self.clang_type.kind)

    @property
    def is_basic_datatype(self):
        return TypeHandler.match_basic_datatype(self.clang_type.kind)

    @property
    def is_complex_type(self):
        return TypeHandler.match_complex_type(self.clang_type.kind)

    @property
    def is_other_type(self):
        return TypeHandler.match_other_type(self.clang_type.kind)

    @property
    def is_pointer_type(self):
        return TypeHandler.match_pointer_type(self.clang_type.kind)

    @property
    def is_function_type(self):
        return TypeHandler.match_function_type(self.clang_type.kind)

    @property
    def is_arraylike_type(self):
        return TypeHandler.match_arraylike_type(self.clang_type.kind)

    @property
    def is_record_type(self):
        return TypeHandler.match_record_type(self.clang_type.kind)

    @property
    def is_enum_type(self):
        return TypeHandler.match_enum_type(self.clang_type.kind)

    @property
    def is_record_or_enum_type(self):
        return TypeHandler.match_record_or_enum_type(self.clang_type.kind)

    @property
    def is_elaborated_type(self):
        return TypeHandler.match_elaborated_type(self.clang_type.kind)

    @property
    def is_typedef_type(self):
        return TypeHandler.match_typedef_type(self.clang_type.kind)

    @property
    def is_innermost_canonical_type_layer_of_basic_type_or_void(self):
        """If the innermost type layer is of basic type or void type."""
        return next(self.categorized_type_layer_kinds(postorder=True)) in (
            TypeHandler.TypeCategory.BASIC,
            TypeHandler.TypeCategory.VOID,
        )

    def is_canonical_const_qualified(self):
        """Returns if the canonical (=fully resolved) type is const qualified."""
        return self.clang_type.get_canonical().is_const_qualified()

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

        rank_counted: int = 0
        for kind in self.clang_type_layer_kinds(canonical=True):
            if constant_array and kind == TypeKind.CONSTANTARRAY:
                rank_counted += 1
            elif incomplete_array and kind == TypeKind.INCOMPLETEARRAY:
                rank_counted += 1
            elif pointer and kind == TypeKind.POINTER:
                rank_counted += 1
            else:
                break
        return rank_counted

    def get_pointer_degree(self, incomplete_array=False) -> int:
        """Returns number of outer type layers which are of TypeKind.POINTER.
        Args:
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.
        """
        return self.get_rank(
            constant_array=False,
            incomplete_array=incomplete_array,
            pointer=True,
        )

    def compare_pointer_degree(self, found_degree, expected_degree):
        """Checks if the found degree equals one of the expected degrees.

        If a positive expected degree is provided, e.g. ``1``, then the found degree
        must match exactly this expression.
        If a negative expected degree is provided, e.g. ``-1``, then the found degree must be
        greater than or equal to the absolute value of the expected degree, e.g. ``found_degree>=1``.
        A tuple consisting of multiple of the above expressions can be provided.
        The function returns ``True`` if there is a match for at least one of the expressions.

        Args:
            found_degree (`int`): The found degree. Assumed to be non-negative.
            expected_degree (`int` or `tuple`): A single expected degree expression or a tuple thereof.

        Returns:
            `bool`: ``True`` if there is a match for one of the expected degree expressions.
        """
        if isinstance(expected_degree, int):
            degrees = (expected_degree,)
        else:
            assert isinstance(
                expected_degree, tuple
            ), "degree: expected int or tuple of int"
            degrees = expected_degree

        for d in degrees:
            if d >= 0:
                if found_degree == d:
                    return True
            else:
                if found_degree >= abs(d):
                    return True
        return False

    def is_pointer_to_kind(
        self, type_kind, degree=1, incomplete_array: bool = False
    ):
        """If this is a pointerof the given ``degree`` to the given type kind. Always uses the canonical type.

        Note:
            Always uses the canonical type.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        if isinstance(type_kind, tuple):
            type_kinds = type_kind
        else:
            assert isinstance(type_kind, clang.cindex.TypeKind)
            type_kinds = (type_kind,)

        layers = list(self.clang_type_layer_kinds(canonical=True))
        found_pointer_degree = self.get_pointer_degree(incomplete_array)
        if layers[found_pointer_degree] in type_kinds:
            return self.compare_pointer_degree(found_pointer_degree, degree)
        return False

    def is_pointer_to_category(
        self,
        type_category,
        degree=1,
        incomplete_array: bool = False,
        subdivide_categories: bool = False,
    ):
        """If this is a pointerof the given ``degree`` to the given type category.

        Args:
            degree (int): Pointer degree. Value < 0 implies any degree >= ``degree`` matches. Defaults to 1.
            incomplete_array (bool, optional): Consider incomplete arrays as pointers too. Defaults to False.

        Note:
            Does not check for any const modifiers.
        """
        if isinstance(type_category, tuple):
            type_categories = type_category
        else:
            assert isinstance(type_category, self.TypeCategory)
            type_categories = (type_category,)

        layers = list(
            self.categorized_type_layer_kinds(
                subdivide_basic_types=subdivide_categories
            )
        )
        found_pointer_degree = self.get_pointer_degree(incomplete_array)
        if layers[found_pointer_degree] in type_categories:
            return self.compare_pointer_degree(found_pointer_degree, degree)
        return False

    def is_constantarray_of_kind_or_category(
        self, type_category=None, type_kind=None
    ):
        """Checks if the type is an (mult-dim) array of a certain category or kind. Always uses the canonical type.

        Args:
            type_category (cparser.TypeHandler.TypeCategory, optional): The category/-ies of the innermost layer. Defaults to None.
            type_kind (clang.cindex.TypeKind, optional): The kind(s) of the innermost layer. Defaults to None.

        Note:
            Always uses the canonical type.

        Note:
            One of type_category or type_kind must be specified.
            Only one of type_category or type_kind must be specified.

        Return:

            A tuple that consists of a bool indicating if a multi-dimensional const array of the searched category/kind has been found
            plus the dimension of that array.

        Example:

            int[16][5]

            has the following layers of type kinds and categories:

            [TypeKind.CONSTANTARRAY,TypeKind.CONSTANTARRAY,TypeKind.INT];
            [TypeCategory.ARRAY,TypeCategory.ARRAY,TypeCategory.BASIC]
        """
        assert (
            type_category or type_kind
        ), "One of type_category or type_kind must be specified."
        assert (
            not type_category or not type_kind
        ), "Only one of type_category or type_kind must be specified."
        from clang.cindex import TypeKind

        if not type_category:
            type_categories = None
        elif isinstance(type_category, tuple):
            type_categories = type_category
        else:
            assert isinstance(type_category, self.TypeCategory)
            type_categories = (type_category,)

        if not type_kind:
            type_kinds = None
        elif isinstance(type_kind, tuple):
            type_kinds = type_kind
        else:
            assert isinstance(type_kind, clang.cindex.TypeKind)
            type_kinds = (type_kind,)

        found_rank = 0
        found_category = False
        found_kind = False
        for kind in self.clang_type_layer_kinds(canonical=True):
            if kind == TypeKind.CONSTANTARRAY:
                found_rank += 1
            elif type_category:
                category = TypeHandler.categorize_clang_type_kind(kind)
                if category in type_categories:
                    found_category = True
            elif type_kind:
                if kind in type_kinds:
                    found_kind = True
        return (found_rank > 0 and (found_category or found_kind), found_rank)
