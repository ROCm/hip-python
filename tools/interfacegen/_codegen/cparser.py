#AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import enum
import hashlib

import clang.cindex

def walk_cursors(root: clang.cindex.Cursor,postorder=False):
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
    def descend_(cursor,level=0,parent_stack=[]):
        if postorder:
            for child in cursor.get_children():
                yield from descend_(child,level+1,parent_stack+[cursor])
        yield (cursor, level, parent_stack) # yield current
        if not postorder:
            for child in cursor.get_children():
                yield from descend_(child,level+1,parent_stack+[cursor])
    yield from descend_(root)

class CParser:
    """Parser for C APIs."""

    def __init__(self, filename: str, append_cflags: list[str] = [], unsaved_files = None):
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
        # print(self._append_cflags)
        self.translation_unit = clang.cindex.TranslationUnit.from_source(
            self.filename,
            args=["-x", "c"] + self.append_cflags,
            options=(
                clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES |
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD # keeps the macro defs as "fake" nodes
            ),
            unsaved_files = self.unsaved_files

        )
        return self

    def walk_cursors(self,cursor=None,postorder=False):
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
        yield from walk_cursors(cursor,postorder)
    
    def render_cursors(self,cursor=None):
        if cursor is None:
            cursor = self.cursor
        result = ""
        for (cursor,level,_) in self.walk_cursors_preorder(cursor):
            indent = "-"*(level)
            result += f"{indent}{str(cursor.kind).replace('CursorKind.','')} '{cursor.spelling}' '{cursor.displayname}' [TYPE-INFO {str(cursor.type.kind).replace('TypeKind.','')} '{cursor.type.spelling}' '{cursor.type.get_canonical().spelling}']"
            if cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                underlying_typedef_type = cursor.underlying_typedef_type
                result += f" [TYPEDEF-INFO '{underlying_typedef_type.spelling}']"
            result += "\n"
        return result

class TypeHandler:
    
    class TypeCategory(enum.IntEnum):
        UNCategorized = -1
        VOID = 0
        BASIC = 10 # basic datatype
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
        CONST_BASIC = BASIC + 100 # basic datatype
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
            return ( value >= TypeHandler.TypeCategory.BASIC.value 
                     and value < TypeHandler.TypeCategory.RECORD.value )

    def __init__(self,clang_type: clang.cindex.Type):
        self.clang_type = clang_type

    @staticmethod
    def is_void_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.VOID

    @staticmethod
    def is_bool_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.BOOL

    @staticmethod
    def is_char_type(type_kind: clang.cindex.TypeKind):
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
    def is_int_type(type_kind: clang.cindex.TypeKind):
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
    def is_float_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.FLOAT,
            clang.cindex.TypeKind.DOUBLE,
            clang.cindex.TypeKind.LONGDOUBLE,
            clang.cindex.TypeKind.FLOAT128,
            clang.cindex.TypeKind.HALF,
            clang.cindex.TypeKind.IBM128,
        )
    
    @staticmethod
    def is_basic_datatype(type_kind: clang.cindex.TypeKind):
        return (
            TypeHandler.is_bool_type(type_kind)
            or TypeHandler.is_char_type(type_kind)
            or TypeHandler.is_int_type(type_kind)
            or TypeHandler.is_float_type(type_kind)
        )

    @staticmethod
    def is_complex_type(type_kind: clang.cindex.TypeKind):
        return clang.cindex.TypeKind == clang.cindex.TypeKind.COMPLEX

    @staticmethod
    def is_other_type(type_kind: clang.cindex.TypeKind):
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
    def is_pointer_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.POINTER, # ATT
            clang.cindex.TypeKind.BLOCKPOINTER, # ATT
            clang.cindex.TypeKind.MEMBERPOINTER, #ATT
        )

    @staticmethod
    def is_function_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.FUNCTIONNOPROTO, # ATT
            clang.cindex.TypeKind.FUNCTIONPROTO, # ATT
        )

    @staticmethod
    def is_arraylike_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.VECTOR, # ATT
            clang.cindex.TypeKind.VARIABLEARRAY, # ATT
            clang.cindex.TypeKind.DEPENDENTSIZEDARRAY, # ATT
            clang.cindex.TypeKind.CONSTANTARRAY, #ATT
            clang.cindex.TypeKind.INCOMPLETEARRAY, #ATT
        )

    @staticmethod
    def is_record_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.RECORD
    
    @staticmethod
    def is_enum_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.ENUM

    @staticmethod
    def is_record_or_enum_type(type_kind: clang.cindex.TypeKind):
        return type_kind in (
            clang.cindex.TypeKind.RECORD, # ATT
            clang.cindex.TypeKind.ENUM, # ATT
        )
    
    @staticmethod
    def is_elaborated_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.ELABORATED #ATT
    
    @staticmethod
    def is_typedef_type(type_kind: clang.cindex.TypeKind):
        return type_kind == clang.cindex.TypeKind.TYPEDEF

    def walk_clang_type_layers(self,postorder=False,canonical=False):
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
            if ( 
                TypeHandler.is_void_type(type_kind)
                or TypeHandler.is_basic_datatype(type_kind)
            ):
                yield clang_type
            elif TypeHandler.is_pointer_type(type_kind):
                pointee = clang_type.get_pointee() 
                if postorder:
                    yield from descend_(clang_type.get_pointee())
                yield clang_type
                if not postorder:
                    yield from descend_(clang_type.get_pointee())
            elif TypeHandler.is_function_type(type_kind):
                yield clang_type
            elif ( TypeHandler.is_arraylike_type(type_kind)
                   or TypeHandler.is_complex_type(type_kind) ):
                if postorder:
                    yield from descend_(clang_type.get_array_element_type())
                yield clang_type
                if not postorder:
                    yield from descend_(clang_type.get_array_element_type())
            elif TypeHandler.is_record_or_enum_type(type_kind):
                yield clang_type
            elif TypeHandler.is_typedef_type(type_kind):
                underlying_type = clang_type.get_declaration().underlying_typedef_type
                if postorder: yield from descend_(underlying_type)
                yield clang_type
                if not postorder: yield from descend_(underlying_type)
            elif TypeHandler.is_elaborated_type(type_kind):
                named_type = clang_type.get_named_type()
                if postorder: yield from descend_(named_type)
                yield clang_type
                if not postorder: yield from descend_(named_type)
            elif TypeHandler.is_other_type(type_kind):
                raise RuntimeError(f"handling types of kind '{type_kind.spelling}' not implemented")
            else:
                raise RuntimeError(f"unknown type kind '{type_kind.spelling}'")
        if canonical:
            yield from descend_(self.clang_type.get_canonical())
        else:
            yield from descend_(self.clang_type)

    def clang_type_layer_kinds(self,postorder=False,canonical=False):
        """Yields the Clang type kinds that constitute this type.

        Args:
            postorder (bool, optional): Post-order walk. Defaults to False.
            canonical (bool, optional): Use the canonical type for the walk.

        Note:
            Note that this is by default a pre-order walk, e.g., if we have a type `void *`,
            we will obtain first the pointer type and then the `void` type.
        """
        for clang_type in self.walk_clang_type_layers(postorder,canonical):
            yield clang_type.kind

    @staticmethod
    def categorize_clang_type_kind(type_kind: clang.cindex.TypeKind,
                                   is_const: bool = False,
                                   subdivide_basic_types: bool = False):
        """
        is_const (bool): If the type is const qualified, a special type category is returned. 
                                   If you do no want this behaviour, just pass False. Defaults to False.
        subdivide_basic_types (bool,optional): If basic datatypes should be further categorized into
                                               the categories: bool, char, int, float. Defaults to false
        """
        if TypeHandler.is_void_type(type_kind):
            result = TypeHandler.TypeCategory.VOID
        elif not subdivide_basic_types and TypeHandler.is_basic_datatype(type_kind):
            result = TypeHandler.TypeCategory.BASIC
        elif TypeHandler.is_bool_type(type_kind):
            result = TypeHandler.TypeCategory.BOOL
        elif TypeHandler.is_char_type(type_kind):
            result = TypeHandler.TypeCategory.CHAR
        elif TypeHandler.is_int_type(type_kind):
            result = TypeHandler.TypeCategory.INT
        elif TypeHandler.is_float_type(type_kind):
            result = TypeHandler.TypeCategory.FLOAT
        elif TypeHandler.is_record_type(type_kind):
            result = TypeHandler.TypeCategory.RECORD
        elif TypeHandler.is_enum_type(type_kind):
            result = TypeHandler.TypeCategory.ENUM
        elif TypeHandler.is_pointer_type(type_kind):
            result = TypeHandler.TypeCategory.POINTER
        elif TypeHandler.is_arraylike_type(type_kind):
            result = TypeHandler.TypeCategory.ARRAY
        elif TypeHandler.is_complex_type(type_kind):
            result = TypeHandler.TypeCategory.COMPLEX
        elif TypeHandler.is_function_type(type_kind):
            result = TypeHandler.TypeCategory.FUNCTION
        else:
            raise ValueError(f"type kind '{type_kind}' could not be Categorized")
        if is_const:
             return TypeHandler.TypeCategory(result.value + TypeHandler.TypeCategory.CONST_VOID.value)
        else:
            return result

    def categorized_type_layers(self,
                               postorder=False,
                               consider_const=False,
                               subdivide_basic_types: bool = False):
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
        for clang_type in self.walk_clang_type_layers(postorder,canonical=True):
            yield TypeHandler.categorize_clang_type_kind(
                clang_type.kind,
                is_const = consider_const and clang_type.is_const_qualified(),
                subdivide_basic_types = subdivide_basic_types,
                )

    def is_canonical_const_qualified(self):
        """Returns if the canonical (=fully resolved) type is const qualified."""
        return self.clang_type.get_canonical().is_const_qualified()
    
class Analysis:
    """Collection of routines for analyzing the contents of a C translation unit.
    """

    @staticmethod
    def _type_analysis_part_header():
        return [
            "cursor.type.spelling",
            "Type Layer Kinds",
            "cursor.type.get_canonical().spelling",
            "Canonical Type Layer Kinds",
            "Canonical Type Layer Kinds (Categorized)",
            "Canonical Type Layer Kinds (Categorized, Const)",
            "Canonical Type Layer Kinds (Categorized, Const, Finer)",
            "Cython C Typename"
        ]

    @staticmethod
    def _type_analysis_part(clang_type: clang.cindex.Type):
        typehandler = TypeHandler(clang_type)
        type_kinds = ",".join([str(t) for t in typehandler.clang_type_layer_kinds(canonical=False,postorder=False)])
        canonical_type_kinds = ",".join([str(t) for t in typehandler.clang_type_layer_kinds(canonical=True,postorder=False)])
        categorized_canonical_type_layer_kinds = ",".join([str(t) for t in typehandler.categorized_type_layers(
            postorder=False)])
        categorized_canonical_type_layer_kinds_w_const = ",".join([str(t) for t in typehandler.categorized_type_layers(
            postorder=False,consider_const=True)])
        categorized_canonical_type_layer_kinds_finer_w_const = ",".join([str(t) for t in typehandler.categorized_type_layers(
            postorder=False,consider_const=True,subdivide_basic_types=True)])

        return [
          f"{clang_type.spelling}",
          f"[{type_kinds}]",
          f"{clang_type.get_canonical().spelling}",
          f"[{canonical_type_kinds}]",
          f"[{categorized_canonical_type_layer_kinds}]",
          f"[{categorized_canonical_type_layer_kinds_w_const}]",
          f"[{categorized_canonical_type_layer_kinds_finer_w_const}]",
          f"{typehandler.get_canonical_cython_type_name()}",
        ]

    @staticmethod
    def subtree_as_csv(root: clang.cindex.Cursor,
                       spelling: str,
                       maxlevel: int,
                       sep: str = ";"):
        """Render a subtree as CSV table.

        Args:
            root (clang.cindex.Cursor): The cursor to do the walk on.
            spelling (str): The spelling/name of the searched cursor.
            maxlevel (int): Max level of the tree to print.
        """
        result = ""
        header = ["Location"]
        header += [f"Level {l}" for l in range(0,maxlevel+1)]
        header += ["cursor.spelling"] + Analysis._type_analysis_part_header()
                   
        result += sep.join(header) + "\n"
        activate_printing = False
        
        def descend_(cursor,level=0):
            nonlocal result
            nonlocal activate_printing
            nonlocal spelling
            nonlocal maxlevel
            nonlocal sep
            if cursor.location is not None:
                if cursor.location.file != None:
                    if spelling == None or cursor.spelling == spelling:
                        activate_printing = True
                    if activate_printing:
                        result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}{sep}"
                        indent = f"{sep}"*(level)
                        result += f"{indent}{str(cursor.kind).replace('CursorKind.','')}"
                        result += (maxlevel-level)*f"{sep}"

                        result += f"{sep}{cursor.spelling}"

                        result += sep + sep.join(Analysis._type_analysis_part(cursor.type))
                        result += "\n"
            for child in cursor.get_children():
                descend_(child,level+1)
            if cursor.spelling == spelling:
                activate_printing = False
        
        descend_(root)
        return result

    @staticmethod
    def type_declarations_as_csv(
        root: clang.cindex.Cursor,
        cursor_filter: callable = lambda cursor: True,
        maxlevel: int = 8,
        include_fields = True,
        sep: str = ";"):
        """Renders nodes associated with type declarations.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["Location"]
        header += [f"Level {l}" for l in range(0,maxlevel+1)] 
        header += ["cursor.spelling"] + Analysis._type_analysis_part_header()
        result += sep.join(header) + "\n"
        for (cursor,level,_) in walk_cursors(root):
            if level > maxlevel:
                continue
            if cursor.kind in (
                clang.cindex.CursorKind.TRANSLATION_UNIT,
                clang.cindex.CursorKind.TYPEDEF_DECL,
                clang.cindex.CursorKind.STRUCT_DECL,
                clang.cindex.CursorKind.UNION_DECL,
                clang.cindex.CursorKind.FIELD_DECL,
                clang.cindex.CursorKind.ENUM_DECL,
                clang.cindex.CursorKind.ENUM_CONSTANT_DECL,
            ):
                if not include_fields and (
                    cursor.kind in (
                        clang.cindex.CursorKind.FIELD_DECL,
                        clang.cindex.CursorKind.ENUM_CONSTANT_DECL
                    )
                ):
                    continue
                if cursor.location is not None:
                    if cursor.location.file != None:
                        if cursor_filter(cursor):
                            result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}{sep}"
                            indent = f"{sep}"*(level)
                            result += f"{indent}{str(cursor.kind).replace('CursorKind.','')}"
                            result += (maxlevel-level)*f"{sep}"
                            result += f"{sep}{cursor.spelling}"

                            result += sep + sep.join(Analysis._type_handler_part(cursor.type))
                            result += "\n"
        return result

    @staticmethod
    def macros_as_csv(
        root: clang.cindex.Cursor,
        cursor_filter: callable = lambda cursor: True,
        sep: str =";"
        ):
        """Returns an overview table of macro definitions.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["cursor.spelling","Tokens (contains macro name and arguments)"]
        result += sep.join(header) + "\n"
        for (cursor,_,parent_stack) in walk_cursors(root):
            if cursor.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
                if cursor_filter(cursor):
                    result += f"{cursor.spelling}"
                    tokens = ",".join([f"'{tk.spelling}'" for tk in cursor.get_tokens()])
                    result += f"{sep}[{tokens}]"
                    result += "\n"
        return result

    @staticmethod
    def parameter_and_return_types_as_csv(
            root: clang.cindex.Cursor,
            cursor_filter: callable = lambda cursor: True,
            sep: str =";"
        ):
        """Returns an overview table of parameter and return values and their types as CSV table.

        Args:
            root (clang.cindex.Cursor): The root cursor
            cursor_filter (callable, optional): Filter for selecting certain cursors, e.g. based on the filename. Defaults to accept-all behavior.
            sep (str,optional): CSV column separator
        """
        result = ""
        header = ["Location"]
        header += ["Function","Reference Kind","CursorKind","cursor.spelling"] + Analysis._type_analysis_part_header()
        result += sep.join(header) + "\n"
        for (cursor,_,parent_stack) in walk_cursors(root):
            if cursor.kind in (
                clang.cindex.CursorKind.PARM_DECL,
                clang.cindex.CursorKind.FUNCTION_DECL,
            ):
                if cursor.location is not None:
                    if cursor.location.file != None:
                        if cursor_filter(cursor): 
                            if cursor.kind == clang.cindex.CursorKind.PARM_DECL:
                                func = parent_stack[1].spelling
                                kind = "Parameter"
                                clang_type = cursor.type
                            else:
                                func = cursor.spelling
                                kind = "Result"
                                clang_type = cursor.result_type                                
                            
                            result += f"{cursor.location.file}:{cursor.location.line}:{cursor.location.column}"
                            result += f"{sep}{func}{sep}{kind}"
                            result += f"{sep}{str(cursor.kind).replace('CursorKind.','')}"
                            result += f"{sep}{cursor.spelling}"

                            result += sep + sep.join(Analysis._type_handler_part(clang_type))
                            result += "\n"
        return result