rocm.bindings.clang.cindex
==========================

.. py:module:: rocm.bindings.clang.cindex

.. autoapi-nested-parse::

   Clang Indexing Library Bindings
   ===============================

   This module provides an interface to the Clang indexing library. It is a
   low-level interface to the indexing library which attempts to match the Clang
   API directly while also being "pythonic". Notable differences from the C API
   are:

    * string results are returned as Python strings, not CXString objects.

    * null cursors are translated to None.

    * access to child cursors is done via iteration, not visitation.

   The major indexing objects are:

     Index

       The top-level object which manages some global library state.

     TranslationUnit

       High-level object encapsulating the AST for a single translation unit. These
       can be loaded from .ast files or parsed on the fly.

     Cursor

       Generic object for representing a node in the AST.

     SourceRange, SourceLocation, and File

       Objects representing information about the input source.

   Most object information is exposed using properties, when the underlying API
   call is efficient.



Exceptions
----------

.. autoapisummary::

   rocm.bindings.clang.cindex.TranslationUnitLoadError


Classes
-------

.. autoapisummary::

   rocm.bindings.clang.cindex.SourceLocation
   rocm.bindings.clang.cindex.SourceRange
   rocm.bindings.clang.cindex.Diagnostic
   rocm.bindings.clang.cindex.FixIt
   rocm.bindings.clang.cindex.TokenKind
   rocm.bindings.clang.cindex.CursorKind
   rocm.bindings.clang.cindex.TemplateArgumentKind
   rocm.bindings.clang.cindex.ExceptionSpecificationKind
   rocm.bindings.clang.cindex.Cursor
   rocm.bindings.clang.cindex.BinaryOperator
   rocm.bindings.clang.cindex.StorageClass
   rocm.bindings.clang.cindex.AvailabilityKind
   rocm.bindings.clang.cindex.AccessSpecifier
   rocm.bindings.clang.cindex.TypeKind
   rocm.bindings.clang.cindex.RefQualifierKind
   rocm.bindings.clang.cindex.LinkageKind
   rocm.bindings.clang.cindex.TLSKind
   rocm.bindings.clang.cindex.Type
   rocm.bindings.clang.cindex.CodeCompletionResults
   rocm.bindings.clang.cindex.Index
   rocm.bindings.clang.cindex.TranslationUnit
   rocm.bindings.clang.cindex.File
   rocm.bindings.clang.cindex.CompileCommand
   rocm.bindings.clang.cindex.CompileCommands
   rocm.bindings.clang.cindex.CompilationDatabase
   rocm.bindings.clang.cindex.Token
   rocm.bindings.clang.cindex.PrintingPolicyProperty
   rocm.bindings.clang.cindex.PrintingPolicy
   rocm.bindings.clang.cindex.Config


Module Contents
---------------

.. py:exception:: TranslationUnitLoadError

   Bases: :py:obj:`Exception`


   Represents an error that occurred when loading a TranslationUnit.

   This is raised in the case where a TranslationUnit could not be
   instantiated due to failure in the libclang library.

   FIXME: Make libclang expose additional error information in this scenario.


.. py:class:: SourceLocation

   Bases: :py:obj:`ctypes.Structure`


   A SourceLocation represents a particular location within a source file.


   .. py:method:: from_position(tu: TranslationUnit, file: File, line: int, column: int) -> SourceLocation
      :staticmethod:


      Retrieve the source location associated with a given file/line/column in
      a particular translation unit.



   .. py:method:: from_offset(tu: TranslationUnit, file: File, offset: int) -> SourceLocation
      :staticmethod:


      Retrieve a SourceLocation from a given character offset.

      tu -- TranslationUnit file belongs to
      file -- File instance to obtain offset from
      offset -- Integer character offset within file



   .. py:property:: file
      :type: File | None


      Get the file represented by this source location.



   .. py:property:: line
      :type: int


      Get the line represented by this source location.



   .. py:property:: column
      :type: int


      Get the column represented by this source location.



   .. py:property:: offset
      :type: int


      Get the file offset represented by this source location.



   .. py:property:: is_in_system_header
      :type: bool


      Returns true if the given source location is in a system header.



.. py:class:: SourceRange

   Bases: :py:obj:`ctypes.Structure`


   A SourceRange describes a range of source locations within the source
   code.


   .. py:method:: from_locations(start: SourceLocation, end: SourceLocation) -> SourceRange
      :staticmethod:



   .. py:property:: start
      :type: SourceLocation


      Return a SourceLocation representing the first character within a
      source range.



   .. py:property:: end
      :type: SourceLocation


      Return a SourceLocation representing the last character within a
      source range.



.. py:class:: Diagnostic(ptr)

   A Diagnostic is a single instance of a Clang diagnostic. It includes the
   diagnostic severity, the message, the location the diagnostic occurred, as
   well as additional source ranges and associated fix-it hints.


   .. py:attribute:: Ignored
      :value: 0



   .. py:attribute:: Note
      :value: 1



   .. py:attribute:: Warning
      :value: 2



   .. py:attribute:: Error
      :value: 3



   .. py:attribute:: Fatal
      :value: 4



   .. py:attribute:: DisplaySourceLocation
      :value: 1



   .. py:attribute:: DisplayColumn
      :value: 2



   .. py:attribute:: DisplaySourceRanges
      :value: 4



   .. py:attribute:: DisplayOption
      :value: 8



   .. py:attribute:: DisplayCategoryId
      :value: 16



   .. py:attribute:: DisplayCategoryName
      :value: 32



   .. py:attribute:: ptr


   .. py:property:: severity


   .. py:property:: location


   .. py:property:: spelling


   .. py:property:: ranges
      :type: NoSliceSequence[SourceRange]



   .. py:property:: fixits
      :type: NoSliceSequence[FixIt]



   .. py:property:: children
      :type: NoSliceSequence[Diagnostic]



   .. py:property:: category_number

      The category number for this diagnostic or 0 if unavailable.



   .. py:property:: category_name

      The string name of the category for this diagnostic.



   .. py:property:: option

      The command-line option that enables this diagnostic.



   .. py:property:: disable_option

      The command-line option that disables this diagnostic.



   .. py:method:: format(options=None)

      Format this diagnostic for display. The options argument takes
      Diagnostic.Display* flags, which can be combined using bitwise OR. If
      the options argument is not provided, the default display options will
      be used.



   .. py:method:: from_param()


.. py:class:: FixIt(range, value)

   A FixIt represents a transformation to be applied to the source to
   "fix-it". The fix-it should be applied by replacing the given source range
   with the given value.


   .. py:attribute:: range


   .. py:attribute:: value


.. py:class:: TokenKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes a specific type of a Token.


   .. py:method:: from_value(value)
      :classmethod:


      Obtain a registered TokenKind instance from its value.



   .. py:attribute:: PUNCTUATION
      :value: 0



   .. py:attribute:: KEYWORD
      :value: 1



   .. py:attribute:: IDENTIFIER
      :value: 2



   .. py:attribute:: LITERAL
      :value: 3



   .. py:attribute:: COMMENT
      :value: 4



.. py:class:: CursorKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   A CursorKind describes the kind of entity that a cursor points to.


   .. py:method:: get_all_kinds()
      :staticmethod:


      Return all CursorKind enumeration instances.



   .. py:method:: is_declaration()

      Test if this is a declaration kind.



   .. py:method:: is_reference()

      Test if this is a reference kind.



   .. py:method:: is_expression()

      Test if this is an expression kind.



   .. py:method:: is_statement()

      Test if this is a statement kind.



   .. py:method:: is_attribute()

      Test if this is an attribute kind.



   .. py:method:: is_invalid()

      Test if this is an invalid kind.



   .. py:method:: is_translation_unit()

      Test if this is a translation unit kind.



   .. py:method:: is_preprocessing()

      Test if this is a preprocessing kind.



   .. py:method:: is_unexposed()

      Test if this is an unexposed kind.



   .. py:attribute:: UNEXPOSED_DECL
      :value: 1



   .. py:attribute:: STRUCT_DECL
      :value: 2



   .. py:attribute:: UNION_DECL
      :value: 3



   .. py:attribute:: CLASS_DECL
      :value: 4



   .. py:attribute:: ENUM_DECL
      :value: 5



   .. py:attribute:: FIELD_DECL
      :value: 6



   .. py:attribute:: ENUM_CONSTANT_DECL
      :value: 7



   .. py:attribute:: FUNCTION_DECL
      :value: 8



   .. py:attribute:: VAR_DECL
      :value: 9



   .. py:attribute:: PARM_DECL
      :value: 10



   .. py:attribute:: OBJC_INTERFACE_DECL
      :value: 11



   .. py:attribute:: OBJC_CATEGORY_DECL
      :value: 12



   .. py:attribute:: OBJC_PROTOCOL_DECL
      :value: 13



   .. py:attribute:: OBJC_PROPERTY_DECL
      :value: 14



   .. py:attribute:: OBJC_IVAR_DECL
      :value: 15



   .. py:attribute:: OBJC_INSTANCE_METHOD_DECL
      :value: 16



   .. py:attribute:: OBJC_CLASS_METHOD_DECL
      :value: 17



   .. py:attribute:: OBJC_IMPLEMENTATION_DECL
      :value: 18



   .. py:attribute:: OBJC_CATEGORY_IMPL_DECL
      :value: 19



   .. py:attribute:: TYPEDEF_DECL
      :value: 20



   .. py:attribute:: CXX_METHOD
      :value: 21



   .. py:attribute:: NAMESPACE
      :value: 22



   .. py:attribute:: LINKAGE_SPEC
      :value: 23



   .. py:attribute:: CONSTRUCTOR
      :value: 24



   .. py:attribute:: DESTRUCTOR
      :value: 25



   .. py:attribute:: CONVERSION_FUNCTION
      :value: 26



   .. py:attribute:: TEMPLATE_TYPE_PARAMETER
      :value: 27



   .. py:attribute:: TEMPLATE_NON_TYPE_PARAMETER
      :value: 28



   .. py:attribute:: TEMPLATE_TEMPLATE_PARAMETER
      :value: 29



   .. py:attribute:: FUNCTION_TEMPLATE
      :value: 30



   .. py:attribute:: CLASS_TEMPLATE
      :value: 31



   .. py:attribute:: CLASS_TEMPLATE_PARTIAL_SPECIALIZATION
      :value: 32



   .. py:attribute:: NAMESPACE_ALIAS
      :value: 33



   .. py:attribute:: USING_DIRECTIVE
      :value: 34



   .. py:attribute:: USING_DECLARATION
      :value: 35



   .. py:attribute:: TYPE_ALIAS_DECL
      :value: 36



   .. py:attribute:: OBJC_SYNTHESIZE_DECL
      :value: 37



   .. py:attribute:: OBJC_DYNAMIC_DECL
      :value: 38



   .. py:attribute:: CXX_ACCESS_SPEC_DECL
      :value: 39



   .. py:attribute:: OBJC_SUPER_CLASS_REF
      :value: 40



   .. py:attribute:: OBJC_PROTOCOL_REF
      :value: 41



   .. py:attribute:: OBJC_CLASS_REF
      :value: 42



   .. py:attribute:: TYPE_REF
      :value: 43



   .. py:attribute:: CXX_BASE_SPECIFIER
      :value: 44



   .. py:attribute:: TEMPLATE_REF
      :value: 45



   .. py:attribute:: NAMESPACE_REF
      :value: 46



   .. py:attribute:: MEMBER_REF
      :value: 47



   .. py:attribute:: LABEL_REF
      :value: 48



   .. py:attribute:: OVERLOADED_DECL_REF
      :value: 49



   .. py:attribute:: VARIABLE_REF
      :value: 50



   .. py:attribute:: INVALID_FILE
      :value: 70



   .. py:attribute:: NO_DECL_FOUND
      :value: 71



   .. py:attribute:: NOT_IMPLEMENTED
      :value: 72



   .. py:attribute:: INVALID_CODE
      :value: 73



   .. py:attribute:: UNEXPOSED_EXPR
      :value: 100



   .. py:attribute:: DECL_REF_EXPR
      :value: 101



   .. py:attribute:: MEMBER_REF_EXPR
      :value: 102



   .. py:attribute:: CALL_EXPR
      :value: 103



   .. py:attribute:: OBJC_MESSAGE_EXPR
      :value: 104



   .. py:attribute:: BLOCK_EXPR
      :value: 105



   .. py:attribute:: INTEGER_LITERAL
      :value: 106



   .. py:attribute:: FLOATING_LITERAL
      :value: 107



   .. py:attribute:: IMAGINARY_LITERAL
      :value: 108



   .. py:attribute:: STRING_LITERAL
      :value: 109



   .. py:attribute:: CHARACTER_LITERAL
      :value: 110



   .. py:attribute:: PAREN_EXPR
      :value: 111



   .. py:attribute:: UNARY_OPERATOR
      :value: 112



   .. py:attribute:: ARRAY_SUBSCRIPT_EXPR
      :value: 113



   .. py:attribute:: BINARY_OPERATOR
      :value: 114



   .. py:attribute:: COMPOUND_ASSIGNMENT_OPERATOR
      :value: 115



   .. py:attribute:: CONDITIONAL_OPERATOR
      :value: 116



   .. py:attribute:: CSTYLE_CAST_EXPR
      :value: 117



   .. py:attribute:: COMPOUND_LITERAL_EXPR
      :value: 118



   .. py:attribute:: INIT_LIST_EXPR
      :value: 119



   .. py:attribute:: ADDR_LABEL_EXPR
      :value: 120



   .. py:attribute:: StmtExpr
      :value: 121



   .. py:attribute:: GENERIC_SELECTION_EXPR
      :value: 122



   .. py:attribute:: GNU_NULL_EXPR
      :value: 123



   .. py:attribute:: CXX_STATIC_CAST_EXPR
      :value: 124



   .. py:attribute:: CXX_DYNAMIC_CAST_EXPR
      :value: 125



   .. py:attribute:: CXX_REINTERPRET_CAST_EXPR
      :value: 126



   .. py:attribute:: CXX_CONST_CAST_EXPR
      :value: 127



   .. py:attribute:: CXX_FUNCTIONAL_CAST_EXPR
      :value: 128



   .. py:attribute:: CXX_TYPEID_EXPR
      :value: 129



   .. py:attribute:: CXX_BOOL_LITERAL_EXPR
      :value: 130



   .. py:attribute:: CXX_NULL_PTR_LITERAL_EXPR
      :value: 131



   .. py:attribute:: CXX_THIS_EXPR
      :value: 132



   .. py:attribute:: CXX_THROW_EXPR
      :value: 133



   .. py:attribute:: CXX_NEW_EXPR
      :value: 134



   .. py:attribute:: CXX_DELETE_EXPR
      :value: 135



   .. py:attribute:: CXX_UNARY_EXPR
      :value: 136



   .. py:attribute:: OBJC_STRING_LITERAL
      :value: 137



   .. py:attribute:: OBJC_ENCODE_EXPR
      :value: 138



   .. py:attribute:: OBJC_SELECTOR_EXPR
      :value: 139



   .. py:attribute:: OBJC_PROTOCOL_EXPR
      :value: 140



   .. py:attribute:: OBJC_BRIDGE_CAST_EXPR
      :value: 141



   .. py:attribute:: PACK_EXPANSION_EXPR
      :value: 142



   .. py:attribute:: SIZE_OF_PACK_EXPR
      :value: 143



   .. py:attribute:: LAMBDA_EXPR
      :value: 144



   .. py:attribute:: OBJ_BOOL_LITERAL_EXPR
      :value: 145



   .. py:attribute:: OBJ_SELF_EXPR
      :value: 146



   .. py:attribute:: OMP_ARRAY_SECTION_EXPR
      :value: 147



   .. py:attribute:: OBJC_AVAILABILITY_CHECK_EXPR
      :value: 148



   .. py:attribute:: FIXED_POINT_LITERAL
      :value: 149



   .. py:attribute:: OMP_ARRAY_SHAPING_EXPR
      :value: 150



   .. py:attribute:: OMP_ITERATOR_EXPR
      :value: 151



   .. py:attribute:: CXX_ADDRSPACE_CAST_EXPR
      :value: 152



   .. py:attribute:: CONCEPT_SPECIALIZATION_EXPR
      :value: 153



   .. py:attribute:: REQUIRES_EXPR
      :value: 154



   .. py:attribute:: CXX_PAREN_LIST_INIT_EXPR
      :value: 155



   .. py:attribute:: PACK_INDEXING_EXPR
      :value: 156



   .. py:attribute:: UNEXPOSED_STMT
      :value: 200



   .. py:attribute:: LABEL_STMT
      :value: 201



   .. py:attribute:: COMPOUND_STMT
      :value: 202



   .. py:attribute:: CASE_STMT
      :value: 203



   .. py:attribute:: DEFAULT_STMT
      :value: 204



   .. py:attribute:: IF_STMT
      :value: 205



   .. py:attribute:: SWITCH_STMT
      :value: 206



   .. py:attribute:: WHILE_STMT
      :value: 207



   .. py:attribute:: DO_STMT
      :value: 208



   .. py:attribute:: FOR_STMT
      :value: 209



   .. py:attribute:: GOTO_STMT
      :value: 210



   .. py:attribute:: INDIRECT_GOTO_STMT
      :value: 211



   .. py:attribute:: CONTINUE_STMT
      :value: 212



   .. py:attribute:: BREAK_STMT
      :value: 213



   .. py:attribute:: RETURN_STMT
      :value: 214



   .. py:attribute:: ASM_STMT
      :value: 215



   .. py:attribute:: OBJC_AT_TRY_STMT
      :value: 216



   .. py:attribute:: OBJC_AT_CATCH_STMT
      :value: 217



   .. py:attribute:: OBJC_AT_FINALLY_STMT
      :value: 218



   .. py:attribute:: OBJC_AT_THROW_STMT
      :value: 219



   .. py:attribute:: OBJC_AT_SYNCHRONIZED_STMT
      :value: 220



   .. py:attribute:: OBJC_AUTORELEASE_POOL_STMT
      :value: 221



   .. py:attribute:: OBJC_FOR_COLLECTION_STMT
      :value: 222



   .. py:attribute:: CXX_CATCH_STMT
      :value: 223



   .. py:attribute:: CXX_TRY_STMT
      :value: 224



   .. py:attribute:: CXX_FOR_RANGE_STMT
      :value: 225



   .. py:attribute:: SEH_TRY_STMT
      :value: 226



   .. py:attribute:: SEH_EXCEPT_STMT
      :value: 227



   .. py:attribute:: SEH_FINALLY_STMT
      :value: 228



   .. py:attribute:: MS_ASM_STMT
      :value: 229



   .. py:attribute:: NULL_STMT
      :value: 230



   .. py:attribute:: DECL_STMT
      :value: 231



   .. py:attribute:: OMP_PARALLEL_DIRECTIVE
      :value: 232



   .. py:attribute:: OMP_SIMD_DIRECTIVE
      :value: 233



   .. py:attribute:: OMP_FOR_DIRECTIVE
      :value: 234



   .. py:attribute:: OMP_SECTIONS_DIRECTIVE
      :value: 235



   .. py:attribute:: OMP_SECTION_DIRECTIVE
      :value: 236



   .. py:attribute:: OMP_SINGLE_DIRECTIVE
      :value: 237



   .. py:attribute:: OMP_PARALLEL_FOR_DIRECTIVE
      :value: 238



   .. py:attribute:: OMP_PARALLEL_SECTIONS_DIRECTIVE
      :value: 239



   .. py:attribute:: OMP_TASK_DIRECTIVE
      :value: 240



   .. py:attribute:: OMP_MASTER_DIRECTIVE
      :value: 241



   .. py:attribute:: OMP_CRITICAL_DIRECTIVE
      :value: 242



   .. py:attribute:: OMP_TASKYIELD_DIRECTIVE
      :value: 243



   .. py:attribute:: OMP_BARRIER_DIRECTIVE
      :value: 244



   .. py:attribute:: OMP_TASKWAIT_DIRECTIVE
      :value: 245



   .. py:attribute:: OMP_FLUSH_DIRECTIVE
      :value: 246



   .. py:attribute:: SEH_LEAVE_STMT
      :value: 247



   .. py:attribute:: OMP_ORDERED_DIRECTIVE
      :value: 248



   .. py:attribute:: OMP_ATOMIC_DIRECTIVE
      :value: 249



   .. py:attribute:: OMP_FOR_SIMD_DIRECTIVE
      :value: 250



   .. py:attribute:: OMP_PARALLELFORSIMD_DIRECTIVE
      :value: 251



   .. py:attribute:: OMP_TARGET_DIRECTIVE
      :value: 252



   .. py:attribute:: OMP_TEAMS_DIRECTIVE
      :value: 253



   .. py:attribute:: OMP_TASKGROUP_DIRECTIVE
      :value: 254



   .. py:attribute:: OMP_CANCELLATION_POINT_DIRECTIVE
      :value: 255



   .. py:attribute:: OMP_CANCEL_DIRECTIVE
      :value: 256



   .. py:attribute:: OMP_TARGET_DATA_DIRECTIVE
      :value: 257



   .. py:attribute:: OMP_TASK_LOOP_DIRECTIVE
      :value: 258



   .. py:attribute:: OMP_TASK_LOOP_SIMD_DIRECTIVE
      :value: 259



   .. py:attribute:: OMP_DISTRIBUTE_DIRECTIVE
      :value: 260



   .. py:attribute:: OMP_TARGET_ENTER_DATA_DIRECTIVE
      :value: 261



   .. py:attribute:: OMP_TARGET_EXIT_DATA_DIRECTIVE
      :value: 262



   .. py:attribute:: OMP_TARGET_PARALLEL_DIRECTIVE
      :value: 263



   .. py:attribute:: OMP_TARGET_PARALLELFOR_DIRECTIVE
      :value: 264



   .. py:attribute:: OMP_TARGET_UPDATE_DIRECTIVE
      :value: 265



   .. py:attribute:: OMP_DISTRIBUTE_PARALLELFOR_DIRECTIVE
      :value: 266



   .. py:attribute:: OMP_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
      :value: 267



   .. py:attribute:: OMP_DISTRIBUTE_SIMD_DIRECTIVE
      :value: 268



   .. py:attribute:: OMP_TARGET_PARALLEL_FOR_SIMD_DIRECTIVE
      :value: 269



   .. py:attribute:: OMP_TARGET_SIMD_DIRECTIVE
      :value: 270



   .. py:attribute:: OMP_TEAMS_DISTRIBUTE_DIRECTIVE
      :value: 271



   .. py:attribute:: OMP_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE
      :value: 272



   .. py:attribute:: OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
      :value: 273



   .. py:attribute:: OMP_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE
      :value: 274



   .. py:attribute:: OMP_TARGET_TEAMS_DIRECTIVE
      :value: 275



   .. py:attribute:: OMP_TARGET_TEAMS_DISTRIBUTE_DIRECTIVE
      :value: 276



   .. py:attribute:: OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_DIRECTIVE
      :value: 277



   .. py:attribute:: OMP_TARGET_TEAMS_DISTRIBUTE_PARALLEL_FOR_SIMD_DIRECTIVE
      :value: 278



   .. py:attribute:: OMP_TARGET_TEAMS_DISTRIBUTE_SIMD_DIRECTIVE
      :value: 279



   .. py:attribute:: BUILTIN_BIT_CAST_EXPR
      :value: 280



   .. py:attribute:: OMP_MASTER_TASK_LOOP_DIRECTIVE
      :value: 281



   .. py:attribute:: OMP_PARALLEL_MASTER_TASK_LOOP_DIRECTIVE
      :value: 282



   .. py:attribute:: OMP_MASTER_TASK_LOOP_SIMD_DIRECTIVE
      :value: 283



   .. py:attribute:: OMP_PARALLEL_MASTER_TASK_LOOP_SIMD_DIRECTIVE
      :value: 284



   .. py:attribute:: OMP_PARALLEL_MASTER_DIRECTIVE
      :value: 285



   .. py:attribute:: OMP_DEPOBJ_DIRECTIVE
      :value: 286



   .. py:attribute:: OMP_SCAN_DIRECTIVE
      :value: 287



   .. py:attribute:: OMP_TILE_DIRECTIVE
      :value: 288



   .. py:attribute:: OMP_CANONICAL_LOOP
      :value: 289



   .. py:attribute:: OMP_INTEROP_DIRECTIVE
      :value: 290



   .. py:attribute:: OMP_DISPATCH_DIRECTIVE
      :value: 291



   .. py:attribute:: OMP_MASKED_DIRECTIVE
      :value: 292



   .. py:attribute:: OMP_UNROLL_DIRECTIVE
      :value: 293



   .. py:attribute:: OMP_META_DIRECTIVE
      :value: 294



   .. py:attribute:: OMP_GENERIC_LOOP_DIRECTIVE
      :value: 295



   .. py:attribute:: OMP_TEAMS_GENERIC_LOOP_DIRECTIVE
      :value: 296



   .. py:attribute:: OMP_TARGET_TEAMS_GENERIC_LOOP_DIRECTIVE
      :value: 297



   .. py:attribute:: OMP_PARALLEL_GENERIC_LOOP_DIRECTIVE
      :value: 298



   .. py:attribute:: OMP_TARGET_PARALLEL_GENERIC_LOOP_DIRECTIVE
      :value: 299



   .. py:attribute:: OMP_PARALLEL_MASKED_DIRECTIVE
      :value: 300



   .. py:attribute:: OMP_MASKED_TASK_LOOP_DIRECTIVE
      :value: 301



   .. py:attribute:: OMP_MASKED_TASK_LOOP_SIMD_DIRECTIVE
      :value: 302



   .. py:attribute:: OMP_PARALLEL_MASKED_TASK_LOOP_DIRECTIVE
      :value: 303



   .. py:attribute:: OMP_PARALLEL_MASKED_TASK_LOOP_SIMD_DIRECTIVE
      :value: 304



   .. py:attribute:: OMP_ERROR_DIRECTIVE
      :value: 305



   .. py:attribute:: OMP_SCOPE_DIRECTIVE
      :value: 306



   .. py:attribute:: OMP_REVERSE_DIRECTIVE
      :value: 307



   .. py:attribute:: OMP_INTERCHANGE_DIRECTIVE
      :value: 308



   .. py:attribute:: OMP_ASSUME_DIRECTIVE
      :value: 309



   .. py:attribute:: OMP_STRIPE_DIRECTIVE
      :value: 310



   .. py:attribute:: OMP_FUSE_DIRECTIVE
      :value: 311



   .. py:attribute:: OMP_SPLIT_DIRECTIVE
      :value: 312



   .. py:attribute:: OPEN_ACC_COMPUTE_DIRECTIVE
      :value: 320



   .. py:attribute:: OPEN_ACC_LOOP_CONSTRUCT
      :value: 321



   .. py:attribute:: OPEN_ACC_COMBINED_CONSTRUCT
      :value: 322



   .. py:attribute:: OPEN_ACC_DATA_CONSTRUCT
      :value: 323



   .. py:attribute:: OPEN_ACC_ENTER_DATA_CONSTRUCT
      :value: 324



   .. py:attribute:: OPEN_ACC_EXIT_DATA_CONSTRUCT
      :value: 325



   .. py:attribute:: OPEN_ACC_HOST_DATA_CONSTRUCT
      :value: 326



   .. py:attribute:: OPEN_ACC_WAIT_CONSTRUCT
      :value: 327



   .. py:attribute:: OPEN_ACC_INIT_CONSTRUCT
      :value: 328



   .. py:attribute:: OPEN_ACC_SHUTDOWN_CONSTRUCT
      :value: 329



   .. py:attribute:: OPEN_ACC_SET_CONSTRUCT
      :value: 330



   .. py:attribute:: OPEN_ACC_UPDATE_CONSTRUCT
      :value: 331



   .. py:attribute:: OPEN_ACC_ATOMIC_CONSTRUCT
      :value: 332



   .. py:attribute:: OPEN_ACC_CACHE_CONSTRUCT
      :value: 333



   .. py:attribute:: TRANSLATION_UNIT
      :value: 350



   .. py:attribute:: UNEXPOSED_ATTR
      :value: 400



   .. py:attribute:: IB_ACTION_ATTR
      :value: 401



   .. py:attribute:: IB_OUTLET_ATTR
      :value: 402



   .. py:attribute:: IB_OUTLET_COLLECTION_ATTR
      :value: 403



   .. py:attribute:: CXX_FINAL_ATTR
      :value: 404



   .. py:attribute:: CXX_OVERRIDE_ATTR
      :value: 405



   .. py:attribute:: ANNOTATE_ATTR
      :value: 406



   .. py:attribute:: ASM_LABEL_ATTR
      :value: 407



   .. py:attribute:: PACKED_ATTR
      :value: 408



   .. py:attribute:: PURE_ATTR
      :value: 409



   .. py:attribute:: CONST_ATTR
      :value: 410



   .. py:attribute:: NODUPLICATE_ATTR
      :value: 411



   .. py:attribute:: CUDACONSTANT_ATTR
      :value: 412



   .. py:attribute:: CUDADEVICE_ATTR
      :value: 413



   .. py:attribute:: CUDAGLOBAL_ATTR
      :value: 414



   .. py:attribute:: CUDAHOST_ATTR
      :value: 415



   .. py:attribute:: CUDASHARED_ATTR
      :value: 416



   .. py:attribute:: VISIBILITY_ATTR
      :value: 417



   .. py:attribute:: DLLEXPORT_ATTR
      :value: 418



   .. py:attribute:: DLLIMPORT_ATTR
      :value: 419



   .. py:attribute:: NS_RETURNS_RETAINED
      :value: 420



   .. py:attribute:: NS_RETURNS_NOT_RETAINED
      :value: 421



   .. py:attribute:: NS_RETURNS_AUTORELEASED
      :value: 422



   .. py:attribute:: NS_CONSUMES_SELF
      :value: 423



   .. py:attribute:: NS_CONSUMED
      :value: 424



   .. py:attribute:: OBJC_EXCEPTION
      :value: 425



   .. py:attribute:: OBJC_NSOBJECT
      :value: 426



   .. py:attribute:: OBJC_INDEPENDENT_CLASS
      :value: 427



   .. py:attribute:: OBJC_PRECISE_LIFETIME
      :value: 428



   .. py:attribute:: OBJC_RETURNS_INNER_POINTER
      :value: 429



   .. py:attribute:: OBJC_REQUIRES_SUPER
      :value: 430



   .. py:attribute:: OBJC_ROOT_CLASS
      :value: 431



   .. py:attribute:: OBJC_SUBCLASSING_RESTRICTED
      :value: 432



   .. py:attribute:: OBJC_EXPLICIT_PROTOCOL_IMPL
      :value: 433



   .. py:attribute:: OBJC_DESIGNATED_INITIALIZER
      :value: 434



   .. py:attribute:: OBJC_RUNTIME_VISIBLE
      :value: 435



   .. py:attribute:: OBJC_BOXABLE
      :value: 436



   .. py:attribute:: FLAG_ENUM
      :value: 437



   .. py:attribute:: CONVERGENT_ATTR
      :value: 438



   .. py:attribute:: WARN_UNUSED_ATTR
      :value: 439



   .. py:attribute:: WARN_UNUSED_RESULT_ATTR
      :value: 440



   .. py:attribute:: ALIGNED_ATTR
      :value: 441



   .. py:attribute:: PREPROCESSING_DIRECTIVE
      :value: 500



   .. py:attribute:: MACRO_DEFINITION
      :value: 501



   .. py:attribute:: MACRO_INSTANTIATION
      :value: 502



   .. py:attribute:: INCLUSION_DIRECTIVE
      :value: 503



   .. py:attribute:: MODULE_IMPORT_DECL
      :value: 600



   .. py:attribute:: TYPE_ALIAS_TEMPLATE_DECL
      :value: 601



   .. py:attribute:: STATIC_ASSERT
      :value: 602



   .. py:attribute:: FRIEND_DECL
      :value: 603



   .. py:attribute:: CONCEPT_DECL
      :value: 604



   .. py:attribute:: OVERLOAD_CANDIDATE
      :value: 700



.. py:class:: TemplateArgumentKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   A TemplateArgumentKind describes the kind of entity that a template argument
   represents.


   .. py:attribute:: NULL
      :value: 0



   .. py:attribute:: TYPE
      :value: 1



   .. py:attribute:: DECLARATION
      :value: 2



   .. py:attribute:: NULLPTR
      :value: 3



   .. py:attribute:: INTEGRAL
      :value: 4



   .. py:attribute:: TEMPLATE
      :value: 5



   .. py:attribute:: TEMPLATE_EXPANSION
      :value: 6



   .. py:attribute:: EXPRESSION
      :value: 7



   .. py:attribute:: PACK
      :value: 8



   .. py:attribute:: INVALID
      :value: 9



.. py:class:: ExceptionSpecificationKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   An ExceptionSpecificationKind describes the kind of exception specification
   that a function has.


   .. py:attribute:: NONE
      :value: 0



   .. py:attribute:: DYNAMIC_NONE
      :value: 1



   .. py:attribute:: DYNAMIC
      :value: 2



   .. py:attribute:: MS_ANY
      :value: 3



   .. py:attribute:: BASIC_NOEXCEPT
      :value: 4



   .. py:attribute:: COMPUTED_NOEXCEPT
      :value: 5



   .. py:attribute:: UNEVALUATED
      :value: 6



   .. py:attribute:: UNINSTANTIATED
      :value: 7



   .. py:attribute:: UNPARSED
      :value: 8



   .. py:attribute:: NOTHROW
      :value: 9



.. py:class:: Cursor

   Bases: :py:obj:`ctypes.Structure`


   The Cursor class represents a reference to an element within the AST. It
   acts as a kind of iterator.

   Null cursors are mapped to None.


   .. py:method:: from_location(tu: TranslationUnit, location: SourceLocation) -> Cursor | None
      :staticmethod:



   .. py:method:: is_null() -> bool


   .. py:method:: is_definition() -> bool

      Returns true if the declaration pointed at by the cursor is also a
      definition of that entity.



   .. py:method:: is_const_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared 'const'.



   .. py:method:: is_converting_constructor() -> bool

      Returns True if the cursor refers to a C++ converting constructor.



   .. py:method:: is_copy_constructor() -> bool

      Returns True if the cursor refers to a C++ copy constructor.



   .. py:method:: is_default_constructor() -> bool

      Returns True if the cursor refers to a C++ default constructor.



   .. py:method:: is_move_constructor() -> bool

      Returns True if the cursor refers to a C++ move constructor.



   .. py:method:: is_default_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared '= default'.



   .. py:method:: is_deleted_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared '= delete'.



   .. py:method:: is_copy_assignment_operator_method() -> bool

      Returnrs True if the cursor refers to a copy-assignment operator.

      A copy-assignment operator `X::operator=` is a non-static,
      non-template member function of _class_ `X` with exactly one
      parameter of type `X`, `X&`, `const X&`, `volatile X&` or `const
      volatile X&`.


      That is, for example, the `operator=` in:

         class Foo {
             bool operator=(const volatile Foo&);
         };

      Is a copy-assignment operator, while the `operator=` in:

         class Bar {
             bool operator=(const int&);
         };

      Is not.



   .. py:method:: is_move_assignment_operator_method() -> bool

      Returnrs True if the cursor refers to a move-assignment operator.

      A move-assignment operator `X::operator=` is a non-static,
      non-template member function of _class_ `X` with exactly one
      parameter of type `X&&`, `const X&&`, `volatile X&&` or `const
      volatile X&&`.


      That is, for example, the `operator=` in:

         class Foo {
             bool operator=(const volatile Foo&&);
         };

      Is a move-assignment operator, while the `operator=` in:

         class Bar {
             bool operator=(const int&&);
         };

      Is not.



   .. py:method:: is_explicit_method() -> bool

      Determines if a C++ constructor or conversion function is
      explicit, returning 1 if such is the case and 0 otherwise.

      Constructors or conversion functions are declared explicit through
      the use of the explicit specifier.

      For example, the following constructor and conversion function are
      not explicit as they lack the explicit specifier:

          class Foo {
              Foo();
              operator int();
          };

      While the following constructor and conversion function are
      explicit as they are declared with the explicit specifier.

          class Foo {
              explicit Foo();
              explicit operator int();
          };

      This method will return 0 when given a cursor pointing to one of
      the former declarations and it will return 1 for a cursor pointing
      to the latter declarations.

      The explicit specifier allows the user to specify a
      conditional compile-time expression whose value decides
      whether the marked element is explicit or not.

      For example:

          constexpr bool foo(int i) { return i % 2 == 0; }

          class Foo {
               explicit(foo(1)) Foo();
               explicit(foo(2)) operator int();
          }

      This method will return 0 for the constructor and 1 for
      the conversion function.



   .. py:method:: is_mutable_field() -> bool

      Returns True if the cursor refers to a C++ field that is declared
      'mutable'.



   .. py:method:: is_pure_virtual_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared pure virtual.



   .. py:method:: is_static_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared 'static'.



   .. py:method:: is_virtual_method() -> bool

      Returns True if the cursor refers to a C++ member function or member
      function template that is declared 'virtual'.



   .. py:method:: is_abstract_record() -> bool

      Returns True if the cursor refers to a C++ record declaration
      that has pure virtual member functions.



   .. py:method:: is_scoped_enum() -> bool

      Returns True if the cursor refers to a scoped enum declaration.



   .. py:method:: get_definition() -> Cursor | None

      If the cursor is a reference to a declaration or a declaration of
      some entity, return a cursor that points to the definition of that
      entity.



   .. py:method:: get_usr() -> str

      Return the Unified Symbol Resolution (USR) for the entity referenced
      by the given cursor.

      A Unified Symbol Resolution (USR) is a string that identifies a
      particular entity (function, class, variable, etc.) within a
      program. USRs can be compared across translation units to determine,
      e.g., when references in one translation refer to an entity defined in
      another translation unit.



   .. py:method:: get_included_file() -> File

      Returns the File that is included by the current inclusion cursor.



   .. py:property:: kind
      :type: CursorKind


      Return the kind of this cursor.



   .. py:property:: spelling
      :type: str


      Return the spelling of the entity pointed at by the cursor.



   .. py:method:: pretty_printed(policy: PrintingPolicy) -> str

      Pretty print declarations.
      Parameters:
      policy -- The policy to control the entities being printed.



   .. py:property:: displayname
      :type: str


      Return the display name for the entity referenced by this cursor.

      The display name contains extra information that helps identify the
      cursor, such as the parameters of a function or template or the
      arguments of a class template specialization.



   .. py:property:: mangled_name
      :type: str


      Return the mangled name for the entity referenced by this cursor.



   .. py:property:: location
      :type: SourceLocation


      Return the source location (the starting character) of the entity
      pointed at by the cursor.



   .. py:property:: linkage
      :type: LinkageKind


      Return the linkage of this cursor.



   .. py:property:: language
      :type: LanguageKind


      Determine the "language" of the entity referred to by a given cursor.



   .. py:property:: tls_kind
      :type: TLSKind


      Return the thread-local storage (TLS) kind of this cursor.



   .. py:property:: extent
      :type: SourceRange


      Return the source range (the range of text) occupied by the entity
      pointed at by the cursor.



   .. py:property:: storage_class
      :type: StorageClass


      Retrieves the storage class (if any) of the entity pointed at by the
      cursor.



   .. py:property:: availability
      :type: AvailabilityKind


      Retrieves the availability of the entity pointed at by the cursor.



   .. py:property:: binary_operator
      :type: BinaryOperator


      Retrieves the opcode if this cursor points to a binary operator
      :return:



   .. py:property:: access_specifier
      :type: AccessSpecifier


      Retrieves the access specifier (if any) of the entity pointed at by the
      cursor.



   .. py:property:: type
      :type: Type


      Retrieve the Type (if any) of the entity pointed at by the cursor.



   .. py:property:: canonical
      :type: Cursor


      Return the canonical Cursor corresponding to this Cursor.

      The canonical cursor is the cursor which is representative for the
      underlying entity. For example, if you have multiple forward
      declarations for the same class, the canonical cursor for the forward
      declarations will be identical.



   .. py:property:: result_type
      :type: Type


      Retrieve the Type of the result for this Cursor.



   .. py:property:: exception_specification_kind
      :type: ExceptionSpecificationKind


      Retrieve the exception specification kind, which is one of the values
      from the ExceptionSpecificationKind enumeration.



   .. py:property:: underlying_typedef_type
      :type: Type


      Return the underlying type of a typedef declaration.

      Returns a Type for the typedef this cursor is a declaration for. If
      the current cursor is not a typedef, this raises.



   .. py:property:: enum_type
      :type: Type


      Return the integer type of an enum declaration.

      Returns a Type corresponding to an integer. If the cursor is not for an
      enum, this raises.



   .. py:property:: enum_value
      :type: int


      Return the value of an enum constant.



   .. py:property:: objc_type_encoding
      :type: str


      Return the Objective-C type encoding as a str.



   .. py:property:: hash
      :type: int


      Returns a hash of the cursor as an int.



   .. py:property:: semantic_parent
      :type: Cursor | None


      Return the semantic parent for this cursor.



   .. py:property:: lexical_parent
      :type: Cursor | None


      Return the lexical parent for this cursor.



   .. py:property:: specialized_template
      :type: Cursor | None


      Return the primary template that this cursor is a specialization of, if any.



   .. py:property:: translation_unit
      :type: TranslationUnit


      Returns the TranslationUnit to which this Cursor belongs.



   .. py:property:: referenced
      :type: Cursor | None


      For a cursor that is a reference, returns a cursor
      representing the entity that it references.



   .. py:property:: brief_comment
      :type: str


      Returns the brief comment text associated with that Cursor



   .. py:property:: raw_comment
      :type: str


      Returns the raw comment text associated with that Cursor



   .. py:method:: get_arguments() -> Iterator[Cursor | None]

      Return an iterator for accessing the arguments of this cursor.



   .. py:method:: get_num_template_arguments() -> int

      Returns the number of template args associated with this cursor.



   .. py:method:: get_template_argument_kind(num: int) -> TemplateArgumentKind

      Returns the TemplateArgumentKind for the indicated template
      argument.



   .. py:method:: get_template_argument_type(num: int) -> Type

      Returns the CXType for the indicated template argument.



   .. py:method:: get_template_argument_value(num: int) -> int

      Returns the value of the indicated arg as a signed 64b integer.



   .. py:method:: get_template_argument_unsigned_value(num: int) -> int

      Returns the value of the indicated arg as an unsigned 64b integer.



   .. py:method:: get_children() -> Iterator[Cursor]

      Return an iterator for accessing the children of this cursor.



   .. py:method:: walk_preorder() -> Iterator[Cursor]

      Depth-first preorder walk over the cursor and its descendants.

      Yields cursors.



   .. py:method:: get_tokens() -> Iterator[Token]

      Obtain Token instances formulating that compose this Cursor.

      This is a generator for Token instances. It returns all tokens which
      occupy the extent this cursor occupies.



   .. py:method:: get_field_offsetof() -> int

      Returns the offsetof the FIELD_DECL pointed by this Cursor.



   .. py:method:: get_base_offsetof(parent: Cursor) -> int

      Returns the offsetof the CXX_BASE_SPECIFIER pointed by this Cursor.



   .. py:method:: is_virtual_base() -> bool

      Returns whether the CXX_BASE_SPECIFIER pointed by this Cursor is virtual.



   .. py:method:: is_anonymous() -> bool

      Check whether this is a record type without a name, or a field where
      the type is a record type without a name.

      Use is_anonymous_record_decl to check whether a record is an
      "anonymous union" as defined in the C/C++ standard.



   .. py:method:: is_anonymous_record_decl() -> bool

      Check if the record is an anonymous union as defined in the C/C++ standard
      (or an "anonymous struct", the corresponding non-standard extension for
      structs).



   .. py:method:: is_bitfield() -> bool

      Check if the field is a bitfield.



   .. py:method:: get_bitfield_width() -> int

      Retrieve the width of a bitfield.



   .. py:method:: is_function_inlined() -> bool

      Check if the function is inlined.



   .. py:method:: has_attrs() -> bool

      Determine whether the given cursor has any attributes.



   .. py:method:: from_result(res: Cursor, arg: Cursor | TranslationUnit | Type) -> Cursor | None
      :staticmethod:



   .. py:method:: from_cursor_result(res: Cursor, arg: Cursor) -> Cursor | None
      :staticmethod:



   .. py:method:: from_non_null_cursor_result(res: Cursor, arg: Cursor | Type) -> Cursor
      :staticmethod:



.. py:class:: BinaryOperator(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the BinaryOperator of a declaration


   .. py:property:: is_assignment


   .. py:attribute:: Invalid
      :value: 0



   .. py:attribute:: PtrMemD
      :value: 1



   .. py:attribute:: PtrMemI
      :value: 2



   .. py:attribute:: Mul
      :value: 3



   .. py:attribute:: Div
      :value: 4



   .. py:attribute:: Rem
      :value: 5



   .. py:attribute:: Add
      :value: 6



   .. py:attribute:: Sub
      :value: 7



   .. py:attribute:: Shl
      :value: 8



   .. py:attribute:: Shr
      :value: 9



   .. py:attribute:: Cmp
      :value: 10



   .. py:attribute:: LT
      :value: 11



   .. py:attribute:: GT
      :value: 12



   .. py:attribute:: LE
      :value: 13



   .. py:attribute:: GE
      :value: 14



   .. py:attribute:: EQ
      :value: 15



   .. py:attribute:: NE
      :value: 16



   .. py:attribute:: And
      :value: 17



   .. py:attribute:: Xor
      :value: 18



   .. py:attribute:: Or
      :value: 19



   .. py:attribute:: LAnd
      :value: 20



   .. py:attribute:: LOr
      :value: 21



   .. py:attribute:: Assign
      :value: 22



   .. py:attribute:: MulAssign
      :value: 23



   .. py:attribute:: DivAssign
      :value: 24



   .. py:attribute:: RemAssign
      :value: 25



   .. py:attribute:: AddAssign
      :value: 26



   .. py:attribute:: SubAssign
      :value: 27



   .. py:attribute:: ShlAssign
      :value: 28



   .. py:attribute:: ShrAssign
      :value: 29



   .. py:attribute:: AndAssign
      :value: 30



   .. py:attribute:: XorAssign
      :value: 31



   .. py:attribute:: OrAssign
      :value: 32



   .. py:attribute:: Comma
      :value: 33



.. py:class:: StorageClass(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the storage class of a declaration


   .. py:attribute:: INVALID
      :value: 0



   .. py:attribute:: NONE
      :value: 1



   .. py:attribute:: EXTERN
      :value: 2



   .. py:attribute:: STATIC
      :value: 3



   .. py:attribute:: PRIVATEEXTERN
      :value: 4



   .. py:attribute:: OPENCLWORKGROUPLOCAL
      :value: 5



   .. py:attribute:: AUTO
      :value: 6



   .. py:attribute:: REGISTER
      :value: 7



.. py:class:: AvailabilityKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the availability of an entity.


   .. py:attribute:: AVAILABLE
      :value: 0



   .. py:attribute:: DEPRECATED
      :value: 1



   .. py:attribute:: NOT_AVAILABLE
      :value: 2



   .. py:attribute:: NOT_ACCESSIBLE
      :value: 3



.. py:class:: AccessSpecifier(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the access of a C++ class member


   .. py:attribute:: INVALID
      :value: 0



   .. py:attribute:: PUBLIC
      :value: 1



   .. py:attribute:: PROTECTED
      :value: 2



   .. py:attribute:: PRIVATE
      :value: 3



.. py:class:: TypeKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the kind of type.


   .. py:property:: spelling

      Retrieve the spelling of this TypeKind.



   .. py:attribute:: INVALID
      :value: 0



   .. py:attribute:: UNEXPOSED
      :value: 1



   .. py:attribute:: VOID
      :value: 2



   .. py:attribute:: BOOL
      :value: 3



   .. py:attribute:: CHAR_U
      :value: 4



   .. py:attribute:: UCHAR
      :value: 5



   .. py:attribute:: CHAR16
      :value: 6



   .. py:attribute:: CHAR32
      :value: 7



   .. py:attribute:: USHORT
      :value: 8



   .. py:attribute:: UINT
      :value: 9



   .. py:attribute:: ULONG
      :value: 10



   .. py:attribute:: ULONGLONG
      :value: 11



   .. py:attribute:: UINT128
      :value: 12



   .. py:attribute:: CHAR_S
      :value: 13



   .. py:attribute:: SCHAR
      :value: 14



   .. py:attribute:: WCHAR
      :value: 15



   .. py:attribute:: SHORT
      :value: 16



   .. py:attribute:: INT
      :value: 17



   .. py:attribute:: LONG
      :value: 18



   .. py:attribute:: LONGLONG
      :value: 19



   .. py:attribute:: INT128
      :value: 20



   .. py:attribute:: FLOAT
      :value: 21



   .. py:attribute:: DOUBLE
      :value: 22



   .. py:attribute:: LONGDOUBLE
      :value: 23



   .. py:attribute:: NULLPTR
      :value: 24



   .. py:attribute:: OVERLOAD
      :value: 25



   .. py:attribute:: DEPENDENT
      :value: 26



   .. py:attribute:: OBJCID
      :value: 27



   .. py:attribute:: OBJCCLASS
      :value: 28



   .. py:attribute:: OBJCSEL
      :value: 29



   .. py:attribute:: FLOAT128
      :value: 30



   .. py:attribute:: HALF
      :value: 31



   .. py:attribute:: FLOAT16
      :value: 32



   .. py:attribute:: SHORTACCUM
      :value: 33



   .. py:attribute:: ACCUM
      :value: 34



   .. py:attribute:: LONGACCUM
      :value: 35



   .. py:attribute:: USHORTACCUM
      :value: 36



   .. py:attribute:: UACCUM
      :value: 37



   .. py:attribute:: ULONGACCUM
      :value: 38



   .. py:attribute:: BFLOAT16
      :value: 39



   .. py:attribute:: IBM128
      :value: 40



   .. py:attribute:: FIRSTBUILTIN


   .. py:attribute:: LASTBUILTIN


   .. py:attribute:: COMPLEX
      :value: 100



   .. py:attribute:: POINTER
      :value: 101



   .. py:attribute:: BLOCKPOINTER
      :value: 102



   .. py:attribute:: LVALUEREFERENCE
      :value: 103



   .. py:attribute:: RVALUEREFERENCE
      :value: 104



   .. py:attribute:: RECORD
      :value: 105



   .. py:attribute:: ENUM
      :value: 106



   .. py:attribute:: TYPEDEF
      :value: 107



   .. py:attribute:: OBJCINTERFACE
      :value: 108



   .. py:attribute:: OBJCOBJECTPOINTER
      :value: 109



   .. py:attribute:: FUNCTIONNOPROTO
      :value: 110



   .. py:attribute:: FUNCTIONPROTO
      :value: 111



   .. py:attribute:: CONSTANTARRAY
      :value: 112



   .. py:attribute:: VECTOR
      :value: 113



   .. py:attribute:: INCOMPLETEARRAY
      :value: 114



   .. py:attribute:: VARIABLEARRAY
      :value: 115



   .. py:attribute:: DEPENDENTSIZEDARRAY
      :value: 116



   .. py:attribute:: MEMBERPOINTER
      :value: 117



   .. py:attribute:: AUTO
      :value: 118



   .. py:attribute:: ELABORATED
      :value: 119



   .. py:attribute:: PIPE
      :value: 120



   .. py:attribute:: OCLIMAGE1DRO
      :value: 121



   .. py:attribute:: OCLIMAGE1DARRAYRO
      :value: 122



   .. py:attribute:: OCLIMAGE1DBUFFERRO
      :value: 123



   .. py:attribute:: OCLIMAGE2DRO
      :value: 124



   .. py:attribute:: OCLIMAGE2DARRAYRO
      :value: 125



   .. py:attribute:: OCLIMAGE2DDEPTHRO
      :value: 126



   .. py:attribute:: OCLIMAGE2DARRAYDEPTHRO
      :value: 127



   .. py:attribute:: OCLIMAGE2DMSAARO
      :value: 128



   .. py:attribute:: OCLIMAGE2DARRAYMSAARO
      :value: 129



   .. py:attribute:: OCLIMAGE2DMSAADEPTHRO
      :value: 130



   .. py:attribute:: OCLIMAGE2DARRAYMSAADEPTHRO
      :value: 131



   .. py:attribute:: OCLIMAGE3DRO
      :value: 132



   .. py:attribute:: OCLIMAGE1DWO
      :value: 133



   .. py:attribute:: OCLIMAGE1DARRAYWO
      :value: 134



   .. py:attribute:: OCLIMAGE1DBUFFERWO
      :value: 135



   .. py:attribute:: OCLIMAGE2DWO
      :value: 136



   .. py:attribute:: OCLIMAGE2DARRAYWO
      :value: 137



   .. py:attribute:: OCLIMAGE2DDEPTHWO
      :value: 138



   .. py:attribute:: OCLIMAGE2DARRAYDEPTHWO
      :value: 139



   .. py:attribute:: OCLIMAGE2DMSAAWO
      :value: 140



   .. py:attribute:: OCLIMAGE2DARRAYMSAAWO
      :value: 141



   .. py:attribute:: OCLIMAGE2DMSAADEPTHWO
      :value: 142



   .. py:attribute:: OCLIMAGE2DARRAYMSAADEPTHWO
      :value: 143



   .. py:attribute:: OCLIMAGE3DWO
      :value: 144



   .. py:attribute:: OCLIMAGE1DRW
      :value: 145



   .. py:attribute:: OCLIMAGE1DARRAYRW
      :value: 146



   .. py:attribute:: OCLIMAGE1DBUFFERRW
      :value: 147



   .. py:attribute:: OCLIMAGE2DRW
      :value: 148



   .. py:attribute:: OCLIMAGE2DARRAYRW
      :value: 149



   .. py:attribute:: OCLIMAGE2DDEPTHRW
      :value: 150



   .. py:attribute:: OCLIMAGE2DARRAYDEPTHRW
      :value: 151



   .. py:attribute:: OCLIMAGE2DMSAARW
      :value: 152



   .. py:attribute:: OCLIMAGE2DARRAYMSAARW
      :value: 153



   .. py:attribute:: OCLIMAGE2DMSAADEPTHRW
      :value: 154



   .. py:attribute:: OCLIMAGE2DARRAYMSAADEPTHRW
      :value: 155



   .. py:attribute:: OCLIMAGE3DRW
      :value: 156



   .. py:attribute:: OCLSAMPLER
      :value: 157



   .. py:attribute:: OCLEVENT
      :value: 158



   .. py:attribute:: OCLQUEUE
      :value: 159



   .. py:attribute:: OCLRESERVEID
      :value: 160



   .. py:attribute:: OBJCOBJECT
      :value: 161



   .. py:attribute:: OBJCTYPEPARAM
      :value: 162



   .. py:attribute:: ATTRIBUTED
      :value: 163



   .. py:attribute:: OCLINTELSUBGROUPAVCMCEPAYLOAD
      :value: 164



   .. py:attribute:: OCLINTELSUBGROUPAVCIMEPAYLOAD
      :value: 165



   .. py:attribute:: OCLINTELSUBGROUPAVCREFPAYLOAD
      :value: 166



   .. py:attribute:: OCLINTELSUBGROUPAVCSICPAYLOAD
      :value: 167



   .. py:attribute:: OCLINTELSUBGROUPAVCMCERESULT
      :value: 168



   .. py:attribute:: OCLINTELSUBGROUPAVCIMERESULT
      :value: 169



   .. py:attribute:: OCLINTELSUBGROUPAVCREFRESULT
      :value: 170



   .. py:attribute:: OCLINTELSUBGROUPAVCSICRESULT
      :value: 171



   .. py:attribute:: OCLINTELSUBGROUPAVCIMERESULTSINGLEREFERENCESTREAMOUT
      :value: 172



   .. py:attribute:: OCLINTELSUBGROUPAVCIMERESULTSDUALREFERENCESTREAMOUT
      :value: 173



   .. py:attribute:: OCLINTELSUBGROUPAVCIMERESULTSSINGLEREFERENCESTREAMIN
      :value: 174



   .. py:attribute:: OCLINTELSUBGROUPAVCIMEDUALREFERENCESTREAMIN
      :value: 175



   .. py:attribute:: EXTVECTOR
      :value: 176



   .. py:attribute:: ATOMIC
      :value: 177



   .. py:attribute:: BTFTAGATTRIBUTED
      :value: 178



   .. py:attribute:: HLSLRESOURCE
      :value: 179



   .. py:attribute:: HLSLATTRIBUTEDRESOURCE
      :value: 180



   .. py:attribute:: HLSLINLINESPIRV
      :value: 181



.. py:class:: RefQualifierKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes a specific ref-qualifier of a type.


   .. py:attribute:: NONE
      :value: 0



   .. py:attribute:: LVALUE
      :value: 1



   .. py:attribute:: RVALUE
      :value: 2



.. py:class:: LinkageKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the kind of linkage of a cursor.


   .. py:attribute:: INVALID
      :value: 0



   .. py:attribute:: NO_LINKAGE
      :value: 1



   .. py:attribute:: INTERNAL
      :value: 2



   .. py:attribute:: UNIQUE_EXTERNAL
      :value: 3



   .. py:attribute:: EXTERNAL
      :value: 4



.. py:class:: TLSKind(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   Describes the kind of thread-local storage (TLS) of a cursor.


   .. py:attribute:: NONE
      :value: 0



   .. py:attribute:: DYNAMIC
      :value: 1



   .. py:attribute:: STATIC
      :value: 2



.. py:class:: Type

   Bases: :py:obj:`ctypes.Structure`


   The type of an element in the abstract syntax tree.


   .. py:property:: kind
      :type: TypeKind


      Return the kind of this type.



   .. py:method:: argument_types() -> NoSliceSequence[Type]

      Retrieve a container for the non-variadic arguments for this type.

      The returned object is iterable and indexable. Each item in the
      container is a Type instance.



   .. py:property:: element_type
      :type: Type


      Retrieve the Type of elements within this Type.

      If accessed on a type that is not an array, complex, or vector type, an
      exception will be raised.



   .. py:property:: element_count
      :type: int


      Retrieve the number of elements in this type.

      Returns an int.

      If the Type is not an array or vector, this raises.



   .. py:property:: translation_unit
      :type: TranslationUnit


      The TranslationUnit to which this Type is associated.



   .. py:method:: from_result(res: Type, arg: Cursor | Type) -> Type
      :staticmethod:



   .. py:method:: get_num_template_arguments() -> int


   .. py:method:: get_template_argument_type(num: int) -> Type


   .. py:method:: get_canonical() -> Type

      Return the canonical type for a Type.

      Clang's type system explicitly models typedefs and all the
      ways a specific type can be represented.  The canonical type
      is the underlying type with all the "sugar" removed.  For
      example, if 'T' is a typedef for 'int', the canonical type for
      'T' would be 'int'.



   .. py:method:: get_fully_qualified_name(policy: PrintingPolicy, with_global_ns_prefix: bool = False) -> str

      Get the fully qualified name for a type.

      This includes full qualification of all template parameters.

      policy - This PrintingPolicy can further refine the type formatting
      with_global_ns_prefix - If true, prepend '::' to qualified names



   .. py:method:: is_const_qualified() -> bool

      Determine whether a Type has the "const" qualifier set.

      This does not look through typedefs that may have added "const"
      at a different level.



   .. py:method:: is_volatile_qualified() -> bool

      Determine whether a Type has the "volatile" qualifier set.

      This does not look through typedefs that may have added "volatile"
      at a different level.



   .. py:method:: is_restrict_qualified() -> bool

      Determine whether a Type has the "restrict" qualifier set.

      This does not look through typedefs that may have added "restrict" at
      a different level.



   .. py:method:: is_function_variadic() -> bool

      Determine whether this function Type is a variadic function type.



   .. py:method:: get_address_space() -> int


   .. py:method:: get_typedef_name() -> str


   .. py:method:: is_pod() -> bool

      Determine whether this Type represents plain old data (POD).



   .. py:method:: get_pointee() -> Type

      For pointer types, returns the type of the pointee.



   .. py:method:: get_declaration() -> Cursor

      Return the cursor for the declaration of the given type.



   .. py:method:: get_result() -> Type

      Retrieve the result type associated with a function type.



   .. py:method:: get_array_element_type() -> Type

      Retrieve the type of the elements of the array type.



   .. py:method:: get_array_size() -> int

      Retrieve the size of the constant array.



   .. py:method:: get_class_type() -> Type

      Retrieve the class type of the member pointer type.



   .. py:method:: get_named_type() -> Type

      Retrieve the type named by the qualified-id.



   .. py:method:: get_align() -> int

      Retrieve the alignment of the record.



   .. py:method:: get_size() -> int

      Retrieve the size of the record.



   .. py:method:: get_offset(fieldname: Union[str, bytes]) -> int

      Retrieve the offset of a field in the record.



   .. py:method:: get_ref_qualifier() -> RefQualifierKind

      Retrieve the ref-qualifier of the type.



   .. py:method:: get_fields() -> Iterator[Cursor]

      Return an iterator for accessing the fields of this type.



   .. py:method:: get_bases() -> Iterator[Cursor]

      Return an iterator for accessing the base classes of this type.



   .. py:method:: get_methods() -> Iterator[Cursor]

      Return an iterator for accessing the methods of this type.



   .. py:method:: get_exception_specification_kind() -> ExceptionSpecificationKind

      Return the kind of the exception specification; a value from
      the ExceptionSpecificationKind enumeration.



   .. py:property:: spelling
      :type: str


      Retrieve the spelling of this Type.



   .. py:method:: pretty_printed(policy: PrintingPolicy) -> str

      Pretty-prints this Type with the given PrintingPolicy



.. py:class:: CodeCompletionResults(ptr: ctypes._Pointer[CCRStructure])

   Bases: :py:obj:`ClangObject`


   A helper for Clang objects. This class helps act as an intermediary for
   the ctypes library and the Clang CIndex library.


   .. py:method:: from_param() -> ctypes._Pointer[CCRStructure]


   .. py:property:: results
      :type: CCRStructure



   .. py:property:: diagnostics
      :type: NoSliceSequence[Diagnostic]



.. py:class:: Index(obj)

   Bases: :py:obj:`ClangObject`


   The Index type provides the primary interface to the Clang CIndex library,
   primarily by providing an interface for reading and parsing translation
   units.


   .. py:method:: create(excludeDecls=False)
      :staticmethod:


      Create a new Index.
      Parameters:
      excludeDecls -- Exclude local declarations from translation units.



   .. py:method:: read(path)

      Load a TranslationUnit from the given AST file.



   .. py:method:: parse(path, args=None, unsaved_files=None, options=0)

      Load the translation unit from the given source code file by running
      clang and generating the AST before loading. Additional command line
      parameters can be passed to clang via the args parameter.

      In-memory contents for files can be provided by passing a list of pairs
      to as unsaved_files, the first item should be the filenames to be mapped
      and the second should be the contents to be substituted for the
      file. The contents may be passed as strings or file objects.

      If an error was encountered during parsing, a TranslationUnitLoadError
      will be raised.



.. py:class:: TranslationUnit(ptr: CObjP, index: Index)

   Bases: :py:obj:`ClangObject`


   Represents a source code translation unit.

   This is one of the main types in the API. Any time you wish to interact
   with Clang's representation of a source file, you typically start with a
   translation unit.


   .. py:attribute:: PARSE_NONE
      :value: 0



   .. py:attribute:: PARSE_DETAILED_PROCESSING_RECORD
      :value: 1



   .. py:attribute:: PARSE_INCOMPLETE
      :value: 2



   .. py:attribute:: PARSE_PRECOMPILED_PREAMBLE
      :value: 4



   .. py:attribute:: PARSE_CACHE_COMPLETION_RESULTS
      :value: 8



   .. py:attribute:: PARSE_SKIP_FUNCTION_BODIES
      :value: 64



   .. py:attribute:: PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION
      :value: 128



   .. py:method:: process_unsaved_files(unsaved_files: list[InMemoryFile]) -> ctypes.Array[UnsavedFile] | None
      :staticmethod:



   .. py:method:: from_source(filename: StrBytesPath | None, args: list[Union[str, bytes]] | None = None, unsaved_files: list[InMemoryFile] | None = None, options: int = 0, index: Index | None = None) -> TranslationUnit
      :classmethod:


      Create a TranslationUnit by parsing source.

      This is capable of processing source code both from files on the
      filesystem as well as in-memory contents.

      Command-line arguments that would be passed to clang are specified as
      a list via args. These can be used to specify include paths, warnings,
      etc. e.g. ["-Wall", "-I/path/to/include"].

      In-memory file content can be provided via unsaved_files. This is a
      list of 2-tuples. The first element is the filename (str, bytes or
      PathLike). The second element defines the content. Content can be
      provided as str or bytes source code, or as file objects (anything with
      a read() method). If a file object is being used, content will be read
      until EOF and the read cursor will not be reset to its original
      position.

      options is a bitwise or of TranslationUnit.PARSE_XXX flags which will
      control parsing behavior.

      index is an Index instance to utilize. If not provided, a new Index
      will be created for this TranslationUnit.

      To parse source from the filesystem, the filename of the file to parse
      is specified by the filename argument. Or, filename could be None and
      the args list would contain the filename(s) to parse.

      To parse source from an in-memory buffer, set filename to the virtual
      filename you wish to associate with this source (e.g. "test.c"). The
      contents of that file are then provided in unsaved_files.

      If an error occurs, a TranslationUnitLoadError is raised.

      Please note that a TranslationUnit with parser errors may be returned.
      It is the caller's responsibility to check tu.diagnostics for errors.

      Also note that Clang infers the source language from the extension of
      the input filename. If you pass in source code containing a C++ class
      declaration with the filename "test.c" parsing will fail.



   .. py:method:: from_ast_file(filename: StrBytesPath, index: Index | None = None) -> TranslationUnit
      :classmethod:


      Create a TranslationUnit instance from a saved AST file.

      A previously-saved AST file (provided with -emit-ast or
      TranslationUnit.save()) is loaded from the filename specified.

      If the file cannot be loaded, a TranslationUnitLoadError will be
      raised.

      index is optional and is the Index instance to use. If not provided,
      a default Index will be created.

      filename can be str or PathLike.



   .. py:attribute:: index


   .. py:property:: cursor
      :type: Cursor | None


      Retrieve the cursor that represents the given translation unit.



   .. py:property:: spelling
      :type: str


      Get the original translation unit source file name.



   .. py:method:: get_includes() -> Iterator[FileInclusion]

      Return an iterable sequence of FileInclusion objects that describe the
      sequence of inclusions in a translation unit. The first object in
      this sequence is always the input file. Note that this method will not
      recursively iterate over header files included through precompiled
      headers.



   .. py:method:: get_file(filename: StrBytesPath) -> File

      Obtain a File from this translation unit.



   .. py:method:: get_location(filename: StrBytesPath, position: int | tuple[int, int]) -> SourceLocation

      Obtain a SourceLocation for a file in this translation unit.

      The position can be specified by passing:

        - Integer file offset. Initial file offset is 0.
        - 2-tuple of (line number, column number). Initial file position is
          (0, 0)



   .. py:method:: get_extent(filename: StrBytesPath, locations: Sequence[SourceLocation] | Sequence[int] | Sequence[Sequence[int]]) -> SourceRange

      Obtain a SourceRange from this translation unit.

      The bounds of the SourceRange must ultimately be defined by a start and
      end SourceLocation. For the locations argument, you can pass:

        - 2 SourceLocation instances in a 2-tuple or list.
        - 2 int file offsets via a 2-tuple or list.
        - 2 2-tuple or lists of (line, column) pairs in a 2-tuple or list.

      e.g.

      get_extent('foo.c', (5, 10))
      get_extent('foo.c', ((1, 1), (1, 15)))



   .. py:property:: diagnostics
      :type: NoSliceSequence[Diagnostic]


      Return an iterable (and indexable) object containing the diagnostics.



   .. py:method:: reparse(unsaved_files: list[InMemoryFile] | None = None, options: int = 0) -> None

      Reparse an already parsed translation unit.

      In-memory contents for files can be provided by passing a list of pairs
      as unsaved_files, the first items should be the filenames to be mapped
      and the second should be the contents to be substituted for the
      file. The contents may be passed as strings or file objects.



   .. py:method:: save(filename: StrBytesPath) -> None

      Saves the TranslationUnit to a file.

      This is equivalent to passing -emit-ast to the clang frontend. The
      saved file can be loaded back into a TranslationUnit. Or, if it
      corresponds to a header, it can be used as a pre-compiled header file.

      If an error occurs while saving, a TranslationUnitSaveError is raised.
      If the error was TranslationUnitSaveError.ERROR_INVALID_TU, this means
      the constructed TranslationUnit was not valid at time of save. In this
      case, the reason(s) why should be available via
      TranslationUnit.diagnostics().

      filename -- The path to save the translation unit to (str or PathLike).



   .. py:method:: codeComplete(path: StrBytesPath, line: int, column: int, unsaved_files: list[InMemoryFile] | None = None, include_macros: bool = False, include_code_patterns: bool = False, include_brief_comments: bool = False) -> CodeCompletionResults | None

      Code complete in this translation unit.

      In-memory contents for files can be provided by passing a list of pairs
      as unsaved_files, the first items should be the filenames to be mapped
      and the second should be the contents to be substituted for the
      file. The contents may be passed as strings or file objects.



   .. py:method:: get_tokens(locations: tuple[SourceLocation, SourceLocation] | None = None, extent: SourceRange | None = None) -> Iterator[Token]

      Obtain tokens in this translation unit.

      This is a generator for Token instances. The caller specifies a range
      of source code to obtain tokens for. The range can be specified as a
      2-tuple of SourceLocation or as a SourceRange. If both are defined,
      behavior is undefined.



.. py:class:: File(obj)

   Bases: :py:obj:`ClangObject`


   The File class represents a particular source file that is part of a
   translation unit.


   .. py:method:: from_name(translation_unit, file_name)
      :staticmethod:


      Retrieve a file handle within the given translation unit.



   .. py:property:: name

      Return the complete file and path name of the file.



   .. py:property:: time

      Return the last modification time of the file.



   .. py:method:: from_result(res, arg)
      :staticmethod:



.. py:class:: CompileCommand(cmd, ccmds)

   Represents the compile command used to build a file


   .. py:attribute:: cmd


   .. py:attribute:: ccmds


   .. py:property:: directory

      Get the working directory for this CompileCommand



   .. py:property:: filename

      Get the working filename for this CompileCommand



   .. py:property:: arguments

      Get an iterable object providing each argument in the
      command line for the compiler invocation as a string.

      Invariant : the first argument is the compiler executable



.. py:class:: CompileCommands(ccmds)

   CompileCommands is an iterable object containing all CompileCommand
   that can be used for building a specific file.


   .. py:attribute:: ccmds


   .. py:method:: from_result(res)
      :staticmethod:



.. py:class:: CompilationDatabase(obj)

   Bases: :py:obj:`ClangObject`


   The CompilationDatabase is a wrapper class around
   clang::tooling::CompilationDatabase

   It enables querying how a specific source file can be built.


   .. py:method:: from_result(res)
      :staticmethod:



   .. py:method:: fromDirectory(buildDir)
      :staticmethod:


      Builds a CompilationDatabase from the database found in buildDir



   .. py:method:: getCompileCommands(filename)

      Get an iterable object providing all the CompileCommands available to
      build filename. Returns None if filename is not found in the database.



   .. py:method:: getAllCompileCommands()

      Get an iterable object providing all the CompileCommands available from
      the database.



.. py:class:: Token

   Bases: :py:obj:`ctypes.Structure`


   Represents a single token from the preprocessor.

   Tokens are effectively segments of source code. Source code is first parsed
   into tokens before being converted into the AST and Cursors.

   Tokens are obtained from parsed TranslationUnit instances. You currently
   can't create tokens manually.


   .. py:property:: spelling

      The spelling of this token.

      This is the textual representation of the token in source.



   .. py:property:: kind

      Obtain the TokenKind of the current token.



   .. py:property:: location

      The SourceLocation this Token occurs at.



   .. py:property:: extent

      The SourceRange this Token occupies.



   .. py:property:: cursor

      The Cursor this Token corresponds to.



.. py:class:: PrintingPolicyProperty(*args, **kwds)

   Bases: :py:obj:`BaseEnumeration`


   A PrintingPolicyProperty identifies a property of a PrintingPolicy.


   .. py:attribute:: Indentation
      :value: 0



   .. py:attribute:: SuppressSpecifiers
      :value: 1



   .. py:attribute:: SuppressTagKeyword
      :value: 2



   .. py:attribute:: IncludeTagDefinition
      :value: 3



   .. py:attribute:: SuppressScope
      :value: 4



   .. py:attribute:: SuppressUnwrittenScope
      :value: 5



   .. py:attribute:: SuppressInitializers
      :value: 6



   .. py:attribute:: ConstantArraySizeAsWritten
      :value: 7



   .. py:attribute:: AnonymousTagLocations
      :value: 8



   .. py:attribute:: SuppressStrongLifetime
      :value: 9



   .. py:attribute:: SuppressLifetimeQualifiers
      :value: 10



   .. py:attribute:: SuppressTemplateArgsInCXXConstructors
      :value: 11



   .. py:attribute:: Bool
      :value: 12



   .. py:attribute:: Restrict
      :value: 13



   .. py:attribute:: Alignof
      :value: 14



   .. py:attribute:: UnderscoreAlignof
      :value: 15



   .. py:attribute:: UseVoidForZeroParams
      :value: 16



   .. py:attribute:: TerseOutput
      :value: 17



   .. py:attribute:: PolishForDeclaration
      :value: 18



   .. py:attribute:: Half
      :value: 19



   .. py:attribute:: MSWChar
      :value: 20



   .. py:attribute:: IncludeNewlines
      :value: 21



   .. py:attribute:: MSVCFormatting
      :value: 22



   .. py:attribute:: ConstantsAsWritten
      :value: 23



   .. py:attribute:: SuppressImplicitBase
      :value: 24



   .. py:attribute:: FullyQualifiedName
      :value: 25



.. py:class:: PrintingPolicy(ptr)

   Bases: :py:obj:`ClangObject`


   The PrintingPolicy is a wrapper class around clang::PrintingPolicy

   It allows specifying how declarations, expressions, and types should be
   pretty-printed.


   .. py:method:: create(cursor)
      :staticmethod:


      Creates a new PrintingPolicy
      Parameters:
      cursor -- Any cursor for a translation unit.



   .. py:method:: get_property(property)

      Get a property value for the given printing policy.



   .. py:method:: set_property(property, value)

      Set a property value for the given printing policy.



.. py:class:: Config

   .. py:attribute:: library_path
      :type:  str | None


   .. py:attribute:: library_file
      :type:  str | None


   .. py:attribute:: compatibility_check
      :value: True



   .. py:attribute:: loaded
      :value: False



   .. py:method:: set_library_path(path: StrPath) -> None
      :staticmethod:


      Set the path in which to search for libclang



   .. py:method:: set_library_file(filename: StrPath) -> None
      :staticmethod:


      Set the exact location of libclang



   .. py:method:: set_compatibility_check(check_status: bool) -> None
      :staticmethod:


      Perform compatibility check when loading libclang

      The python bindings are only tested and evaluated with the version of
      libclang they are provided with. To ensure correct behavior a (limited)
      compatibility check is performed when loading the bindings. This check
      will throw an exception, as soon as it fails.

      In case these bindings are used with an older version of libclang, parts
      that have been stable between releases may still work. Users of the
      python bindings can disable the compatibility check. This will cause
      the python bindings to load, even though they are written for a newer
      version of libclang. Failures now arise if unsupported or incompatible
      features are accessed. The user is required to test themselves if the
      features they are using are available and compatible between different
      libclang versions.



   .. py:method:: lib() -> ctypes.CDLL


   .. py:method:: get_filename() -> str


   .. py:method:: get_cindex_library() -> ctypes.CDLL


   .. py:method:: get_clang_version() -> str

      Returns the libclang version string used by the bindings



