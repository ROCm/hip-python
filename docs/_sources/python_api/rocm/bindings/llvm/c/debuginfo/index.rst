rocm.bindings.llvm.c.debuginfo
==============================

.. py:module:: rocm.bindings.llvm.c.debuginfo


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.debuginfo.LLVMMDStringMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMConstantAsMetadataMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMLocalAsMetadataMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDistinctMDOperandPlaceholderMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMMDTupleMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDILocationMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIExpressionMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIGlobalVariableExpressionMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMGenericDINodeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDISubrangeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIEnumeratorMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIBasicTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIDerivedTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDICompositeTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDISubroutineTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIFileMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDICompileUnitMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDISubprogramMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDILexicalBlockMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDILexicalBlockFileMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDINamespaceMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIModuleMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDITemplateTypeParameterMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDITemplateValueParameterMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIGlobalVariableMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDILocalVariableMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDILabelMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIObjCPropertyMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIImportedEntityMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIMacroMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIMacroFileMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDICommonBlockMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIStringTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIGenericSubrangeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIArgListMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIAssignIDMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDISubrangeTypeMetadataKind
   rocm.bindings.llvm.c.debuginfo.LLVMDIFixedPointTypeMetadataKind


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.debuginfo.LLVMDIFlags
   rocm.bindings.llvm.c.debuginfo.LLVMDWARFSourceLanguage
   rocm.bindings.llvm.c.debuginfo.LLVMDWARFEmissionKind
   rocm.bindings.llvm.c.debuginfo.LLVMChecksumKind
   rocm.bindings.llvm.c.debuginfo.LLVMDWARFMacinfoRecordType


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.debuginfo.has_symbol
   rocm.bindings.llvm.c.debuginfo.LLVMDebugMetadataVersion
   rocm.bindings.llvm.c.debuginfo.LLVMGetModuleDebugMetadataVersion
   rocm.bindings.llvm.c.debuginfo.LLVMStripModuleDebugInfo
   rocm.bindings.llvm.c.debuginfo.LLVMCreateDIBuilderDisallowUnresolved
   rocm.bindings.llvm.c.debuginfo.LLVMCreateDIBuilder
   rocm.bindings.llvm.c.debuginfo.LLVMDisposeDIBuilder
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderFinalize
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderFinalizeSubprogram
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateCompileUnit
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateFile
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateFileWithChecksum
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateModule
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateNameSpace
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateFunction
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateLexicalBlock
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateLexicalBlockFile
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateImportedModuleFromNamespace
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateImportedModuleFromAlias
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateImportedModuleFromModule
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateImportedDeclaration
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateDebugLocation
   rocm.bindings.llvm.c.debuginfo.LLVMDILocationGetLine
   rocm.bindings.llvm.c.debuginfo.LLVMDILocationGetColumn
   rocm.bindings.llvm.c.debuginfo.LLVMDILocationGetScope
   rocm.bindings.llvm.c.debuginfo.LLVMDILocationGetInlinedAt
   rocm.bindings.llvm.c.debuginfo.LLVMDIScopeGetFile
   rocm.bindings.llvm.c.debuginfo.LLVMDIFileGetDirectory
   rocm.bindings.llvm.c.debuginfo.LLVMDIFileGetFilename
   rocm.bindings.llvm.c.debuginfo.LLVMDIFileGetSource
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderGetOrCreateTypeArray
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateSubroutineType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateMacro
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateTempMacroFile
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateEnumerator
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateEnumerationType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateUnionType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateArrayType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateSetType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateSubrangeType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateDynamicArrayType
   rocm.bindings.llvm.c.debuginfo.LLVMReplaceArrays
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateVectorType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateUnspecifiedType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateBasicType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreatePointerType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateStructType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateMemberType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateStaticMemberType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateMemberPointerType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateObjCIVar
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateObjCProperty
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateObjectPointerType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateQualifiedType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateReferenceType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateNullPtrType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateTypedef
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateInheritance
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateForwardDecl
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateReplaceableCompositeType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateBitFieldMemberType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateClassType
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateArtificialType
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetName
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetSizeInBits
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetOffsetInBits
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetAlignInBits
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetLine
   rocm.bindings.llvm.c.debuginfo.LLVMDITypeGetFlags
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderGetOrCreateSubrange
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderGetOrCreateArray
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateExpression
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateConstantValueExpression
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateGlobalVariableExpression
   rocm.bindings.llvm.c.debuginfo.LLVMGetDINodeTag
   rocm.bindings.llvm.c.debuginfo.LLVMDIGlobalVariableExpressionGetVariable
   rocm.bindings.llvm.c.debuginfo.LLVMDIGlobalVariableExpressionGetExpression
   rocm.bindings.llvm.c.debuginfo.LLVMDIVariableGetFile
   rocm.bindings.llvm.c.debuginfo.LLVMDIVariableGetScope
   rocm.bindings.llvm.c.debuginfo.LLVMDIVariableGetLine
   rocm.bindings.llvm.c.debuginfo.LLVMTemporaryMDNode
   rocm.bindings.llvm.c.debuginfo.LLVMDisposeTemporaryMDNode
   rocm.bindings.llvm.c.debuginfo.LLVMMetadataReplaceAllUsesWith
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateTempGlobalVariableFwdDecl
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertDeclareRecordBefore
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertDeclareRecordAtEnd
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertDbgValueRecordBefore
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertDbgValueRecordAtEnd
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateAutoVariable
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateParameterVariable
   rocm.bindings.llvm.c.debuginfo.LLVMGetSubprogram
   rocm.bindings.llvm.c.debuginfo.LLVMSetSubprogram
   rocm.bindings.llvm.c.debuginfo.LLVMDISubprogramGetLine
   rocm.bindings.llvm.c.debuginfo.LLVMDISubprogramReplaceType
   rocm.bindings.llvm.c.debuginfo.LLVMInstructionGetDebugLoc
   rocm.bindings.llvm.c.debuginfo.LLVMInstructionSetDebugLoc
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderCreateLabel
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertLabelBefore
   rocm.bindings.llvm.c.debuginfo.LLVMDIBuilderInsertLabelAtEnd
   rocm.bindings.llvm.c.debuginfo.LLVMGetMetadataKind


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMDIFlags

   Bases: :py:obj:`enum.IntEnum`


   Debug info flags.
       


   .. py:attribute:: LLVMDIFlagZero
      :type:  int


   .. py:attribute:: LLVMDIFlagPrivate
      :type:  int


   .. py:attribute:: LLVMDIFlagProtected
      :type:  int


   .. py:attribute:: LLVMDIFlagPublic
      :type:  int


   .. py:attribute:: LLVMDIFlagFwdDecl
      :type:  int


   .. py:attribute:: LLVMDIFlagAppleBlock
      :type:  int


   .. py:attribute:: LLVMDIFlagReservedBit4
      :type:  int


   .. py:attribute:: LLVMDIFlagVirtual
      :type:  int


   .. py:attribute:: LLVMDIFlagArtificial
      :type:  int


   .. py:attribute:: LLVMDIFlagExplicit
      :type:  int


   .. py:attribute:: LLVMDIFlagPrototyped
      :type:  int


   .. py:attribute:: LLVMDIFlagObjcClassComplete
      :type:  int


   .. py:attribute:: LLVMDIFlagObjectPointer
      :type:  int


   .. py:attribute:: LLVMDIFlagVector
      :type:  int


   .. py:attribute:: LLVMDIFlagStaticMember
      :type:  int


   .. py:attribute:: LLVMDIFlagLValueReference
      :type:  int


   .. py:attribute:: LLVMDIFlagRValueReference
      :type:  int


   .. py:attribute:: LLVMDIFlagReserved
      :type:  int


   .. py:attribute:: LLVMDIFlagSingleInheritance
      :type:  int


   .. py:attribute:: LLVMDIFlagMultipleInheritance
      :type:  int


   .. py:attribute:: LLVMDIFlagVirtualInheritance
      :type:  int


   .. py:attribute:: LLVMDIFlagIntroducedVirtual
      :type:  int


   .. py:attribute:: LLVMDIFlagBitField
      :type:  int


   .. py:attribute:: LLVMDIFlagNoReturn
      :type:  int


   .. py:attribute:: LLVMDIFlagTypePassByValue
      :type:  int


   .. py:attribute:: LLVMDIFlagTypePassByReference
      :type:  int


   .. py:attribute:: LLVMDIFlagEnumClass
      :type:  int


   .. py:attribute:: LLVMDIFlagFixedEnum
      :type:  int


   .. py:attribute:: LLVMDIFlagThunk
      :type:  int


   .. py:attribute:: LLVMDIFlagNonTrivial
      :type:  int


   .. py:attribute:: LLVMDIFlagBigEndian
      :type:  int


   .. py:attribute:: LLVMDIFlagLittleEndian
      :type:  int


   .. py:attribute:: LLVMDIFlagIndirectVirtualBase
      :type:  int


   .. py:attribute:: LLVMDIFlagAccessibility
      :type:  int


   .. py:attribute:: LLVMDIFlagPtrToMemberRep
      :type:  int


.. py:class:: LLVMDWARFSourceLanguage

   Bases: :py:obj:`enum.IntEnum`


   Source languages known by DWARF.
       


   .. py:attribute:: LLVMDWARFSourceLanguageC89
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageAda83
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageCobol74
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageCobol85
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran77
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran90
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguagePascal83
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageModula2
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageJava
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC99
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageAda95
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran95
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguagePLI
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageObjC
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageObjC_plus_plus
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageUPC
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageD
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguagePython
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageOpenCL
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageGo
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageModula3
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageHaskell
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus_03
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus_11
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageOCaml
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageRust
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC11
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageSwift
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageJulia
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageDylan
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus_14
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran03
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran08
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageRenderScript
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageBLISS
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageKotlin
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageZig
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageCrystal
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus_17
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_plus_plus_20
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC17
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageFortran18
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageAda2005
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageAda2012
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageHIP
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageAssembly
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageC_sharp
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageMojo
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageGLSL
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageGLSL_ES
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageHLSL
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageOpenCL_CPP
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageCPP_for_OpenCL
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageSYCL
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageRuby
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageMove
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageHylo
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageMetal
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageMips_Assembler
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageGOOGLE_RenderScript
      :type:  int


   .. py:attribute:: LLVMDWARFSourceLanguageBORLAND_Delphi
      :type:  int


.. py:class:: LLVMDWARFEmissionKind

   Bases: :py:obj:`enum.IntEnum`


   The amount of debug information to emit.
       


   .. py:attribute:: LLVMDWARFEmissionNone
      :type:  int


   .. py:attribute:: LLVMDWARFEmissionFull
      :type:  int


   .. py:attribute:: LLVMDWARFEmissionLineTablesOnly
      :type:  int


.. py:data:: LLVMMDStringMetadataKind
   :type:  int

.. py:data:: LLVMConstantAsMetadataMetadataKind
   :type:  int

.. py:data:: LLVMLocalAsMetadataMetadataKind
   :type:  int

.. py:data:: LLVMDistinctMDOperandPlaceholderMetadataKind
   :type:  int

.. py:data:: LLVMMDTupleMetadataKind
   :type:  int

.. py:data:: LLVMDILocationMetadataKind
   :type:  int

.. py:data:: LLVMDIExpressionMetadataKind
   :type:  int

.. py:data:: LLVMDIGlobalVariableExpressionMetadataKind
   :type:  int

.. py:data:: LLVMGenericDINodeMetadataKind
   :type:  int

.. py:data:: LLVMDISubrangeMetadataKind
   :type:  int

.. py:data:: LLVMDIEnumeratorMetadataKind
   :type:  int

.. py:data:: LLVMDIBasicTypeMetadataKind
   :type:  int

.. py:data:: LLVMDIDerivedTypeMetadataKind
   :type:  int

.. py:data:: LLVMDICompositeTypeMetadataKind
   :type:  int

.. py:data:: LLVMDISubroutineTypeMetadataKind
   :type:  int

.. py:data:: LLVMDIFileMetadataKind
   :type:  int

.. py:data:: LLVMDICompileUnitMetadataKind
   :type:  int

.. py:data:: LLVMDISubprogramMetadataKind
   :type:  int

.. py:data:: LLVMDILexicalBlockMetadataKind
   :type:  int

.. py:data:: LLVMDILexicalBlockFileMetadataKind
   :type:  int

.. py:data:: LLVMDINamespaceMetadataKind
   :type:  int

.. py:data:: LLVMDIModuleMetadataKind
   :type:  int

.. py:data:: LLVMDITemplateTypeParameterMetadataKind
   :type:  int

.. py:data:: LLVMDITemplateValueParameterMetadataKind
   :type:  int

.. py:data:: LLVMDIGlobalVariableMetadataKind
   :type:  int

.. py:data:: LLVMDILocalVariableMetadataKind
   :type:  int

.. py:data:: LLVMDILabelMetadataKind
   :type:  int

.. py:data:: LLVMDIObjCPropertyMetadataKind
   :type:  int

.. py:data:: LLVMDIImportedEntityMetadataKind
   :type:  int

.. py:data:: LLVMDIMacroMetadataKind
   :type:  int

.. py:data:: LLVMDIMacroFileMetadataKind
   :type:  int

.. py:data:: LLVMDICommonBlockMetadataKind
   :type:  int

.. py:data:: LLVMDIStringTypeMetadataKind
   :type:  int

.. py:data:: LLVMDIGenericSubrangeMetadataKind
   :type:  int

.. py:data:: LLVMDIArgListMetadataKind
   :type:  int

.. py:data:: LLVMDIAssignIDMetadataKind
   :type:  int

.. py:data:: LLVMDISubrangeTypeMetadataKind
   :type:  int

.. py:data:: LLVMDIFixedPointTypeMetadataKind
   :type:  int

.. py:class:: LLVMChecksumKind

   Bases: :py:obj:`enum.IntEnum`


   The kind of checksum to emit.
       


   .. py:attribute:: CSK_MD5
      :type:  int


   .. py:attribute:: CSK_SHA1
      :type:  int


   .. py:attribute:: CSK_SHA256
      :type:  int


.. py:class:: LLVMDWARFMacinfoRecordType

   Bases: :py:obj:`enum.IntEnum`


   Describes the kind of macro declaration used for LLVMDIBuilderCreateMacro.

   See:
       llvm::dwarf::MacinfoRecordType

   Note:
       Values are from DW_MACINFO_* constants in the DWARF specification.


   .. py:attribute:: LLVMDWARFMacinfoRecordTypeDefine
      :type:  int


   .. py:attribute:: LLVMDWARFMacinfoRecordTypeMacro
      :type:  int


   .. py:attribute:: LLVMDWARFMacinfoRecordTypeStartFile
      :type:  int


   .. py:attribute:: LLVMDWARFMacinfoRecordTypeEndFile
      :type:  int


   .. py:attribute:: LLVMDWARFMacinfoRecordTypeVendorExt
      :type:  int


.. py:function:: LLVMDebugMetadataVersion()

   The current debug metadata version number.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDebugMetadataVersion()


.. py:function:: LLVMGetModuleDebugMetadataVersion(Module)

   The version of debug metadata that's present in the provided ``Module.``

   Args:
       Module (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMGetModuleDebugMetadataVersion(LLVMModuleRef Module)


.. py:function:: LLVMStripModuleDebugInfo(Module)

   Strip debug info in the module if it exists.

   To do this, we remove all calls to the debugger intrinsics and any named
   metadata for debugging. We also remove debug locations for instructions.
   Return true if module is modified.

   Args:
       Module (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMBool LLVMStripModuleDebugInfo(LLVMModuleRef Module)


.. py:function:: LLVMCreateDIBuilderDisallowUnresolved(M)

   Construct a builder for a module, and do not allow for unresolved nodes
   attached to the module.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDIBuilderRef LLVMCreateDIBuilderDisallowUnresolved(LLVMModuleRef M)


.. py:function:: LLVMCreateDIBuilder(M)

   Construct a builder for a module and collect unresolved nodes attached to the module in order to resolve cycles during a call to ``LLVMDIBuilderFinalize.``

   Construct a builder for a module and collect unresolved nodes attached
   to the module in order to resolve cycles during a call to
   ``LLVMDIBuilderFinalize.``

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDIBuilderRef LLVMCreateDIBuilder(LLVMModuleRef M)


.. py:function:: LLVMDisposeDIBuilder(Builder)

   Deallocates the ``DIBuilder`` and everything it owns.

   Note:
       You must call ``LLVMDIBuilderFinalize`` before this

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeDIBuilder(LLVMDIBuilderRef Builder)


.. py:function:: LLVMDIBuilderFinalize(Builder)

   Construct any deferred debug info descriptors.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDIBuilderFinalize(LLVMDIBuilderRef Builder)


.. py:function:: LLVMDIBuilderFinalizeSubprogram(Builder, Subprogram)

   Finalize a specific subprogram.

   No new variables may be added to this subprogram afterwards.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Subprogram (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDIBuilderFinalizeSubprogram(LLVMDIBuilderRef Builder, LLVMMetadataRef Subprogram)


.. py:function:: LLVMDIBuilderCreateCompileUnit(Builder, Lang, FileRef, Producer, ProducerLen, isOptimized, Flags, FlagsLen, RuntimeVer, SplitName, SplitNameLen, Kind, DWOId, SplitDebugInlining, DebugInfoForProfiling, SysRoot, SysRootLen, SDK, SDKLen)

   A CompileUnit provides an anchor for all debugging
   information generated during this instance of compilation.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Lang (:py:obj:`~.LLVMDWARFSourceLanguage`):
           Source programming language, eg.
           ``LLVMDWARFSourceLanguageC99``

       FileRef (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File info.

       Producer (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Identify the producer of debugging information
           and code.  Usually this is a compiler
           version string.

       ProducerLen (:py:obj:`~.int`):
           The length of the C string passed to ``Producer.``

       isOptimized (:py:obj:`~.int`):
           A boolean flag which indicates whether optimization
           is enabled or not.

       Flags (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           This string lists command line options. This
           string is directly embedded in debug info
           output which may be used by a tool
           analyzing generated debugging information.

       FlagsLen (:py:obj:`~.int`):
           The length of the C string passed to ``Flags.``

       RuntimeVer (:py:obj:`~.int`):
           This indicates runtime version for languages like
           Objective-C.

       SplitName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The name of the file that we'll split debug info
           out into.

       SplitNameLen (:py:obj:`~.int`):
           The length of the C string passed to ``SplitName.``

       Kind (:py:obj:`~.LLVMDWARFEmissionKind`):
           The kind of debug information to generate.

       DWOId (:py:obj:`~.int`):
           The DWOId if this is a split skeleton compile unit.

       SplitDebugInlining (:py:obj:`~.int`):
           Whether to emit inline debug info.

       DebugInfoForProfiling (:py:obj:`~.int`):
           Whether to emit extra debug info for
           profile collection.

       SysRoot (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The Clang system root (value of -isysroot).

       SysRootLen (:py:obj:`~.int`):
           The length of the C string passed to ``SysRoot.``

       SDK (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The SDK. On Darwin, the last component of the sysroot.

       SDKLen (:py:obj:`~.int`):
           The length of the C string passed to ``SDK.``

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateCompileUnit(LLVMDIBuilderRef Builder, LLVMDWARFSourceLanguage Lang, LLVMMetadataRef FileRef, const char * Producer, size_t ProducerLen, LLVMBool isOptimized, const char * Flags, size_t FlagsLen, unsigned int RuntimeVer, const char * SplitName, size_t SplitNameLen, LLVMDWARFEmissionKind Kind, unsigned int DWOId, LLVMBool SplitDebugInlining, LLVMBool DebugInfoForProfiling, const char * SysRoot, size_t SysRootLen, const char * SDK, size_t SDKLen)


.. py:function:: LLVMDIBuilderCreateFile(Builder, Filename, FilenameLen, Directory, DirectoryLen)

   Create a file descriptor to hold debugging information for a file.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Filename (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           File name.

       FilenameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Filename.``

       Directory (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Directory.

       DirectoryLen (:py:obj:`~.int`):
           The length of the C string passed to ``Directory.``

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateFile(LLVMDIBuilderRef Builder, const char * Filename, size_t FilenameLen, const char * Directory, size_t DirectoryLen)


.. py:function:: LLVMDIBuilderCreateFileWithChecksum(Builder, Filename, FilenameLen, Directory, DirectoryLen, ChecksumKind, Checksum, ChecksumLen, Source, SourceLen)

   Create a file descriptor to hold debugging information for a file.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Filename (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           File name.

       FilenameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Filename.``

       Directory (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Directory.

       DirectoryLen (:py:obj:`~.int`):
           The length of the C string passed to ``Directory.``

       ChecksumKind (:py:obj:`~.LLVMChecksumKind`):
           The kind of checksum. eg MD5, SHA256

       Checksum (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The checksum.

       ChecksumLen (:py:obj:`~.int`):
           The length of the checksum.

       Source (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       SourceLen (:py:obj:`~.int`):
           The length of the source.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateFileWithChecksum(LLVMDIBuilderRef Builder, const char * Filename, size_t FilenameLen, const char * Directory, size_t DirectoryLen, LLVMChecksumKind ChecksumKind, const char * Checksum, size_t ChecksumLen, const char * Source, size_t SourceLen)


.. py:function:: LLVMDIBuilderCreateModule(Builder, ParentScope, Name, NameLen, ConfigMacros, ConfigMacrosLen, IncludePath, IncludePathLen, APINotesFile, APINotesFileLen)

   Creates a new descriptor for a module with the specified parent scope.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       ParentScope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The parent scope containing this module declaration.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Module name.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       ConfigMacros (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A space-separated shell-quoted list of -D macro
           definitions as they would appear on a command line.

       ConfigMacrosLen (:py:obj:`~.int`):
           The length of the C string passed to ``ConfigMacros.``

       IncludePath (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The path to the module map file.

       IncludePathLen (:py:obj:`~.int`):
           The length of the C string passed to ``IncludePath.``

       APINotesFile (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The path to an API notes file for the module.

       APINotesFileLen (:py:obj:`~.int`):
           The length of the C string passed to ``APINotestFile.``

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateModule(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char * Name, size_t NameLen, const char * ConfigMacros, size_t ConfigMacrosLen, const char * IncludePath, size_t IncludePathLen, const char * APINotesFile, size_t APINotesFileLen)


.. py:function:: LLVMDIBuilderCreateNameSpace(Builder, ParentScope, Name, NameLen, ExportSymbols)

   Creates a new descriptor for a namespace with the specified parent scope.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       ParentScope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The parent scope containing this module declaration.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           NameSpace name.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       ExportSymbols (:py:obj:`~.int`):
           Whether or not the namespace exports symbols, e.g.
           this is true of C++ inline namespaces.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateNameSpace(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentScope, const char * Name, size_t NameLen, LLVMBool ExportSymbols)


.. py:function:: LLVMDIBuilderCreateFunction(Builder, Scope, Name, NameLen, LinkageName, LinkageNameLen, File, LineNo, Ty, IsLocalToUnit, IsDefinition, ScopeLine, Flags, IsOptimized)

   Create a new descriptor for the specified subprogram.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Function scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Function name.

       NameLen (:py:obj:`~.int`):
           Length of enumeration name.

       LinkageName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Mangled function name.

       LinkageNameLen (:py:obj:`~.int`):
           Length of linkage name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this variable is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Function type.

       IsLocalToUnit (:py:obj:`~.int`):
           True if this function is not externally visible.

       IsDefinition (:py:obj:`~.int`):
           True if this is a function definition.

       ScopeLine (:py:obj:`~.int`):
           Set to the beginning of the scope this starts

       Flags (:py:obj:`~.LLVMDIFlags`):
           E.g.: ``LLVMDIFlagLValueReference.`` These flags are
           used to emit dwarf attributes.

       IsOptimized (:py:obj:`~.int`):
           True if optimization is ON.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateFunction(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, const char * LinkageName, size_t LinkageNameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool IsLocalToUnit, LLVMBool IsDefinition, unsigned int ScopeLine, LLVMDIFlags Flags, LLVMBool IsOptimized)


.. py:function:: LLVMDIBuilderCreateLexicalBlock(Builder, Scope, File, Line, Column)

   Create a descriptor for a lexical block with the specified parent context.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Parent lexical block.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Source file.

       Line (:py:obj:`~.int`):
           The line in the source file.

       Column (:py:obj:`~.int`):
           The column in the source file.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateLexicalBlock(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int Column)


.. py:function:: LLVMDIBuilderCreateLexicalBlockFile(Builder, Scope, File, Discriminator)

   Create a descriptor for a lexical block with a new file attached.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Lexical block.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Source file.

       Discriminator (:py:obj:`~.int`):
           DWARF path discriminator value.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateLexicalBlockFile(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Discriminator)


.. py:function:: LLVMDIBuilderCreateImportedModuleFromNamespace(Builder, Scope, NS, File, Line)

   Create a descriptor for an imported namespace.

   Suitable for e.g. C++
   using declarations.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope this module is imported into

       NS (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the declaration is located.

       Line (:py:obj:`~.int`):
           Line number of the declaration.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromNamespace(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef NS, LLVMMetadataRef File, unsigned int Line)


.. py:function:: LLVMDIBuilderCreateImportedModuleFromAlias(Builder, Scope, ImportedEntity, File, Line, Elements, NumElements)

   Create a descriptor for an imported module that aliases another
   imported entity descriptor.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope this module is imported into

       ImportedEntity (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Previous imported entity to alias.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the declaration is located.

       Line (:py:obj:`~.int`):
           Line number of the declaration.

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Renamed elements.

       NumElements (:py:obj:`~.int`):
           Number of renamed elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromAlias(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef ImportedEntity, LLVMMetadataRef File, unsigned int Line, LLVMMetadataRef * Elements, unsigned int NumElements)


.. py:function:: LLVMDIBuilderCreateImportedModuleFromModule(Builder, Scope, M, File, Line, Elements, NumElements)

   Create a descriptor for an imported module.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The ``DIBuilder.``

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope this module is imported into

       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The module being imported here

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the declaration is located.

       Line (:py:obj:`~.int`):
           Line number of the declaration.

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Renamed elements.

       NumElements (:py:obj:`~.int`):
           Number of renamed elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateImportedModuleFromModule(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef M, LLVMMetadataRef File, unsigned int Line, LLVMMetadataRef * Elements, unsigned int NumElements)


.. py:function:: LLVMDIBuilderCreateImportedDeclaration(Builder, Scope, Decl, File, Line, Name, NameLen, Elements, NumElements)

   Create a descriptor for an imported function, type, or variable.

   Suitable
   for e.g. FORTRAN-style USE declarations.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope this module is imported into.

       Decl (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The declaration (or definition) of a function, type,
           or variable.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the declaration is located.

       Line (:py:obj:`~.int`):
           Line number of the declaration.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A name that uniquely identifies this imported
           declaration.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Renamed elements.

       NumElements (:py:obj:`~.int`):
           Number of renamed elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateImportedDeclaration(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, LLVMMetadataRef Decl, LLVMMetadataRef File, unsigned int Line, const char * Name, size_t NameLen, LLVMMetadataRef * Elements, unsigned int NumElements)


.. py:function:: LLVMDIBuilderCreateDebugLocation(Ctx, Line, Column, Scope, InlinedAt)

   Creates a new DebugLocation that describes a source location.

   Note:
       If the item to which this location is attached cannot be
       attributed to a source line, pass 0 for the line and column.

   Args:
       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Line (:py:obj:`~.int`):
           The line in the source file.

       Column (:py:obj:`~.int`):
           The column in the source file.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope in which the location resides.

       InlinedAt (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope where this location was inlined, if at all.
           (optional).

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateDebugLocation(LLVMContextRef Ctx, unsigned int Line, unsigned int Column, LLVMMetadataRef Scope, LLVMMetadataRef InlinedAt)


.. py:function:: LLVMDILocationGetLine(Location)

   Get the line number of this debug location.

   See:
       DILocation::getLine()

   Args:
       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug location.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDILocationGetLine(LLVMMetadataRef Location)


.. py:function:: LLVMDILocationGetColumn(Location)

   Get the column number of this debug location.

   See:
       DILocation::getColumn()

   Args:
       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug location.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDILocationGetColumn(LLVMMetadataRef Location)


.. py:function:: LLVMDILocationGetScope(Location)

   Get the local scope associated with this debug location.

   See:
       DILocation::getScope()

   Args:
       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug location.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDILocationGetScope(LLVMMetadataRef Location)


.. py:function:: LLVMDILocationGetInlinedAt(Location)

   Get the "inline at" location associated with this debug location.

   See:
       DILocation::getInlinedAt()

   Args:
       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug location.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDILocationGetInlinedAt(LLVMMetadataRef Location)


.. py:function:: LLVMDIScopeGetFile(Scope)

   Get the metadata of the file associated with a given scope.

   See:
       DIScope::getFile()

   Args:
       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope object.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIScopeGetFile(LLVMMetadataRef Scope)


.. py:function:: LLVMDIFileGetDirectory(File, Len)

   Get the directory of a given file.

   See:
       DIFile::getDirectory()

   Args:
       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The file object.

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           The length of the returned string.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMDIFileGetDirectory(LLVMMetadataRef File, unsigned int * Len)


.. py:function:: LLVMDIFileGetFilename(File, Len)

   Get the name of a given file.

   See:
       DIFile::getFilename()

   Args:
       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The file object.

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           The length of the returned string.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMDIFileGetFilename(LLVMMetadataRef File, unsigned int * Len)


.. py:function:: LLVMDIFileGetSource(File, Len)

   Get the source of a given file.

   See:
       DIFile::getSource()

   Args:
       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The file object.

       Len (:py:obj:`~.rocm.bindings.util.types.PointerToUnsigned`/:py:obj:`~.object`):
           The length of the returned string.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMDIFileGetSource(LLVMMetadataRef File, unsigned int * Len)


.. py:function:: LLVMDIBuilderGetOrCreateTypeArray(Builder, Data, NumElements)

   Create a type array.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The type elements.

       NumElements (:py:obj:`~.int`):
           Number of type elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderGetOrCreateTypeArray(LLVMDIBuilderRef Builder, LLVMMetadataRef * Data, size_t NumElements)


.. py:function:: LLVMDIBuilderCreateSubroutineType(Builder, File, ParameterTypes, NumParameterTypes, Flags)

   Create subroutine type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The file in which the subroutine resides.

       ParameterTypes (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           An array of subroutine parameter types. This
           includes return type at 0th index.

       NumParameterTypes (:py:obj:`~.int`):
           The number of parameter types in ``ParameterTypes``

       Flags (:py:obj:`~.LLVMDIFlags`):
           E.g.: ``LLVMDIFlagLValueReference.``
           These flags are used to emit dwarf attributes.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateSubroutineType(LLVMDIBuilderRef Builder, LLVMMetadataRef File, LLVMMetadataRef * ParameterTypes, unsigned int NumParameterTypes, LLVMDIFlags Flags)


.. py:function:: LLVMDIBuilderCreateMacro(Builder, ParentMacroFile, Line, RecordType, Name, NameLen, Value, ValueLen)

   Create debugging information entry for a macro.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       ParentMacroFile (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Macro parent (could be NULL).

       Line (:py:obj:`~.int`):
           Source line number where the macro is defined.

       RecordType (:py:obj:`~.LLVMDWARFMacinfoRecordType`):
           DW_MACINFO_define or DW_MACINFO_undef.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Macro name.

       NameLen (:py:obj:`~.int`):
           Macro name length.

       Value (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Macro value.

       ValueLen (:py:obj:`~.int`):
           Macro value length.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateMacro(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned int Line, LLVMDWARFMacinfoRecordType RecordType, const char * Name, size_t NameLen, const char * Value, size_t ValueLen)


.. py:function:: LLVMDIBuilderCreateTempMacroFile(Builder, ParentMacroFile, Line, File)

   Create debugging information temporary entry for a macro file.

   List of macro node direct children will be calculated by DIBuilder,
   using the ``ParentMacroFile`` relationship.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       ParentMacroFile (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Macro parent (could be NULL).

       Line (:py:obj:`~.int`):
           Source line number where the macro file is included.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File descriptor containing the name of the macro file.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateTempMacroFile(LLVMDIBuilderRef Builder, LLVMMetadataRef ParentMacroFile, unsigned int Line, LLVMMetadataRef File)


.. py:function:: LLVMDIBuilderCreateEnumerator(Builder, Name, NameLen, Value, IsUnsigned)

   Create debugging information entry for an enumerator.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Enumerator name.

       NameLen (:py:obj:`~.int`):
           Length of enumerator name.

       Value (:py:obj:`~.int`):
           Enumerator value.

       IsUnsigned (:py:obj:`~.int`):
           True if the value is unsigned.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateEnumerator(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen, int64_t Value, LLVMBool IsUnsigned)


.. py:function:: LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision(Builder, Name, NameLen, SizeInBits, Words, IsUnsigned)

   Create debugging information entry for an enumerator of arbitrary precision.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Enumerator name.

       NameLen (:py:obj:`~.int`):
           Length of enumerator name.

       SizeInBits (:py:obj:`~.int`):
           Number of bits of the value.

       Words (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The words that make up the value.

       IsUnsigned (:py:obj:`~.int`):
           True if the value is unsigned.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateEnumeratorOfArbitraryPrecision(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen, uint64_t SizeInBits, const uint64_t[] Words, LLVMBool IsUnsigned)


.. py:function:: LLVMDIBuilderCreateEnumerationType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, AlignInBits, Elements, NumElements, ClassTy)

   Create debugging information entry for an enumeration.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Scope in which this enumeration is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Enumeration name.

       NameLen (:py:obj:`~.int`):
           Length of enumeration name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNumber (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Enumeration elements.

       NumElements (:py:obj:`~.int`):
           Number of enumeration elements.

       ClassTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Underlying type of a C++11/ObjC fixed enum.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateEnumerationType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef * Elements, unsigned int NumElements, LLVMMetadataRef ClassTy)


.. py:function:: LLVMDIBuilderCreateUnionType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, AlignInBits, Flags, Elements, NumElements, RunTimeLang, UniqueId, UniqueIdLen)

   Create debugging information entry for a union.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Scope in which this union is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Union name.

       NameLen (:py:obj:`~.int`):
           Length of union name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNumber (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Union elements.

       NumElements (:py:obj:`~.int`):
           Number of union elements.

       RunTimeLang (:py:obj:`~.int`):
           Optional parameter, Objective-C runtime version.

       UniqueId (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A unique identifier for the union.

       UniqueIdLen (:py:obj:`~.int`):
           Length of unique identifier.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateUnionType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef * Elements, unsigned int NumElements, unsigned int RunTimeLang, const char * UniqueId, size_t UniqueIdLen)


.. py:function:: LLVMDIBuilderCreateArrayType(Builder, Size, AlignInBits, Ty, Subscripts, NumSubscripts)

   Create debugging information entry for an array.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Size (:py:obj:`~.int`):
           Array size.

       AlignInBits (:py:obj:`~.int`):
           Alignment.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Element type.

       Subscripts (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Subscripts.

       NumSubscripts (:py:obj:`~.int`):
           Number of subscripts.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateArrayType(LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef * Subscripts, unsigned int NumSubscripts)


.. py:function:: LLVMDIBuilderCreateSetType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, AlignInBits, BaseTy)

   Create debugging information entry for a set.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope in which the set is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A name that uniquely identifies this set.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the set is located.

       LineNumber (:py:obj:`~.int`):
           (undocumented)

       SizeInBits (:py:obj:`~.int`):
           Set size.

       AlignInBits (:py:obj:`~.int`):
           Set alignment.

       BaseTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The base type of the set.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateSetType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMMetadataRef BaseTy)


.. py:function:: LLVMDIBuilderCreateSubrangeType(Builder, Scope, Name, NameLen, LineNo, File, SizeInBits, AlignInBits, Flags, BaseTy, LowerBound, UpperBound, Stride, Bias)

   Create a descriptor for a subrange with dynamic bounds.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The scope in which the subrange is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A name that uniquely identifies this subrange.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       LineNo (:py:obj:`~.int`):
           Line number.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where the subrange is located.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags.

       BaseTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The base type of the subrange. eg integer or enumeration

       LowerBound (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Lower bound of the subrange.

       UpperBound (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Upper bound of the subrange.

       Stride (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Stride of the subrange.

       Bias (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Bias of the subrange.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateSubrangeType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, unsigned int LineNo, LLVMMetadataRef File, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef BaseTy, LLVMMetadataRef LowerBound, LLVMMetadataRef UpperBound, LLVMMetadataRef Stride, LLVMMetadataRef Bias)


.. py:function:: LLVMDIBuilderCreateDynamicArrayType(Builder, Scope, Name, NameLen, LineNo, File, Size, AlignInBits, Ty, Subscripts, NumSubscripts, DataLocation, Associated, Allocated, Rank, BitStride)

   Create debugging information entry for a dynamic array.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       NameLen (:py:obj:`~.int`):
           (undocumented)

       LineNo (:py:obj:`~.int`):
           (undocumented)

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Size (:py:obj:`~.int`):
           Array size.

       AlignInBits (:py:obj:`~.int`):
           Alignment.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Element type.

       Subscripts (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Subscripts.

       NumSubscripts (:py:obj:`~.int`):
           Number of subscripts.

       DataLocation (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           DataLocation. (DIVariable, DIExpression or NULL)

       Associated (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Associated. (DIVariable, DIExpression or NULL)

       Allocated (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Allocated. (DIVariable, DIExpression or NULL)

       Rank (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Rank. (DIVariable, DIExpression or NULL)

       BitStride (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           BitStride.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateDynamicArrayType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, unsigned int LineNo, LLVMMetadataRef File, uint64_t Size, uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef * Subscripts, unsigned int NumSubscripts, LLVMMetadataRef DataLocation, LLVMMetadataRef Associated, LLVMMetadataRef Allocated, LLVMMetadataRef Rank, LLVMMetadataRef BitStride)


.. py:function:: LLVMReplaceArrays(Builder, T, Elements, NumElements)

   Replace arrays.

   See:
       DIBuilder::replaceArrays()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       T (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumElements (:py:obj:`~.int`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMReplaceArrays(LLVMDIBuilderRef Builder, LLVMMetadataRef * T, LLVMMetadataRef * Elements, unsigned int NumElements)


.. py:function:: LLVMDIBuilderCreateVectorType(Builder, Size, AlignInBits, Ty, Subscripts, NumSubscripts)

   Create debugging information entry for a vector type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Size (:py:obj:`~.int`):
           Vector size.

       AlignInBits (:py:obj:`~.int`):
           Alignment.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Element type.

       Subscripts (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Subscripts.

       NumSubscripts (:py:obj:`~.int`):
           Number of subscripts.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateVectorType(LLVMDIBuilderRef Builder, uint64_t Size, uint32_t AlignInBits, LLVMMetadataRef Ty, LLVMMetadataRef * Subscripts, unsigned int NumSubscripts)


.. py:function:: LLVMDIBuilderCreateUnspecifiedType(Builder, Name, NameLen)

   Create a DWARF unspecified type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           The unspecified type's name.

       NameLen (:py:obj:`~.int`):
           Length of type name.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateUnspecifiedType(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen)


.. py:function:: LLVMDIBuilderCreateBasicType(Builder, Name, NameLen, SizeInBits, Encoding, Flags)

   Create debugging information entry for a basic
   type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Type name.

       NameLen (:py:obj:`~.int`):
           Length of type name.

       SizeInBits (:py:obj:`~.int`):
           Size of the type.

       Encoding (:py:obj:`~.int`):
           DWARF encoding code, e.g. ``LLVMDWARFTypeEncoding_float.``

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode optional attribute like endianity

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateBasicType(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen, uint64_t SizeInBits, LLVMDWARFTypeEncoding Encoding, LLVMDIFlags Flags)


.. py:function:: LLVMDIBuilderCreatePointerType(Builder, PointeeTy, SizeInBits, AlignInBits, AddressSpace, MS, Name, NameLen)

   Create debugging information entry for a pointer.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       PointeeTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type pointed by this pointer.

       SizeInBits (:py:obj:`~.int`):
           Size.

       AlignInBits (:py:obj:`~.int`):
           Alignment. (optional, pass 0 to ignore)

       AddressSpace (:py:obj:`~.int`):
           DWARF address space. (optional, pass 0 to ignore)

       MS (:py:obj:`~.int`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Pointer type name. (optional)

       NameLen (:py:obj:`~.int`):
           Length of pointer type name. (optional)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreatePointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeTy, uint64_t SizeInBits, uint32_t AlignInBits, unsigned int AddressSpace, LLVMDWARFMemorySpace MS, const char * Name, size_t NameLen)


.. py:function:: LLVMDIBuilderCreateStructType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, AlignInBits, Flags, DerivedFrom, Elements, NumElements, RunTimeLang, VTableHolder, UniqueId, UniqueIdLen)

   Create debugging information entry for a struct.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Scope in which this struct is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Struct name.

       NameLen (:py:obj:`~.int`):
           Struct name length.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNumber (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private

       DerivedFrom (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Struct elements.

       NumElements (:py:obj:`~.int`):
           Number of struct elements.

       RunTimeLang (:py:obj:`~.int`):
           Optional parameter, Objective-C runtime version.

       VTableHolder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The object containing the vtable for the struct.

       UniqueId (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A unique identifier for the struct.

       UniqueIdLen (:py:obj:`~.int`):
           Length of the unique identifier for the struct.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateStructType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, LLVMMetadataRef DerivedFrom, LLVMMetadataRef * Elements, unsigned int NumElements, unsigned int RunTimeLang, LLVMMetadataRef VTableHolder, const char * UniqueId, size_t UniqueIdLen)


.. py:function:: LLVMDIBuilderCreateMemberType(Builder, Scope, Name, NameLen, File, LineNo, SizeInBits, AlignInBits, OffsetInBits, Flags, Ty)

   Create debugging information entry for a member.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Member scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Member name.

       NameLen (:py:obj:`~.int`):
           Length of member name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       OffsetInBits (:py:obj:`~.int`):
           Member offset.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Parent type.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Ty)


.. py:function:: LLVMDIBuilderCreateStaticMemberType(Builder, Scope, Name, NameLen, File, LineNumber, Type, Flags, ConstantVal, AlignInBits)

   Create debugging information entry for a
   C++ static data member.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Member scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Member name.

       NameLen (:py:obj:`~.int`):
           Length of member name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is declared.

       LineNumber (:py:obj:`~.int`):
           Line number.

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type of the static member.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private.

       ConstantVal (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Const initializer of the member.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateStaticMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, LLVMMetadataRef Type, LLVMDIFlags Flags, LLVMValueRef ConstantVal, uint32_t AlignInBits)


.. py:function:: LLVMDIBuilderCreateMemberPointerType(Builder, PointeeType, ClassType, SizeInBits, AlignInBits, Flags)

   Create debugging information entry for a pointer to member.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       PointeeType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type pointed to by this pointer.

       ClassType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type for which this pointer points to members of.

       SizeInBits (:py:obj:`~.int`):
           Size.

       AlignInBits (:py:obj:`~.int`):
           Alignment.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateMemberPointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef PointeeType, LLVMMetadataRef ClassType, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags)


.. py:function:: LLVMDIBuilderCreateObjCIVar(Builder, Name, NameLen, File, LineNo, SizeInBits, AlignInBits, OffsetInBits, Flags, Ty, PropertyNode)

   Create debugging information entry for Objective-C instance variable.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Member name.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       OffsetInBits (:py:obj:`~.int`):
           Member offset.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Parent type.

       PropertyNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Property associated with this ivar.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateObjCIVar(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Ty, LLVMMetadataRef PropertyNode)


.. py:function:: LLVMDIBuilderCreateObjCProperty(Builder, Name, NameLen, File, LineNo, GetterName, GetterNameLen, SetterName, SetterNameLen, PropertyAttributes, Ty)

   Create debugging information entry for Objective-C property.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Property name.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this property is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       GetterName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Name of the Objective C property getter selector.

       GetterNameLen (:py:obj:`~.int`):
           The length of the C string passed to ``GetterName.``

       SetterName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Name of the Objective C property setter selector.

       SetterNameLen (:py:obj:`~.int`):
           The length of the C string passed to ``SetterName.``

       PropertyAttributes (:py:obj:`~.int`):
           Objective C property attributes.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateObjCProperty(LLVMDIBuilderRef Builder, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, const char * GetterName, size_t GetterNameLen, const char * SetterName, size_t SetterNameLen, unsigned int PropertyAttributes, LLVMMetadataRef Ty)


.. py:function:: LLVMDIBuilderCreateObjectPointerType(Builder, Type, Implicit)

   Create a uniqued DIType* clone with FlagObjectPointer.

   If ``Implicit``
   is true, then also set FlagArtificial.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The underlying type to which this pointer points.

       Implicit (:py:obj:`~.int`):
           Indicates whether this pointer was implicitly generated
           (i.e., not spelled out in source).

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateObjectPointerType(LLVMDIBuilderRef Builder, LLVMMetadataRef Type, LLVMBool Implicit)


.. py:function:: LLVMDIBuilderCreateQualifiedType(Builder, Tag, Type)

   Create debugging information entry for a qualified
   type, e.g.

   'const int'.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Tag (:py:obj:`~.int`):
           Tag identifying type,
           e.g. LLVMDWARFTypeQualifier_volatile_type

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Base Type.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateQualifiedType(LLVMDIBuilderRef Builder, unsigned int Tag, LLVMMetadataRef Type)


.. py:function:: LLVMDIBuilderCreateReferenceType(Builder, Tag, Type, AddressSpace, MemorySpace)

   Create debugging information entry for a c++
   style reference or rvalue reference type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Tag (:py:obj:`~.int`):
           Tag identifying type,

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Base Type.

       AddressSpace (:py:obj:`~.int`):
           DWARF address space. (optional, pass 0 to ignore)

       MemorySpace (:py:obj:`~.int`):
           DWARF memory space (optional, pass 0 for none).

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateReferenceType(LLVMDIBuilderRef Builder, unsigned int Tag, LLVMMetadataRef Type, unsigned int AddressSpace, LLVMDWARFMemorySpace MemorySpace)


.. py:function:: LLVMDIBuilderCreateNullPtrType(Builder)

   Create C++11 nullptr type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateNullPtrType(LLVMDIBuilderRef Builder)


.. py:function:: LLVMDIBuilderCreateTypedef(Builder, Type, Name, NameLen, File, LineNo, Scope, AlignInBits)

   Create debugging information entry for a typedef.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Original type.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Typedef name.

       NameLen (:py:obj:`~.int`):
           (undocumented)

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this type is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The surrounding context for the typedef.

       AlignInBits (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateTypedef(LLVMDIBuilderRef Builder, LLVMMetadataRef Type, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Scope, uint32_t AlignInBits)


.. py:function:: LLVMDIBuilderCreateInheritance(Builder, Ty, BaseTy, BaseOffset, VBPtrOffset, Flags)

   Create debugging information entry to establish inheritance relationship
   between two types.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Original type.

       BaseTy (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Base type. Ty is inherits from base.

       BaseOffset (:py:obj:`~.int`):
           Base offset.

       VBPtrOffset (:py:obj:`~.int`):
           Virtual base pointer offset.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to describe inheritance attribute, e.g. private

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateInheritance(LLVMDIBuilderRef Builder, LLVMMetadataRef Ty, LLVMMetadataRef BaseTy, uint64_t BaseOffset, uint32_t VBPtrOffset, LLVMDIFlags Flags)


.. py:function:: LLVMDIBuilderCreateForwardDecl(Builder, Tag, Name, NameLen, Scope, File, Line, RuntimeLang, SizeInBits, AlignInBits, UniqueIdentifier, UniqueIdentifierLen)

   Create a permanent forward-declared type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Tag (:py:obj:`~.int`):
           A unique tag for this type.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Type name.

       NameLen (:py:obj:`~.int`):
           Length of type name.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type scope.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this type is defined.

       Line (:py:obj:`~.int`):
           Line number where this type is defined.

       RuntimeLang (:py:obj:`~.int`):
           Indicates runtime version for languages like
           Objective-C.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       UniqueIdentifier (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A unique identifier for the type.

       UniqueIdentifierLen (:py:obj:`~.int`):
           Length of the unique identifier.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateForwardDecl(LLVMDIBuilderRef Builder, unsigned int Tag, const char * Name, size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits, const char * UniqueIdentifier, size_t UniqueIdentifierLen)


.. py:function:: LLVMDIBuilderCreateReplaceableCompositeType(Builder, Tag, Name, NameLen, Scope, File, Line, RuntimeLang, SizeInBits, AlignInBits, Flags, UniqueIdentifier, UniqueIdentifierLen)

   Create a temporary forward-declared type.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Tag (:py:obj:`~.int`):
           A unique tag for this type.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Type name.

       NameLen (:py:obj:`~.int`):
           Length of type name.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Type scope.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this type is defined.

       Line (:py:obj:`~.int`):
           Line number where this type is defined.

       RuntimeLang (:py:obj:`~.int`):
           Indicates runtime version for languages like
           Objective-C.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags.

       UniqueIdentifier (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A unique identifier for the type.

       UniqueIdentifierLen (:py:obj:`~.int`):
           Length of the unique identifier.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateReplaceableCompositeType(LLVMDIBuilderRef Builder, unsigned int Tag, const char * Name, size_t NameLen, LLVMMetadataRef Scope, LLVMMetadataRef File, unsigned int Line, unsigned int RuntimeLang, uint64_t SizeInBits, uint32_t AlignInBits, LLVMDIFlags Flags, const char * UniqueIdentifier, size_t UniqueIdentifierLen)


.. py:function:: LLVMDIBuilderCreateBitFieldMemberType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, OffsetInBits, StorageOffsetInBits, Flags, Type)

   Create debugging information entry for a bit field member.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Member scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Member name.

       NameLen (:py:obj:`~.int`):
           Length of member name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNumber (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       OffsetInBits (:py:obj:`~.int`):
           Member offset.

       StorageOffsetInBits (:py:obj:`~.int`):
           Member storage offset.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute.

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Parent type.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateBitFieldMemberType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint64_t OffsetInBits, uint64_t StorageOffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef Type)


.. py:function:: LLVMDIBuilderCreateClassType(Builder, Scope, Name, NameLen, File, LineNumber, SizeInBits, AlignInBits, OffsetInBits, Flags, DerivedFrom, Elements, NumElements, VTableHolder, TemplateParamsNode, UniqueIdentifier, UniqueIdentifierLen)

   Create debugging information entry for a class.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Scope in which this class is defined.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Class name.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this member is defined.

       LineNumber (:py:obj:`~.int`):
           Line number.

       SizeInBits (:py:obj:`~.int`):
           Member size.

       AlignInBits (:py:obj:`~.int`):
           Member alignment.

       OffsetInBits (:py:obj:`~.int`):
           Member offset.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags to encode member attribute, e.g. private.

       DerivedFrom (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info of the base class of this type.

       Elements (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Class members.

       NumElements (:py:obj:`~.int`):
           Number of class elements.

       VTableHolder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info of the base class that contains vtable
           for this type. This is used in
           DW_AT_containing_type. See DWARF documentation
           for more info.

       TemplateParamsNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Template type parameters.

       UniqueIdentifier (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           A unique identifier for the type.

       UniqueIdentifierLen (:py:obj:`~.int`):
           Length of the unique identifier.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateClassType(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNumber, uint64_t SizeInBits, uint32_t AlignInBits, uint64_t OffsetInBits, LLVMDIFlags Flags, LLVMMetadataRef DerivedFrom, LLVMMetadataRef * Elements, unsigned int NumElements, LLVMMetadataRef VTableHolder, LLVMMetadataRef TemplateParamsNode, const char * UniqueIdentifier, size_t UniqueIdentifierLen)


.. py:function:: LLVMDIBuilderCreateArtificialType(Builder, Type)

   Create a uniqued DIType* clone with FlagArtificial set.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Type (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The underlying type.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateArtificialType(LLVMDIBuilderRef Builder, LLVMMetadataRef Type)


.. py:function:: LLVMDITypeGetName(DType, Length)

   Get the name of this DIType.

   See:
       DIType::getName()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

       Length (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           The length of the returned string.

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMDITypeGetName(LLVMMetadataRef DType, size_t * Length)


.. py:function:: LLVMDITypeGetSizeInBits(DType)

   Get the size of this DIType in bits.

   See:
       DIType::getSizeInBits()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMDITypeGetSizeInBits(LLVMMetadataRef DType)


.. py:function:: LLVMDITypeGetOffsetInBits(DType)

   Get the offset of this DIType in bits.

   See:
       DIType::getOffsetInBits()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint64_t LLVMDITypeGetOffsetInBits(LLVMMetadataRef DType)


.. py:function:: LLVMDITypeGetAlignInBits(DType)

   Get the alignment of this DIType in bits.

   See:
       DIType::getAlignInBits()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint32_t LLVMDITypeGetAlignInBits(LLVMMetadataRef DType)


.. py:function:: LLVMDITypeGetLine(DType)

   Get the source line where this DIType is declared.

   See:
       DIType::getLine()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDITypeGetLine(LLVMMetadataRef DType)


.. py:function:: LLVMDITypeGetFlags(DType)

   Get the flags associated with this DIType.

   See:
       DIType::getFlags()

   Args:
       DType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIType.

   Returns:
       :py:obj:`~.LLVMDIFlags`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDIFlags LLVMDITypeGetFlags(LLVMMetadataRef DType)


.. py:function:: LLVMDIBuilderGetOrCreateSubrange(Builder, LowerBound, Count)

   Create a descriptor for a value range.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       LowerBound (:py:obj:`~.int`):
           Lower bound of the subrange, e.g. 0 for C, 1 for Fortran.

       Count (:py:obj:`~.int`):
           Count of elements in the subrange.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderGetOrCreateSubrange(LLVMDIBuilderRef Builder, int64_t LowerBound, int64_t Count)


.. py:function:: LLVMDIBuilderGetOrCreateArray(Builder, Data, NumElements)

   Create an array of DI Nodes.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DI Node elements.

       NumElements (:py:obj:`~.int`):
           Number of DI Node elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderGetOrCreateArray(LLVMDIBuilderRef Builder, LLVMMetadataRef * Data, size_t NumElements)


.. py:function:: LLVMDIBuilderCreateExpression(Builder, Addr, Length)

   Create a new descriptor for the specified variable which has a complex
   address expression for its address.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Addr (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           An array of complex address operations.

       Length (:py:obj:`~.int`):
           Length of the address operation array.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateExpression(LLVMDIBuilderRef Builder, uint64_t * Addr, size_t Length)


.. py:function:: LLVMDIBuilderCreateConstantValueExpression(Builder, Value)

   Create a new descriptor for the specified variable that does not have an
   address, but does have a constant value.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Value (:py:obj:`~.int`):
           The constant value.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateConstantValueExpression(LLVMDIBuilderRef Builder, uint64_t Value)


.. py:function:: LLVMDIBuilderCreateGlobalVariableExpression(Builder, Scope, Name, NameLen, Linkage, LinkLen, File, LineNo, Ty, LocalToUnit, Expr, Decl, MS, AlignInBits)

   Create a new descriptor for the specified variable.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Variable scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Name of the variable.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       Linkage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Mangled  name of the variable.

       LinkLen (:py:obj:`~.int`):
           The length of the C string passed to ``Linkage.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this variable is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Variable Type.

       LocalToUnit (:py:obj:`~.int`):
           Boolean flag indicate whether this variable is
           externally visible or not.

       Expr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The location of the global relative to the attached
           GlobalVariable.

       Decl (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Reference to the corresponding declaration.
           variables.

       MS (:py:obj:`~.int`):
           (undocumented)

       AlignInBits (:py:obj:`~.int`):
           Variable alignment(or 0 if no alignment attr was
           specified)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateGlobalVariableExpression(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, const char * Linkage, size_t LinkLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit, LLVMMetadataRef Expr, LLVMMetadataRef Decl, LLVMDWARFMemorySpace MS, uint32_t AlignInBits)


.. py:function:: LLVMGetDINodeTag(MD)

   Get the dwarf::Tag of a DINode

   Args:
       MD (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       uint16_t LLVMGetDINodeTag(LLVMMetadataRef MD)


.. py:function:: LLVMDIGlobalVariableExpressionGetVariable(GVE)

   Retrieves the ``DIVariable`` associated with this global variable expression.

   See:
       llvm::DIGlobalVariableExpression::getVariable()

   Args:
       GVE (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The global variable expression.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIGlobalVariableExpressionGetVariable(LLVMMetadataRef GVE)


.. py:function:: LLVMDIGlobalVariableExpressionGetExpression(GVE)

   Retrieves the ``DIExpression`` associated with this global variable expression.

   See:
       llvm::DIGlobalVariableExpression::getExpression()

   Args:
       GVE (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The global variable expression.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIGlobalVariableExpressionGetExpression(LLVMMetadataRef GVE)


.. py:function:: LLVMDIVariableGetFile(Var)

   Get the metadata of the file associated with a given variable.

   See:
       DIVariable::getFile()

   Args:
       Var (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable object.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIVariableGetFile(LLVMMetadataRef Var)


.. py:function:: LLVMDIVariableGetScope(Var)

   Get the metadata of the scope associated with a given variable.

   See:
       DIVariable::getScope()

   Args:
       Var (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable object.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIVariableGetScope(LLVMMetadataRef Var)


.. py:function:: LLVMDIVariableGetLine(Var)

   Get the source line where this ``DIVariable`` is declared.

   See:
       DIVariable::getLine()

   Args:
       Var (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIVariable.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDIVariableGetLine(LLVMMetadataRef Var)


.. py:function:: LLVMTemporaryMDNode(Ctx, Data, NumElements)

   Create a new temporary ``MDNode.``  Suitable for use in constructing cyclic
   ``MDNode`` structures.

   A temporary ``MDNode`` is not uniqued, may be RAUW'd,
   and must be manually deleted with ``LLVMDisposeTemporaryMDNode.``

   Args:
       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The context in which to construct the temporary node.

       Data (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The metadata elements.

       NumElements (:py:obj:`~.int`):
           Number of metadata elements.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMTemporaryMDNode(LLVMContextRef Ctx, LLVMMetadataRef * Data, size_t NumElements)


.. py:function:: LLVMDisposeTemporaryMDNode(TempNode)

   Deallocate a temporary node.

   Calls ``replaceAllUsesWith(nullptr)`` before deleting, so any remaining
   references will be reset.

   Args:
       TempNode (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The temporary metadata node.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDisposeTemporaryMDNode(LLVMMetadataRef TempNode)


.. py:function:: LLVMMetadataReplaceAllUsesWith(TempTargetMetadata, Replacement)

   Replace all uses of temporary metadata.

   Args:
       TempTargetMetadata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The temporary metadata node.

       Replacement (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The replacement metadata node.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMMetadataReplaceAllUsesWith(LLVMMetadataRef TempTargetMetadata, LLVMMetadataRef Replacement)


.. py:function:: LLVMDIBuilderCreateTempGlobalVariableFwdDecl(Builder, Scope, Name, NameLen, Linkage, LnkLen, File, LineNo, Ty, LocalToUnit, Decl, AlignInBits)

   Create a new descriptor for the specified global variable that is temporary
   and meant to be RAUWed.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Variable scope.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Name of the variable.

       NameLen (:py:obj:`~.int`):
           The length of the C string passed to ``Name.``

       Linkage (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Mangled  name of the variable.

       LnkLen (:py:obj:`~.int`):
           The length of the C string passed to ``Linkage.``

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this variable is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Variable Type.

       LocalToUnit (:py:obj:`~.int`):
           Boolean flag indicate whether this variable is
           externally visible or not.

       Decl (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Reference to the corresponding declaration.

       AlignInBits (:py:obj:`~.int`):
           Variable alignment(or 0 if no alignment attr was
           specified)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateTempGlobalVariableFwdDecl(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, const char * Linkage, size_t LnkLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool LocalToUnit, LLVMMetadataRef Decl, uint32_t AlignInBits)


.. py:function:: LLVMDIBuilderInsertDeclareRecordBefore(Builder, Storage, VarInfo, Expr, DebugLoc, Instr)

   Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   The debug format can be switched later after inserting the records using
   LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.

   Insert a Declare DbgRecord before the given instruction.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Storage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The storage of the variable to declare.

       VarInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable's debug info descriptor.

       Expr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           A complex location expression for the variable.

       DebugLoc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info location.

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Instruction acting as a location for the new record.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordBefore(LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr)


.. py:function:: LLVMDIBuilderInsertDeclareRecordAtEnd(Builder, Storage, VarInfo, Expr, DebugLoc, Block)

   Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   The debug format can be switched later after inserting the records using
   LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.

   Insert a Declare DbgRecord at the end of the given basic block. If the basic
   block has a terminator instruction, the record is inserted before that
   terminator instruction.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Storage (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The storage of the variable to declare.

       VarInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable's debug info descriptor.

       Expr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           A complex location expression for the variable.

       DebugLoc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info location.

       Block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Basic block acting as a location for the new record.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertDeclareRecordAtEnd(LLVMDIBuilderRef Builder, LLVMValueRef Storage, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block)


.. py:function:: LLVMDIBuilderInsertDbgValueRecordBefore(Builder, Val, VarInfo, Expr, DebugLoc, Instr)

   Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   The debug format can be switched later after inserting the records using
   LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.

   Insert a new debug record before the given instruction.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The value of the variable.

       VarInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable's debug info descriptor.

       Expr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           A complex location expression for the variable.

       DebugLoc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info location.

       Instr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Instruction acting as a location for the new record.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordBefore(LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr)


.. py:function:: LLVMDIBuilderInsertDbgValueRecordAtEnd(Builder, Val, VarInfo, Expr, DebugLoc, Block)

   Only use in "new debug format" (LLVMIsNewDbgInfoFormat() is true).

   See https://llvm.org/docs/RemoveDIsDebugInfo.html:py:obj:`~.c`-api-changes

   The debug format can be switched later after inserting the records using
   LLVMSetIsNewDbgInfoFormat, if needed for legacy or transitionary reasons.

   Insert a new debug record at the end of the given basic block. If the
   basic block has a terminator instruction, the record is inserted before
   that terminator instruction.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Val (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The value of the variable.

       VarInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The variable's debug info descriptor.

       Expr (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           A complex location expression for the variable.

       DebugLoc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Debug info location.

       Block (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Basic block acting as a location for the new record.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertDbgValueRecordAtEnd(LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo, LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMBasicBlockRef Block)


.. py:function:: LLVMDIBuilderCreateAutoVariable(Builder, Scope, Name, NameLen, File, LineNo, Ty, AlwaysPreserve, Flags, MS, AlignInBits)

   Create a new descriptor for a local auto variable.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The local scope the variable is declared in.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Variable name.

       NameLen (:py:obj:`~.int`):
           Length of variable name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this variable is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Metadata describing the type of the variable.

       AlwaysPreserve (:py:obj:`~.int`):
           If true, this descriptor will survive optimizations.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags.

       MS (:py:obj:`~.int`):
           (undocumented)

       AlignInBits (:py:obj:`~.int`):
           Variable alignment.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateAutoVariable(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags, LLVMDWARFMemorySpace MS, uint32_t AlignInBits)


.. py:function:: LLVMDIBuilderCreateParameterVariable(Builder, Scope, Name, NameLen, ArgNo, File, LineNo, Ty, AlwaysPreserve, Flags)

   Create a new descriptor for a function parameter variable.

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Scope (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The local scope the variable is declared in.

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Variable name.

       NameLen (:py:obj:`~.int`):
           Length of variable name.

       ArgNo (:py:obj:`~.int`):
           Unique argument number for this variable; starts at 1.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           File where this variable is defined.

       LineNo (:py:obj:`~.int`):
           Line number.

       Ty (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Metadata describing the type of the variable.

       AlwaysPreserve (:py:obj:`~.int`):
           If true, this descriptor will survive optimizations.

       Flags (:py:obj:`~.LLVMDIFlags`):
           Flags.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateParameterVariable(LLVMDIBuilderRef Builder, LLVMMetadataRef Scope, const char * Name, size_t NameLen, unsigned int ArgNo, LLVMMetadataRef File, unsigned int LineNo, LLVMMetadataRef Ty, LLVMBool AlwaysPreserve, LLVMDIFlags Flags)


.. py:function:: LLVMGetSubprogram(Func)

   Get the metadata of the subprogram attached to a function.

   See:
       llvm::Function::getSubprogram()

   Args:
       Func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMGetSubprogram(LLVMValueRef Func)


.. py:function:: LLVMSetSubprogram(Func, SP)

   Set the subprogram attached to a function.

   See:
       llvm::Function::setSubprogram()

   Args:
       Func (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       SP (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMSetSubprogram(LLVMValueRef Func, LLVMMetadataRef SP)


.. py:function:: LLVMDISubprogramGetLine(Subprogram)

   Get the line associated with a given subprogram.

   See:
       DISubprogram::getLine()

   Args:
       Subprogram (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The subprogram object.

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       unsigned int LLVMDISubprogramGetLine(LLVMMetadataRef Subprogram)


.. py:function:: LLVMDISubprogramReplaceType(Subprogram, SubroutineType)

   Replace the subprogram subroutine type.

   See:
       DISubprogram::replaceType()

   Args:
       Subprogram (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The subprogram object.

       SubroutineType (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The new subroutine type.

   .. rubric:: C signature

   .. code-block:: c

       void LLVMDISubprogramReplaceType(LLVMMetadataRef Subprogram, LLVMMetadataRef SubroutineType)


.. py:function:: LLVMInstructionGetDebugLoc(Inst)

   Get the debug location for the given instruction.

   See:
       llvm::Instruction::getDebugLoc()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMInstructionGetDebugLoc(LLVMValueRef Inst)


.. py:function:: LLVMInstructionSetDebugLoc(Inst, Loc)

   Set the debug location for the given instruction.

   To clear the location metadata of the given instruction, pass NULL to ``Loc.``

   See:
       llvm::Instruction::setDebugLoc()

   Args:
       Inst (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Loc (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMInstructionSetDebugLoc(LLVMValueRef Inst, LLVMMetadataRef Loc)


.. py:function:: LLVMDIBuilderCreateLabel(Builder, Context, Name, NameLen, File, LineNo, AlwaysPreserve)

   Create a new descriptor for a label

   See:
       llvm::DIBuilder::createLabel()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       Context (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           Variable name.

       NameLen (:py:obj:`~.int`):
           Length of variable name.

       File (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The file to create the label in.

       LineNo (:py:obj:`~.int`):
           Line Number.

       AlwaysPreserve (:py:obj:`~.int`):
           Preserve the label regardless of optimization.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataRef LLVMDIBuilderCreateLabel(LLVMDIBuilderRef Builder, LLVMMetadataRef Context, const char * Name, size_t NameLen, LLVMMetadataRef File, unsigned int LineNo, LLVMBool AlwaysPreserve)


.. py:function:: LLVMDIBuilderInsertLabelBefore(Builder, LabelInfo, Location, InsertBefore)

   Insert a new llvm.dbg.label intrinsic call

   See:
       llvm::DIBuilder::insertLabel()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       LabelInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The Label's debug info descriptor

       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug info location

       InsertBefore (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Location for the new intrinsic.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertLabelBefore(LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo, LLVMMetadataRef Location, LLVMValueRef InsertBefore)


.. py:function:: LLVMDIBuilderInsertLabelAtEnd(Builder, LabelInfo, Location, InsertAtEnd)

   Insert a new llvm.dbg.label intrinsic call

   See:
       llvm::DIBuilder::insertLabel()

   Args:
       Builder (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The DIBuilder.

       LabelInfo (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The Label's debug info descriptor

       Location (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           The debug info location

       InsertAtEnd (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           Location for the new intrinsic.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMDbgRecordRef LLVMDIBuilderInsertLabelAtEnd(LLVMDIBuilderRef Builder, LLVMMetadataRef LabelInfo, LLVMMetadataRef Location, LLVMBasicBlockRef InsertAtEnd)


.. py:function:: LLVMGetMetadataKind(Metadata)

   Obtain the enumerated type of a Metadata instance.

   See:
       llvm::Metadata::getMetadataID()

   Args:
       Metadata (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.int`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMMetadataKind LLVMGetMetadataKind(LLVMMetadataRef Metadata)


