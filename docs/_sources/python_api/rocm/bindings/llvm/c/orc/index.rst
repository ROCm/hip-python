rocm.bindings.llvm.c.orc
========================

.. py:module:: rocm.bindings.llvm.c.orc


Attributes
----------

.. autoapisummary::

   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionRef
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolStringPoolRef
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolStringPoolEntryRef
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolFlagsMapPairs
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolMapPairs
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolAliasMapPairs
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibRef
   rocm.bindings.llvm.c.orc.LLVMOrcCDependenceMapPairs
   rocm.bindings.llvm.c.orc.LLVMOrcCJITDylibSearchOrder
   rocm.bindings.llvm.c.orc.LLVMOrcCLookupSet
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationUnitRef
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityRef
   rocm.bindings.llvm.c.orc.LLVMOrcResourceTrackerRef
   rocm.bindings.llvm.c.orc.LLVMOrcDefinitionGeneratorRef
   rocm.bindings.llvm.c.orc.LLVMOrcLookupStateRef
   rocm.bindings.llvm.c.orc.LLVMOrcThreadSafeContextRef
   rocm.bindings.llvm.c.orc.LLVMOrcThreadSafeModuleRef
   rocm.bindings.llvm.c.orc.LLVMOrcJITTargetMachineBuilderRef
   rocm.bindings.llvm.c.orc.LLVMOrcObjectLayerRef
   rocm.bindings.llvm.c.orc.LLVMOrcObjectLinkingLayerRef
   rocm.bindings.llvm.c.orc.LLVMOrcIRTransformLayerRef
   rocm.bindings.llvm.c.orc.LLVMOrcObjectTransformLayerRef
   rocm.bindings.llvm.c.orc.LLVMOrcIndirectStubsManagerRef
   rocm.bindings.llvm.c.orc.LLVMOrcLazyCallThroughManagerRef
   rocm.bindings.llvm.c.orc.LLVMOrcDumpObjectsRef


Classes
-------

.. autoapisummary::

   rocm.bindings.llvm.c.orc.LLVMJITSymbolGenericFlags
   rocm.bindings.llvm.c.orc.LLVMJITSymbolFlags
   rocm.bindings.llvm.c.orc.LLVMJITEvaluatedSymbol
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueExecutionSession
   rocm.bindings.llvm.c.orc.LLVMOrcErrorReporterFunction
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueSymbolStringPool
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueSymbolStringPoolEntry
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolFlagsMapPair
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolMapPair
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolAliasMapEntry
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolAliasMapPair
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueJITDylib
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolsList
   rocm.bindings.llvm.c.orc.LLVMOrcCDependenceMapPair
   rocm.bindings.llvm.c.orc.LLVMOrcCSymbolDependenceGroup
   rocm.bindings.llvm.c.orc.LLVMOrcLookupKind
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibLookupFlags
   rocm.bindings.llvm.c.orc.LLVMOrcCJITDylibSearchOrderElement
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolLookupFlags
   rocm.bindings.llvm.c.orc.LLVMOrcCLookupSetElement
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueMaterializationUnit
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueMaterializationResponsibility
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationUnitMaterializeFunction
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationUnitDiscardFunction
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationUnitDestroyFunction
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueResourceTracker
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueDefinitionGenerator
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueLookupState
   rocm.bindings.llvm.c.orc.LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeCAPIDefinitionGeneratorFunction
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolPredicate
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueThreadSafeContext
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueThreadSafeModule
   rocm.bindings.llvm.c.orc.LLVMOrcGenericIRModuleOperationFunction
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueJITTargetMachineBuilder
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueObjectLayer
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueObjectLinkingLayer
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueIRTransformLayer
   rocm.bindings.llvm.c.orc.LLVMOrcIRTransformLayerTransformFunction
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueObjectTransformLayer
   rocm.bindings.llvm.c.orc.LLVMOrcObjectTransformLayerTransformFunction
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueIndirectStubsManager
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueLazyCallThroughManager
   rocm.bindings.llvm.c.orc.LLVMOrcOpaqueDumpObjects
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionLookupHandleResultFunction


Functions
---------

.. autoapisummary::

   rocm.bindings.llvm.c.orc.has_symbol
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionSetErrorReporter
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionGetSymbolStringPool
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolStringPoolClearDeadEntries
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionIntern
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionLookup
   rocm.bindings.llvm.c.orc.LLVMOrcRetainSymbolStringPoolEntry
   rocm.bindings.llvm.c.orc.LLVMOrcReleaseSymbolStringPoolEntry
   rocm.bindings.llvm.c.orc.LLVMOrcSymbolStringPoolEntryStr
   rocm.bindings.llvm.c.orc.LLVMOrcReleaseResourceTracker
   rocm.bindings.llvm.c.orc.LLVMOrcResourceTrackerTransferTo
   rocm.bindings.llvm.c.orc.LLVMOrcResourceTrackerRemove
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeDefinitionGenerator
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeMaterializationUnit
   rocm.bindings.llvm.c.orc.LLVMOrcCreateCustomMaterializationUnit
   rocm.bindings.llvm.c.orc.LLVMOrcAbsoluteSymbols
   rocm.bindings.llvm.c.orc.LLVMOrcLazyReexports
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeMaterializationResponsibility
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityGetTargetDylib
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityGetExecutionSession
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityGetSymbols
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeCSymbolFlagsMap
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityGetInitializerSymbol
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityGetRequestedSymbols
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeSymbols
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityNotifyResolved
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityNotifyEmitted
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityDefineMaterializing
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityFailMaterialization
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityReplace
   rocm.bindings.llvm.c.orc.LLVMOrcMaterializationResponsibilityDelegate
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionCreateBareJITDylib
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionCreateJITDylib
   rocm.bindings.llvm.c.orc.LLVMOrcExecutionSessionGetJITDylibByName
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibCreateResourceTracker
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibGetDefaultResourceTracker
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibDefine
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibClear
   rocm.bindings.llvm.c.orc.LLVMOrcJITDylibAddGenerator
   rocm.bindings.llvm.c.orc.LLVMOrcCreateCustomCAPIDefinitionGenerator
   rocm.bindings.llvm.c.orc.LLVMOrcLookupStateContinueLookup
   rocm.bindings.llvm.c.orc.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess
   rocm.bindings.llvm.c.orc.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath
   rocm.bindings.llvm.c.orc.LLVMOrcCreateStaticLibrarySearchGeneratorForPath
   rocm.bindings.llvm.c.orc.LLVMOrcCreateNewThreadSafeContext
   rocm.bindings.llvm.c.orc.LLVMOrcCreateNewThreadSafeContextFromLLVMContext
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeThreadSafeContext
   rocm.bindings.llvm.c.orc.LLVMOrcCreateNewThreadSafeModule
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeThreadSafeModule
   rocm.bindings.llvm.c.orc.LLVMOrcThreadSafeModuleWithModuleDo
   rocm.bindings.llvm.c.orc.LLVMOrcJITTargetMachineBuilderDetectHost
   rocm.bindings.llvm.c.orc.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeJITTargetMachineBuilder
   rocm.bindings.llvm.c.orc.LLVMOrcJITTargetMachineBuilderGetTargetTriple
   rocm.bindings.llvm.c.orc.LLVMOrcJITTargetMachineBuilderSetTargetTriple
   rocm.bindings.llvm.c.orc.LLVMOrcObjectLayerAddObjectFile
   rocm.bindings.llvm.c.orc.LLVMOrcObjectLayerAddObjectFileWithRT
   rocm.bindings.llvm.c.orc.LLVMOrcObjectLayerEmit
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeObjectLayer
   rocm.bindings.llvm.c.orc.LLVMOrcIRTransformLayerEmit
   rocm.bindings.llvm.c.orc.LLVMOrcIRTransformLayerSetTransform
   rocm.bindings.llvm.c.orc.LLVMOrcObjectTransformLayerSetTransform
   rocm.bindings.llvm.c.orc.LLVMOrcCreateLocalIndirectStubsManager
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeIndirectStubsManager
   rocm.bindings.llvm.c.orc.LLVMOrcCreateLocalLazyCallThroughManager
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeLazyCallThroughManager
   rocm.bindings.llvm.c.orc.LLVMOrcCreateDumpObjects
   rocm.bindings.llvm.c.orc.LLVMOrcDisposeDumpObjects
   rocm.bindings.llvm.c.orc.LLVMOrcDumpObjects_CallOperator


Module Contents
---------------

.. py:function:: has_symbol(name: str | bytes | bytearray) -> bool

.. py:class:: LLVMJITSymbolGenericFlags

   Bases: :py:obj:`enum.IntEnum`


   Represents generic linkage flags for a symbol definition.
       


   .. py:attribute:: LLVMJITSymbolGenericFlagsNone
      :type:  int


   .. py:attribute:: LLVMJITSymbolGenericFlagsExported
      :type:  int


   .. py:attribute:: LLVMJITSymbolGenericFlagsWeak
      :type:  int


   .. py:attribute:: LLVMJITSymbolGenericFlagsCallable
      :type:  int


   .. py:attribute:: LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly
      :type:  int


.. py:class:: LLVMJITSymbolFlags(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents the linkage flags for a symbol definition.
       


   .. py:attribute:: GenericFlags
      :type:  Any


   .. py:attribute:: TargetFlags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMJITEvaluatedSymbol(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents an evaluated symbol address and flags.
       


   .. py:attribute:: Address
      :type:  Any


   .. py:attribute:: Flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMOrcOpaqueExecutionSession(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcExecutionSessionRef

.. py:class:: LLVMOrcErrorReporterFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Error reporter function.
       


.. py:class:: LLVMOrcOpaqueSymbolStringPool(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcSymbolStringPoolRef

.. py:class:: LLVMOrcOpaqueSymbolStringPoolEntry(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcSymbolStringPoolEntryRef

.. py:class:: LLVMOrcCSymbolFlagsMapPair(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a pair of a symbol name and LLVMJITSymbolFlags.
       


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: Flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCSymbolFlagsMapPairs

.. py:class:: LLVMOrcCSymbolMapPair(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a pair of a symbol name and an evaluated symbol.
       


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: Sym
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCSymbolMapPairs

.. py:class:: LLVMOrcCSymbolAliasMapEntry(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a SymbolAliasMapEntry
       


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: Flags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMOrcCSymbolAliasMapPair(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a pair of a symbol name and SymbolAliasMapEntry.
       


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: Entry
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCSymbolAliasMapPairs

.. py:class:: LLVMOrcOpaqueJITDylib(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcJITDylibRef

.. py:class:: LLVMOrcCSymbolsList(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a list of LLVMOrcSymbolStringPoolEntryRef and the associated
   length.


   .. py:attribute:: Symbols
      :type:  Any


   .. py:attribute:: Length
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMOrcCDependenceMapPair(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Represents a pair of a JITDylib and LLVMOrcCSymbolsList.
       


   .. py:attribute:: JD
      :type:  Any


   .. py:attribute:: Names
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCDependenceMapPairs

.. py:class:: LLVMOrcCSymbolDependenceGroup(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A set of symbols that share dependencies.
       


   .. py:attribute:: Symbols
      :type:  Any


   .. py:attribute:: Dependencies
      :type:  Any


   .. py:attribute:: NumDependencies
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:class:: LLVMOrcLookupKind

   Bases: :py:obj:`enum.IntEnum`


   Lookup kind.

   This can be used by definition generators when deciding whether
   to produce a definition for a requested symbol.

   This enum should be kept in sync with llvm::orc::LookupKind.


   .. py:attribute:: LLVMOrcLookupKindStatic
      :type:  int


   .. py:attribute:: LLVMOrcLookupKindDLSym
      :type:  int


.. py:class:: LLVMOrcJITDylibLookupFlags

   Bases: :py:obj:`enum.IntEnum`


   JITDylib lookup flags.

   This can be used by definition generators when
   deciding whether to produce a definition for a requested symbol.

   This enum should be kept in sync with llvm::orc::JITDylibLookupFlags.


   .. py:attribute:: LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly
      :type:  int


   .. py:attribute:: LLVMOrcJITDylibLookupFlagsMatchAllSymbols
      :type:  int


.. py:class:: LLVMOrcCJITDylibSearchOrderElement(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   An element type for a JITDylib search order.
       


   .. py:attribute:: JD
      :type:  Any


   .. py:attribute:: JDLookupFlags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCJITDylibSearchOrder

.. py:class:: LLVMOrcSymbolLookupFlags

   Bases: :py:obj:`enum.IntEnum`


   Symbol lookup flags for lookup sets.

   This should be kept in sync with
   llvm::orc::SymbolLookupFlags.


   .. py:attribute:: LLVMOrcSymbolLookupFlagsRequiredSymbol
      :type:  int


   .. py:attribute:: LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol
      :type:  int


.. py:class:: LLVMOrcCLookupSetElement(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   An element type for a symbol lookup set.
       


   .. py:attribute:: Name
      :type:  Any


   .. py:attribute:: LookupFlags
      :type:  Any


   .. py:method:: PROPERTIES() -> list[str]
      :staticmethod:



   .. py:method:: fromObj(pyobj) -> Any
      :staticmethod:



   .. py:method:: allocate(count: int = 1) -> Any
      :staticmethod:



   .. py:method:: c_sizeof() -> int


   .. py:method:: as_c_void_p() -> Any


.. py:data:: LLVMOrcCLookupSet

.. py:class:: LLVMOrcOpaqueMaterializationUnit(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcMaterializationUnitRef

.. py:class:: LLVMOrcOpaqueMaterializationResponsibility(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcMaterializationResponsibilityRef

.. py:class:: LLVMOrcMaterializationUnitMaterializeFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A MaterializationUnit materialize callback.

   Ownership of the Ctx and MR arguments passes to the callback which must
   adhere to the LLVMOrcMaterializationResponsibilityRef contract (see comment
   for that type).

   If this callback is called then the LLVMOrcMaterializationUnitDestroy
   callback will NOT be called.


.. py:class:: LLVMOrcMaterializationUnitDiscardFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A MaterializationUnit discard callback.

   Ownership of JD and Symbol remain with the caller: These arguments should
   not be disposed of or released.


.. py:class:: LLVMOrcMaterializationUnitDestroyFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A MaterializationUnit destruction callback.

   If a custom MaterializationUnit is destroyed before its Materialize
   function is called then this function will be called to provide an
   opportunity for the underlying program representation to be destroyed.


.. py:class:: LLVMOrcOpaqueResourceTracker(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcResourceTrackerRef

.. py:class:: LLVMOrcOpaqueDefinitionGenerator(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcDefinitionGeneratorRef

.. py:class:: LLVMOrcOpaqueLookupState(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcLookupStateRef

.. py:class:: LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A custom generator function.

   This can be used to create a custom generator
   object using LLVMOrcCreateCustomCAPIDefinitionGenerator. The resulting
   object can be attached to a JITDylib, via LLVMOrcJITDylibAddGenerator, to
   receive callbacks when lookups fail to match existing definitions.

   GeneratorObj will contain the address of the custom generator object.

   Ctx will contain the context object passed to
   LLVMOrcCreateCustomCAPIDefinitionGenerator.

   LookupState will contain a pointer to an LLVMOrcLookupStateRef object. This
   can optionally be modified to make the definition generation process
   asynchronous: If the LookupStateRef value is copied, and the original
   LLVMOrcLookupStateRef set to null, the lookup will be suspended. Once the
   asynchronous definition process has been completed clients must call
   LLVMOrcLookupStateContinueLookup to continue the lookup (this should be
   done unconditionally, even if errors have occurred in the mean time, to
   free the lookup state memory and notify the query object of the failures).
   If LookupState is captured this function must return LLVMErrorSuccess.

   The Kind argument can be inspected to determine the lookup kind (e.g.
   as-if-during-static-link, or as-if-during-dlsym).

   The JD argument specifies which JITDylib the definitions should be generated
   into.

   The JDLookupFlags argument can be inspected to determine whether the original
   lookup included non-exported symbols.

   Finally, the LookupSet argument contains the set of symbols that could not
   be found in JD already (the set of generation candidates).


.. py:class:: LLVMOrcDisposeCAPIDefinitionGeneratorFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Disposer for a custom generator.

   Will be called by ORC when the JITDylib that the generator is attached to
   is destroyed.


.. py:class:: LLVMOrcSymbolPredicate(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Predicate function for SymbolStringPoolEntries.
       


.. py:class:: LLVMOrcOpaqueThreadSafeContext(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcThreadSafeContextRef

.. py:class:: LLVMOrcOpaqueThreadSafeModule(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcThreadSafeModuleRef

.. py:class:: LLVMOrcGenericIRModuleOperationFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A function for inspecting/mutating IR modules, suitable for use with
   LLVMOrcThreadSafeModuleWithModuleDo.


.. py:class:: LLVMOrcOpaqueJITTargetMachineBuilder(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcJITTargetMachineBuilderRef

.. py:class:: LLVMOrcOpaqueObjectLayer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcObjectLayerRef

.. py:class:: LLVMOrcOpaqueObjectLinkingLayer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcObjectLinkingLayerRef

.. py:class:: LLVMOrcOpaqueIRTransformLayer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcIRTransformLayerRef

.. py:class:: LLVMOrcIRTransformLayerTransformFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A function for applying transformations as part of an transform layer.

   Implementations of this type are responsible for managing the lifetime
   of the Module pointed to by ModInOut: If the LLVMModuleRef value is
   overwritten then the function is responsible for disposing of the incoming
   module. If the module is simply accessed/mutated in-place then ownership
   returns to the caller and the function does not need to do any lifetime
   management.

   Clients can call LLVMOrcLLJITGetIRTransformLayer to obtain the transform
   layer of a LLJIT instance, and use LLVMOrcIRTransformLayerSetTransform
   to set the function. This can be used to override the default transform
   layer.


.. py:class:: LLVMOrcOpaqueObjectTransformLayer(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcObjectTransformLayerRef

.. py:class:: LLVMOrcObjectTransformLayerTransformFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   A function for applying transformations to an object file buffer.

   Implementations of this type are responsible for managing the lifetime
   of the memory buffer pointed to by ObjInOut: If the LLVMMemoryBufferRef
   value is overwritten then the function is responsible for disposing of the
   incoming buffer. If the buffer is simply accessed/mutated in-place then
   ownership returns to the caller and the function does not need to do any
   lifetime management.

   The transform is allowed to return an error, in which case the ObjInOut
   buffer should be disposed of and set to null.


.. py:class:: LLVMOrcOpaqueIndirectStubsManager(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcIndirectStubsManagerRef

.. py:class:: LLVMOrcOpaqueLazyCallThroughManager(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcLazyCallThroughManagerRef

.. py:class:: LLVMOrcOpaqueDumpObjects(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   (No short description)
       


.. py:data:: LLVMOrcDumpObjectsRef

.. py:function:: LLVMOrcExecutionSessionSetErrorReporter(ES, ReportError, Ctx)

   Attach a custom error reporter function to the ExecutionSession.

   The error reporter will be called to deliver failure notices that can not be
   directly reported to a caller. For example, failure to resolve symbols in
   the JIT linker is typically reported via the error reporter (callers
   requesting definitions from the JIT will typically be delivered a
   FailureToMaterialize error instead).

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       ReportError (:py:obj:`~.LLVMOrcErrorReporterFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcExecutionSessionSetErrorReporter(LLVMOrcExecutionSessionRef ES, LLVMOrcErrorReporterFunction ReportError, void * Ctx)


.. py:function:: LLVMOrcExecutionSessionGetSymbolStringPool(ES)

   Return a reference to the SymbolStringPool for an ExecutionSession.

   Ownership of the pool remains with the ExecutionSession: The caller is
   not required to free the pool.

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcSymbolStringPoolRef LLVMOrcExecutionSessionGetSymbolStringPool(LLVMOrcExecutionSessionRef ES)


.. py:function:: LLVMOrcSymbolStringPoolClearDeadEntries(SSP)

   Clear all unreferenced symbol string pool entries.

   This can be called at any time to release unused entries in the
   ExecutionSession's string pool. Since it locks the pool (preventing
   interning of any new strings) it is recommended that it only be called
   infrequently, ideally when the caller has reason to believe that some
   entries will have become unreferenced, e.g. after removing a module or
   closing a JITDylib.

   Args:
       SSP (:py:obj:`~.LLVMOrcOpaqueSymbolStringPool`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcSymbolStringPoolClearDeadEntries(LLVMOrcSymbolStringPoolRef SSP)


.. py:function:: LLVMOrcExecutionSessionIntern(ES, Name)

   Intern a string in the ExecutionSession's SymbolStringPool and return a
   reference to it.

   This increments the ref-count of the pool entry, and the
   returned value should be released once the client is done with it by
   calling LLVMOrcReleaseSymbolStringPoolEntry.

   Since strings are uniqued within the SymbolStringPool
   LLVMOrcSymbolStringPoolEntryRefs can be compared by value to test string
   equality.

   Note that this function does not perform linker-mangling on the string.

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcSymbolStringPoolEntryRef LLVMOrcExecutionSessionIntern(LLVMOrcExecutionSessionRef ES, const char * Name)


.. py:class:: LLVMOrcExecutionSessionLookupHandleResultFunction(*args, **kwargs)

   Bases: :py:obj:`rocm.bindings.util.types.Pointer`


   Callback type for ExecutionSession lookups.

   If Err is LLVMErrorSuccess then Result will contain a pointer to a
   list of ( SymbolStringPtr, JITEvaluatedSymbol ) pairs of length NumPairs.

   If Err is a failure value then Result and Ctx are undefined and should
   not be accessed. The Callback is responsible for handling the error
   value (e.g. by calling LLVMGetErrorMessage + LLVMDisposeErrorMessage).

   The caller retains ownership of the Result array and will release all
   contained symbol names. Clients are responsible for retaining any symbol
   names that they wish to hold after the function returns.


.. py:function:: LLVMOrcExecutionSessionLookup(ES, K, SearchOrder, SearchOrderSize, Symbols, SymbolsSize, HandleResult, Ctx)

   Look up symbols in an execution session.

   This is a wrapper around the general ExecutionSession::lookup function.

   The SearchOrder argument contains a list of (JITDylibs, JITDylibSearchFlags)
   pairs that describe the search order. The JITDylibs will be searched in the
   given order to try to find the symbols in the Symbols argument.

   The Symbols argument should contain a null-terminated array of
   (SymbolStringPtr, SymbolLookupFlags) pairs describing the symbols to be
   searched for. This function takes ownership of the elements of the Symbols
   array. The Name fields of the Symbols elements are taken to have been
   retained by the client for this function. The client should *not* release the
   Name fields, but are still responsible for destroying the array itself.

   The HandleResult function will be called once all searched for symbols have
   been found, or an error occurs. The HandleResult function will be passed an
   LLVMErrorRef indicating success or failure, and (on success) a
   null-terminated LLVMOrcCSymbolMapPairs array containing the function result,
   and the Ctx value passed to the lookup function.

   The client is fully responsible for managing the lifetime of the Ctx object.
   A common idiom is to allocate the context prior to the lookup and deallocate
   it in the handler.

   THIS API IS EXPERIMENTAL AND LIKELY TO CHANGE IN THE NEAR FUTURE!

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       K (:py:obj:`~.LLVMOrcLookupKind`):
           (undocumented)

       SearchOrder (:py:obj:`~.LLVMOrcCJITDylibSearchOrderElement`/:py:obj:`~.object`):
           (undocumented)

       SearchOrderSize (:py:obj:`~.int`):
           (undocumented)

       Symbols (:py:obj:`~.LLVMOrcCLookupSetElement`/:py:obj:`~.object`):
           (undocumented)

       SymbolsSize (:py:obj:`~.int`):
           (undocumented)

       HandleResult (:py:obj:`~.LLVMOrcExecutionSessionLookupHandleResultFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcExecutionSessionLookup(LLVMOrcExecutionSessionRef ES, LLVMOrcLookupKind K, LLVMOrcCJITDylibSearchOrder SearchOrder, size_t SearchOrderSize, LLVMOrcCLookupSet Symbols, size_t SymbolsSize, LLVMOrcExecutionSessionLookupHandleResultFunction HandleResult, void * Ctx)


.. py:function:: LLVMOrcRetainSymbolStringPoolEntry(S)

   Increments the ref-count for a SymbolStringPool entry.

   Args:
       S (:py:obj:`~.LLVMOrcOpaqueSymbolStringPoolEntry`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcRetainSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)


.. py:function:: LLVMOrcReleaseSymbolStringPoolEntry(S)

   Reduces the ref-count for of a SymbolStringPool entry.

   Args:
       S (:py:obj:`~.LLVMOrcOpaqueSymbolStringPoolEntry`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcReleaseSymbolStringPoolEntry(LLVMOrcSymbolStringPoolEntryRef S)


.. py:function:: LLVMOrcSymbolStringPoolEntryStr(S)

   Return the c-string for the given symbol.

   This string will remain valid until
   the entry is freed (once all LLVMOrcSymbolStringPoolEntryRefs have been
   released).

   Args:
       S (:py:obj:`~.LLVMOrcOpaqueSymbolStringPoolEntry`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       const char * LLVMOrcSymbolStringPoolEntryStr(LLVMOrcSymbolStringPoolEntryRef S)


.. py:function:: LLVMOrcReleaseResourceTracker(RT)

   Reduces the ref-count of a ResourceTracker.

   Args:
       RT (:py:obj:`~.LLVMOrcOpaqueResourceTracker`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcReleaseResourceTracker(LLVMOrcResourceTrackerRef RT)


.. py:function:: LLVMOrcResourceTrackerTransferTo(SrcRT, DstRT)

   Transfers tracking of all resources associated with resource tracker SrcRT
   to resource tracker DstRT.

   Args:
       SrcRT (:py:obj:`~.LLVMOrcOpaqueResourceTracker`/:py:obj:`~.object`):
           (undocumented)

       DstRT (:py:obj:`~.LLVMOrcOpaqueResourceTracker`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcResourceTrackerTransferTo(LLVMOrcResourceTrackerRef SrcRT, LLVMOrcResourceTrackerRef DstRT)


.. py:function:: LLVMOrcResourceTrackerRemove(RT)

   Remove all resources associated with the given tracker.

   See
   ResourceTracker::remove().

   Args:
       RT (:py:obj:`~.LLVMOrcOpaqueResourceTracker`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcResourceTrackerRemove(LLVMOrcResourceTrackerRef RT)


.. py:function:: LLVMOrcDisposeDefinitionGenerator(DG)

   Dispose of a JITDylib::DefinitionGenerator.

   This should only be called if
   ownership has not been passed to a JITDylib (e.g. because some error
   prevented the client from calling LLVMOrcJITDylibAddGenerator).

   Args:
       DG (:py:obj:`~.LLVMOrcOpaqueDefinitionGenerator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeDefinitionGenerator(LLVMOrcDefinitionGeneratorRef DG)


.. py:function:: LLVMOrcDisposeMaterializationUnit(MU)

   Dispose of a MaterializationUnit.

   Args:
       MU (:py:obj:`~.LLVMOrcOpaqueMaterializationUnit`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeMaterializationUnit(LLVMOrcMaterializationUnitRef MU)


.. py:function:: LLVMOrcCreateCustomMaterializationUnit(Name, Ctx, Syms, NumSyms, InitSym, Materialize, Discard, Destroy)

   Create a custom MaterializationUnit.

   Name is a name for this MaterializationUnit to be used for identification
   and logging purposes (e.g. if this MaterializationUnit produces an
   object buffer then the name of that buffer will be derived from this name).

   The Syms list contains the names and linkages of the symbols provided by this
   unit. This function takes ownership of the elements of the Syms array. The
   Name fields of the array elements are taken to have been retained for this
   function. The client should *not* release the elements of the array, but is
   still responsible for destroying the array itself.

   The InitSym argument indicates whether or not this MaterializationUnit
   contains static initializers. If three are no static initializers (the common
   case) then this argument should be null. If there are static initializers
   then InitSym should be set to a unique name that also appears in the Syms
   list with the LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly flag
   set. This function takes ownership of the InitSym, which should have been
   retained twice on behalf of this function: once for the Syms entry and once
   for InitSym. If clients wish to use the InitSym value after this function
   returns they must retain it once more for themselves.

   If any of the symbols in the Syms list is looked up then the Materialize
   function will be called.

   If any of the symbols in the Syms list is overridden then the Discard
   function will be called.

   The caller owns the underling MaterializationUnit and is responsible for
   either passing it to a JITDylib (via LLVMOrcJITDylibDefine) or disposing
   of it by calling LLVMOrcDisposeMaterializationUnit.

   Args:
       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Syms (:py:obj:`~.LLVMOrcCSymbolFlagsMapPair`/:py:obj:`~.object`):
           (undocumented)

       NumSyms (:py:obj:`~.int`):
           (undocumented)

       InitSym (:py:obj:`~.LLVMOrcOpaqueSymbolStringPoolEntry`/:py:obj:`~.object`):
           (undocumented)

       Materialize (:py:obj:`~.LLVMOrcMaterializationUnitMaterializeFunction`/:py:obj:`~.object`):
           (undocumented)

       Discard (:py:obj:`~.LLVMOrcMaterializationUnitDiscardFunction`/:py:obj:`~.object`):
           (undocumented)

       Destroy (:py:obj:`~.LLVMOrcMaterializationUnitDestroyFunction`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcMaterializationUnitRef LLVMOrcCreateCustomMaterializationUnit(const char * Name, void * Ctx, LLVMOrcCSymbolFlagsMapPairs Syms, size_t NumSyms, LLVMOrcSymbolStringPoolEntryRef InitSym, LLVMOrcMaterializationUnitMaterializeFunction Materialize, LLVMOrcMaterializationUnitDiscardFunction Discard, LLVMOrcMaterializationUnitDestroyFunction Destroy)


.. py:function:: LLVMOrcAbsoluteSymbols(Syms, NumPairs)

   Create a MaterializationUnit to define the given symbols as pointing to
   the corresponding raw addresses.

   This function takes ownership of the elements of the Syms array. The Name
   fields of the array elements are taken to have been retained for this
   function. This allows the following pattern...

     size_t NumPairs;
     LLVMOrcCSymbolMapPairs Sym;
     -- Build Syms array --
     LLVMOrcMaterializationUnitRef MU =
         LLVMOrcAbsoluteSymbols(Syms, NumPairs);

   ... without requiring cleanup of the elements of the Sym array afterwards.

   The client is still responsible for deleting the Sym array itself.

   If a client wishes to reuse elements of the Sym array after this call they
   must explicitly retain each of the elements for themselves.

   Args:
       Syms (:py:obj:`~.LLVMOrcCSymbolMapPair`/:py:obj:`~.object`):
           (undocumented)

       NumPairs (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcMaterializationUnitRef LLVMOrcAbsoluteSymbols(LLVMOrcCSymbolMapPairs Syms, size_t NumPairs)


.. py:function:: LLVMOrcLazyReexports(LCTM, ISM, SourceRef, CallableAliases, NumPairs)

   Create a MaterializationUnit to define lazy re-expots.

   These are callable
   entry points that call through to the given symbols.

   This function takes ownership of the CallableAliases array. The Name
   fields of the array elements are taken to have been retained for this
   function. This allows the following pattern...

     size_t NumPairs;
     LLVMOrcCSymbolAliasMapPairs CallableAliases;
     -- Build CallableAliases array --
     LLVMOrcMaterializationUnitRef MU =
        LLVMOrcLazyReexports(LCTM, ISM, JD, CallableAliases, NumPairs);

   ... without requiring cleanup of the elements of the CallableAliases array afterwards.

   The client is still responsible for deleting the CallableAliases array itself.

   If a client wishes to reuse elements of the CallableAliases array after this call they
   must explicitly retain each of the elements for themselves.

   Args:
       LCTM (:py:obj:`~.LLVMOrcOpaqueLazyCallThroughManager`/:py:obj:`~.object`):
           (undocumented)

       ISM (:py:obj:`~.LLVMOrcOpaqueIndirectStubsManager`/:py:obj:`~.object`):
           (undocumented)

       SourceRef (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

       CallableAliases (:py:obj:`~.LLVMOrcCSymbolAliasMapPair`/:py:obj:`~.object`):
           (undocumented)

       NumPairs (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcMaterializationUnitRef LLVMOrcLazyReexports(LLVMOrcLazyCallThroughManagerRef LCTM, LLVMOrcIndirectStubsManagerRef ISM, LLVMOrcJITDylibRef SourceRef, LLVMOrcCSymbolAliasMapPairs CallableAliases, size_t NumPairs)


.. py:function:: LLVMOrcDisposeMaterializationResponsibility(MR)

   Disposes of the passed MaterializationResponsibility object.

   This should only be done after the symbols covered by the object have either
   been resolved and emitted (via
   LLVMOrcMaterializationResponsibilityNotifyResolved and
   LLVMOrcMaterializationResponsibilityNotifyEmitted) or failed (via
   LLVMOrcMaterializationResponsibilityFailMaterialization).

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeMaterializationResponsibility(LLVMOrcMaterializationResponsibilityRef MR)


.. py:function:: LLVMOrcMaterializationResponsibilityGetTargetDylib(MR)

   Returns the target JITDylib that these symbols are being materialized into.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcJITDylibRef LLVMOrcMaterializationResponsibilityGetTargetDylib(LLVMOrcMaterializationResponsibilityRef MR)


.. py:function:: LLVMOrcMaterializationResponsibilityGetExecutionSession(MR)

   Returns the ExecutionSession for this MaterializationResponsibility.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcExecutionSessionRef LLVMOrcMaterializationResponsibilityGetExecutionSession(LLVMOrcMaterializationResponsibilityRef MR)


.. py:function:: LLVMOrcMaterializationResponsibilityGetSymbols(MR, NumPairs)

   Returns the symbol flags map for this responsibility instance.

   The length of the array is returned in NumPairs and the caller is responsible
   for the returned memory and needs to call LLVMOrcDisposeCSymbolFlagsMap.

   To use the returned symbols beyond the livetime of the
   MaterializationResponsibility requires the caller to retain the symbols
   explicitly.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       NumPairs (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcCSymbolFlagsMapPairs LLVMOrcMaterializationResponsibilityGetSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t * NumPairs)


.. py:function:: LLVMOrcDisposeCSymbolFlagsMap(Pairs)

   Disposes of the passed LLVMOrcCSymbolFlagsMap.

   Does not release the entries themselves.

   Args:
       Pairs (:py:obj:`~.LLVMOrcCSymbolFlagsMapPair`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeCSymbolFlagsMap(LLVMOrcCSymbolFlagsMapPairs Pairs)


.. py:function:: LLVMOrcMaterializationResponsibilityGetInitializerSymbol(MR)

   Returns the initialization pseudo-symbol, if any.

   This symbol will also
   be present in the SymbolFlagsMap for this MaterializationResponsibility
   object.

   The returned symbol is not retained over any mutating operation of the
   MaterializationResponsbility or beyond the lifetime thereof.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcSymbolStringPoolEntryRef LLVMOrcMaterializationResponsibilityGetInitializerSymbol(LLVMOrcMaterializationResponsibilityRef MR)


.. py:function:: LLVMOrcMaterializationResponsibilityGetRequestedSymbols(MR, NumSymbols)

   Returns the names of any symbols covered by this
   MaterializationResponsibility object that have queries pending.

   This
   information can be used to return responsibility for unrequested symbols
   back to the JITDylib via the delegate method.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       NumSymbols (:py:obj:`~.rocm.bindings.util.types.PointerToUInt64`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcSymbolStringPoolEntryRef * LLVMOrcMaterializationResponsibilityGetRequestedSymbols(LLVMOrcMaterializationResponsibilityRef MR, size_t * NumSymbols)


.. py:function:: LLVMOrcDisposeSymbols(Symbols)

   Disposes of the passed LLVMOrcSymbolStringPoolEntryRef* .

   Does not release the symbols themselves.

   Args:
       Symbols (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeSymbols(LLVMOrcSymbolStringPoolEntryRef * Symbols)


.. py:function:: LLVMOrcMaterializationResponsibilityNotifyResolved(MR, Symbols, NumPairs)

   Notifies the target JITDylib that the given symbols have been resolved.

   This will update the given symbols' addresses in the JITDylib, and notify
   any pending queries on the given symbols of their resolution. The given
   symbols must be ones covered by this MaterializationResponsibility
   instance. Individual calls to this method may resolve a subset of the
   symbols, but all symbols must have been resolved prior to calling emit.

   This method will return an error if any symbols being resolved have been
   moved to the error state due to the failure of a dependency. If this
   method returns an error then clients should log it and call
   LLVMOrcMaterializationResponsibilityFailMaterialization. If no dependencies
   have been registered for the symbols covered by this
   MaterializationResponsibility then this method is guaranteed to return
   LLVMErrorSuccess.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       Symbols (:py:obj:`~.LLVMOrcCSymbolMapPair`/:py:obj:`~.object`):
           (undocumented)

       NumPairs (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyResolved(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolMapPairs Symbols, size_t NumPairs)


.. py:function:: LLVMOrcMaterializationResponsibilityNotifyEmitted(MR, SymbolDepGroups, NumSymbolDepGroups)

   Notifies the target JITDylib (and any pending queries on that JITDylib)
   that all symbols covered by this MaterializationResponsibility instance
   have been emitted.

   This function takes ownership of the symbols in the Dependencies struct.
   This allows the following pattern...

     LLVMOrcSymbolStringPoolEntryRef Names[] = {...};
     LLVMOrcCDependenceMapPair Dependence = {JD, {Names, sizeof(Names)}}
     LLVMOrcMaterializationResponsibilityAddDependencies(JD, Name, &Dependence,
   1);

   ... without requiring cleanup of the elements of the Names array afterwards.

   The client is still responsible for deleting the Dependencies.Names arrays,
   and the Dependencies array itself.

   This method will return an error if any symbols being resolved have been
   moved to the error state due to the failure of a dependency. If this
   method returns an error then clients should log it and call
   LLVMOrcMaterializationResponsibilityFailMaterialization.
   If no dependencies have been registered for the symbols covered by this
   MaterializationResponsibility then this method is guaranteed to return
   LLVMErrorSuccess.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       SymbolDepGroups (:py:obj:`~.LLVMOrcCSymbolDependenceGroup`/:py:obj:`~.object`):
           (undocumented)

       NumSymbolDepGroups (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcMaterializationResponsibilityNotifyEmitted(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolDependenceGroup * SymbolDepGroups, size_t NumSymbolDepGroups)


.. py:function:: LLVMOrcMaterializationResponsibilityDefineMaterializing(MR, Pairs, NumPairs)

   Attempt to claim responsibility for new definitions.

   This method can be
   used to claim responsibility for symbols that are added to a
   materialization unit during the compilation process (e.g. literal pool
   symbols). Symbol linkage rules are the same as for symbols that are
   defined up front: duplicate strong definitions will result in errors.
   Duplicate weak definitions will be discarded (in which case they will
   not be added to this responsibility instance).

   This method can be used by materialization units that want to add
   additional symbols at materialization time (e.g. stubs, compile
   callbacks, metadata)

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       Pairs (:py:obj:`~.LLVMOrcCSymbolFlagsMapPair`/:py:obj:`~.object`):
           (undocumented)

       NumPairs (:py:obj:`~.int`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcMaterializationResponsibilityDefineMaterializing(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcCSymbolFlagsMapPairs Pairs, size_t NumPairs)


.. py:function:: LLVMOrcMaterializationResponsibilityFailMaterialization(MR)

   Notify all not-yet-emitted covered by this MaterializationResponsibility
   instance that an error has occurred.

   This will remove all symbols covered by this MaterializationResponsibility
   from the target JITDylib, and send an error to any queries waiting on
   these symbols.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcMaterializationResponsibilityFailMaterialization(LLVMOrcMaterializationResponsibilityRef MR)


.. py:function:: LLVMOrcMaterializationResponsibilityReplace(MR, MU)

   Transfers responsibility to the given MaterializationUnit for all
   symbols defined by that MaterializationUnit.

   This allows
   materializers to break up work based on run-time information (e.g.
   by introspecting which symbols have actually been looked up and
   materializing only those).

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       MU (:py:obj:`~.LLVMOrcOpaqueMaterializationUnit`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcMaterializationResponsibilityReplace(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcMaterializationUnitRef MU)


.. py:function:: LLVMOrcMaterializationResponsibilityDelegate(MR, Symbols, NumSymbols, Result)

   Delegates responsibility for the given symbols to the returned
   materialization responsibility.

   Useful for breaking up work between
   threads, or different kinds of materialization processes.

   The caller retains responsibility of the the passed
   MaterializationResponsibility.

   Args:
       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       Symbols (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       NumSymbols (:py:obj:`~.int`):
           (undocumented)

       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcMaterializationResponsibilityDelegate(LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcSymbolStringPoolEntryRef * Symbols, size_t NumSymbols, LLVMOrcMaterializationResponsibilityRef * Result)


.. py:function:: LLVMOrcExecutionSessionCreateBareJITDylib(ES, Name)

   Create a "bare" JITDylib.

   The client is responsible for ensuring that the JITDylib's name is unique,
   e.g. by calling LLVMOrcExecutionSessionGetJTIDylibByName first.

   This call does not install any library code or symbols into the newly
   created JITDylib. The client is responsible for all configuration.

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcJITDylibRef LLVMOrcExecutionSessionCreateBareJITDylib(LLVMOrcExecutionSessionRef ES, const char * Name)


.. py:function:: LLVMOrcExecutionSessionCreateJITDylib(ES, Result, Name)

   Create a JITDylib.

   The client is responsible for ensuring that the JITDylib's name is unique,
   e.g. by calling LLVMOrcExecutionSessionGetJTIDylibByName first.

   If a Platform is attached to the ExecutionSession then
   Platform::setupJITDylib will be called to install standard platform symbols
   (e.g. standard library interposes). If no Platform is installed then this
   call is equivalent to LLVMExecutionSessionRefCreateBareJITDylib and will
   always return success.

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcExecutionSessionCreateJITDylib(LLVMOrcExecutionSessionRef ES, LLVMOrcJITDylibRef * Result, const char * Name)


.. py:function:: LLVMOrcExecutionSessionGetJITDylibByName(ES, Name)

   Returns the JITDylib with the given name, or NULL if no such JITDylib
   exists.

   Args:
       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       Name (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcJITDylibRef LLVMOrcExecutionSessionGetJITDylibByName(LLVMOrcExecutionSessionRef ES, const char * Name)


.. py:function:: LLVMOrcJITDylibCreateResourceTracker(JD)

   Return a reference to a newly created resource tracker associated with JD.

   The tracker is returned with an initial ref-count of 1, and must be released
   with LLVMOrcReleaseResourceTracker when no longer needed.

   Args:
       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcResourceTrackerRef LLVMOrcJITDylibCreateResourceTracker(LLVMOrcJITDylibRef JD)


.. py:function:: LLVMOrcJITDylibGetDefaultResourceTracker(JD)

   Return a reference to the default resource tracker for the given JITDylib.

   This operation will increase the retain count of the tracker: Clients should
   call LLVMOrcReleaseResourceTracker when the result is no longer needed.

   Args:
       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcResourceTrackerRef LLVMOrcJITDylibGetDefaultResourceTracker(LLVMOrcJITDylibRef JD)


.. py:function:: LLVMOrcJITDylibDefine(JD, MU)

   Add the given MaterializationUnit to the given JITDylib.

   If this operation succeeds then JITDylib JD will take ownership of MU.
   If the operation fails then ownership remains with the caller who should
   call LLVMOrcDisposeMaterializationUnit to destroy it.

   Args:
       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

       MU (:py:obj:`~.LLVMOrcOpaqueMaterializationUnit`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcJITDylibDefine(LLVMOrcJITDylibRef JD, LLVMOrcMaterializationUnitRef MU)


.. py:function:: LLVMOrcJITDylibClear(JD)

   Calls remove on all trackers associated with this JITDylib, see
   JITDylib::clear().

   Args:
       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcJITDylibClear(LLVMOrcJITDylibRef JD)


.. py:function:: LLVMOrcJITDylibAddGenerator(JD, DG)

   Add a DefinitionGenerator to the given JITDylib.

   The JITDylib will take ownership of the given generator: The client is no
   longer responsible for managing its memory.

   Args:
       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

       DG (:py:obj:`~.LLVMOrcOpaqueDefinitionGenerator`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcJITDylibAddGenerator(LLVMOrcJITDylibRef JD, LLVMOrcDefinitionGeneratorRef DG)


.. py:function:: LLVMOrcCreateCustomCAPIDefinitionGenerator(F, Ctx, Dispose)

   Create a custom generator.

   The F argument will be used to implement the DefinitionGenerator's
   tryToGenerate method (see
   LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction).

   Ctx is a context object that will be passed to F. This argument is
   permitted to be null.

   Dispose is the disposal function for Ctx. This argument is permitted to be
   null (in which case the client is responsible for the lifetime of Ctx).

   Args:
       F (:py:obj:`~.LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       Dispose (:py:obj:`~.LLVMOrcDisposeCAPIDefinitionGeneratorFunction`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcDefinitionGeneratorRef LLVMOrcCreateCustomCAPIDefinitionGenerator(LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction F, void * Ctx, LLVMOrcDisposeCAPIDefinitionGeneratorFunction Dispose)


.. py:function:: LLVMOrcLookupStateContinueLookup(S, Err)

   Continue a lookup that was suspended in a generator (see
   LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction).

   Args:
       S (:py:obj:`~.LLVMOrcOpaqueLookupState`/:py:obj:`~.object`):
           (undocumented)

       Err (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcLookupStateContinueLookup(LLVMOrcLookupStateRef S, LLVMErrorRef Err)


.. py:function:: LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(Result, GlobalPrefx, Filter, FilterCtx)

   Get a DynamicLibrarySearchGenerator that will reflect process symbols into
   the JITDylib.

   On success the resulting generator is owned by the client.
   Ownership is typically transferred by adding the instance to a JITDylib
   using LLVMOrcJITDylibAddGenerator,

   The GlobalPrefix argument specifies the character that appears on the front
   of linker-mangled symbols for the target platform (e.g. '_' on MachO).
   If non-null, this character will be stripped from the start of all symbol
   strings before passing the remaining substring to dlsym.

   The optional Filter and Ctx arguments can be used to supply a symbol name
   filter: Only symbols for which the filter returns true will be visible to
   JIT'd code. If the Filter argument is null then all process symbols will
   be visible to JIT'd code. Note that the symbol name passed to the Filter
   function is the full mangled symbol: The client is responsible for stripping
   the global prefix if present.

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       GlobalPrefx (:py:obj:`~.int`):
           (undocumented)

       Filter (:py:obj:`~.LLVMOrcSymbolPredicate`/:py:obj:`~.object`):
           (undocumented)

       FilterCtx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(LLVMOrcDefinitionGeneratorRef * Result, char GlobalPrefx, LLVMOrcSymbolPredicate Filter, void * FilterCtx)


.. py:function:: LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(Result, FileName, GlobalPrefix, Filter, FilterCtx)

   Get a LLVMOrcCreateDynamicLibararySearchGeneratorForPath that will reflect
   library symbols into the JITDylib.

   On success the resulting generator is
   owned by the client. Ownership is typically transferred by adding the
   instance to a JITDylib using LLVMOrcJITDylibAddGenerator,

   The GlobalPrefix argument specifies the character that appears on the front
   of linker-mangled symbols for the target platform (e.g. '_' on MachO).
   If non-null, this character will be stripped from the start of all symbol
   strings before passing the remaining substring to dlsym.

   The optional Filter and Ctx arguments can be used to supply a symbol name
   filter: Only symbols for which the filter returns true will be visible to
   JIT'd code. If the Filter argument is null then all library symbols will
   be visible to JIT'd code. Note that the symbol name passed to the Filter
   function is the full mangled symbol: The client is responsible for stripping
   the global prefix if present.

   THIS API IS EXPERIMENTAL AND LIKELY TO CHANGE IN THE NEAR FUTURE!

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       FileName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       GlobalPrefix (:py:obj:`~.int`):
           (undocumented)

       Filter (:py:obj:`~.LLVMOrcSymbolPredicate`/:py:obj:`~.object`):
           (undocumented)

       FilterCtx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef * Result, const char * FileName, char GlobalPrefix, LLVMOrcSymbolPredicate Filter, void * FilterCtx)


.. py:function:: LLVMOrcCreateStaticLibrarySearchGeneratorForPath(Result, ObjLayer, FileName)

   Get a LLVMOrcCreateStaticLibrarySearchGeneratorForPath that will reflect
   static library symbols into the JITDylib.

   On success the resulting
   generator is owned by the client. Ownership is typically transferred by
   adding the instance to a JITDylib using LLVMOrcJITDylibAddGenerator,

   Call with the optional TargetTriple argument will succeed if the file at
   the given path is a static library or a MachO universal binary containing a
   static library that is compatible with the given triple. Otherwise it will
   return an error.

   THIS API IS EXPERIMENTAL AND LIKELY TO CHANGE IN THE NEAR FUTURE!

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       ObjLayer (:py:obj:`~.LLVMOrcOpaqueObjectLayer`/:py:obj:`~.object`):
           (undocumented)

       FileName (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateStaticLibrarySearchGeneratorForPath(LLVMOrcDefinitionGeneratorRef * Result, LLVMOrcObjectLayerRef ObjLayer, const char * FileName)


.. py:function:: LLVMOrcCreateNewThreadSafeContext()

   Create a ThreadSafeContextRef containing a new LLVMContext.

   Ownership of the underlying ThreadSafeContext data is shared: Clients
   can and should dispose of their ThreadSafeContextRef as soon as they no
   longer need to refer to it directly. Other references (e.g. from
   ThreadSafeModules) will keep the underlying data alive as long as it is
   needed.

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContext()


.. py:function:: LLVMOrcCreateNewThreadSafeContextFromLLVMContext(Ctx)

   Create a ThreadSafeContextRef from a given LLVMContext, which must not be
   associated with any existing ThreadSafeContext.

   The underlying ThreadSafeContext will take ownership of the LLVMContext
   object, so clients should not free the LLVMContext passed to this
   function.

   Ownership of the underlying ThreadSafeContext data is shared: Clients
   can and should dispose of their ThreadSafeContextRef as soon as they no
   longer need to refer to it directly. Other references (e.g. from
   ThreadSafeModules) will keep the underlying data alive as long as it is
   needed.

   Args:
       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcThreadSafeContextRef LLVMOrcCreateNewThreadSafeContextFromLLVMContext(LLVMContextRef Ctx)


.. py:function:: LLVMOrcDisposeThreadSafeContext(TSCtx)

   Dispose of a ThreadSafeContext.

   Args:
       TSCtx (:py:obj:`~.LLVMOrcOpaqueThreadSafeContext`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeThreadSafeContext(LLVMOrcThreadSafeContextRef TSCtx)


.. py:function:: LLVMOrcCreateNewThreadSafeModule(M, TSCtx)

   Create a ThreadSafeModule wrapper around the given LLVM module.

   This takes
   ownership of the M argument which should not be disposed of or referenced
   after this function returns.

   Ownership of the ThreadSafeModule is unique: If it is transferred to the JIT
   (e.g. by LLVMOrcLLJITAddLLVMIRModule) then the client is no longer
   responsible for it. If it is not transferred to the JIT then the client
   should call LLVMOrcDisposeThreadSafeModule to dispose of it.

   Args:
       M (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

       TSCtx (:py:obj:`~.LLVMOrcOpaqueThreadSafeContext`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcThreadSafeModuleRef LLVMOrcCreateNewThreadSafeModule(LLVMModuleRef M, LLVMOrcThreadSafeContextRef TSCtx)


.. py:function:: LLVMOrcDisposeThreadSafeModule(TSM)

   Dispose of a ThreadSafeModule.

   This should only be called if ownership has
   not been passed to LLJIT (e.g. because some error prevented the client from
   adding this to the JIT).

   Args:
       TSM (:py:obj:`~.LLVMOrcOpaqueThreadSafeModule`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeThreadSafeModule(LLVMOrcThreadSafeModuleRef TSM)


.. py:function:: LLVMOrcThreadSafeModuleWithModuleDo(TSM, F, Ctx)

   Apply the given function to the module contained in this ThreadSafeModule.

   Args:
       TSM (:py:obj:`~.LLVMOrcOpaqueThreadSafeModule`/:py:obj:`~.object`):
           (undocumented)

       F (:py:obj:`~.LLVMOrcGenericIRModuleOperationFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcThreadSafeModuleWithModuleDo(LLVMOrcThreadSafeModuleRef TSM, LLVMOrcGenericIRModuleOperationFunction F, void * Ctx)


.. py:function:: LLVMOrcJITTargetMachineBuilderDetectHost(Result)

   Create a JITTargetMachineBuilder by detecting the host.

   On success the client owns the resulting JITTargetMachineBuilder. It must be
   passed to a consuming operation (e.g.
   LLVMOrcLLJITBuilderSetJITTargetMachineBuilder) or disposed of by calling
   LLVMOrcDisposeJITTargetMachineBuilder.

   Args:
       Result (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcJITTargetMachineBuilderDetectHost(LLVMOrcJITTargetMachineBuilderRef * Result)


.. py:function:: LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(TM)

   Create a JITTargetMachineBuilder from the given TargetMachine template.

   This operation takes ownership of the given TargetMachine and destroys it
   before returing. The resulting JITTargetMachineBuilder is owned by the client
   and must be passed to a consuming operation (e.g.
   LLVMOrcLLJITBuilderSetJITTargetMachineBuilder) or disposed of by calling
   LLVMOrcDisposeJITTargetMachineBuilder.

   Args:
       TM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcJITTargetMachineBuilderRef LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(LLVMTargetMachineRef TM)


.. py:function:: LLVMOrcDisposeJITTargetMachineBuilder(JTMB)

   Dispose of a JITTargetMachineBuilder.

   Args:
       JTMB (:py:obj:`~.LLVMOrcOpaqueJITTargetMachineBuilder`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeJITTargetMachineBuilder(LLVMOrcJITTargetMachineBuilderRef JTMB)


.. py:function:: LLVMOrcJITTargetMachineBuilderGetTargetTriple(JTMB)

   Returns the target triple for the given JITTargetMachineBuilder as a string.

   The caller owns the resulting string as must dispose of it by calling
   LLVMDisposeMessage

   Args:
       JTMB (:py:obj:`~.LLVMOrcOpaqueJITTargetMachineBuilder`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.bytes`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       char * LLVMOrcJITTargetMachineBuilderGetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB)


.. py:function:: LLVMOrcJITTargetMachineBuilderSetTargetTriple(JTMB, TargetTriple)

   Sets the target triple for the given JITTargetMachineBuilder to the given
   string.

   Args:
       JTMB (:py:obj:`~.LLVMOrcOpaqueJITTargetMachineBuilder`/:py:obj:`~.object`):
           (undocumented)

       TargetTriple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcJITTargetMachineBuilderSetTargetTriple(LLVMOrcJITTargetMachineBuilderRef JTMB, const char * TargetTriple)


.. py:function:: LLVMOrcObjectLayerAddObjectFile(ObjLayer, JD, ObjBuffer)

   Add an object to an ObjectLayer to the given JITDylib.

   Adds a buffer representing an object file to the given JITDylib using the
   given ObjectLayer instance. This operation transfers ownership of the buffer
   to the ObjectLayer instance. The buffer should not be disposed of or
   referenced once this function returns.

   Resources associated with the given object will be tracked by the given
   JITDylib's default ResourceTracker.

   Args:
       ObjLayer (:py:obj:`~.LLVMOrcOpaqueObjectLayer`/:py:obj:`~.object`):
           (undocumented)

       JD (:py:obj:`~.LLVMOrcOpaqueJITDylib`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcObjectLayerAddObjectFile(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcJITDylibRef JD, LLVMMemoryBufferRef ObjBuffer)


.. py:function:: LLVMOrcObjectLayerAddObjectFileWithRT(ObjLayer, RT, ObjBuffer)

   Add an object to an ObjectLayer using the given ResourceTracker.

   Adds a buffer representing an object file to the given ResourceTracker's
   JITDylib using the given ObjectLayer instance. This operation transfers
   ownership of the buffer to the ObjectLayer instance. The buffer should not
   be disposed of or referenced once this function returns.

   Resources associated with the given object will be tracked by
   ResourceTracker RT.

   Args:
       ObjLayer (:py:obj:`~.LLVMOrcOpaqueObjectLayer`/:py:obj:`~.object`):
           (undocumented)

       RT (:py:obj:`~.LLVMOrcOpaqueResourceTracker`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcObjectLayerAddObjectFileWithRT(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcResourceTrackerRef RT, LLVMMemoryBufferRef ObjBuffer)


.. py:function:: LLVMOrcObjectLayerEmit(ObjLayer, R, ObjBuffer)

   Emit an object buffer to an ObjectLayer.

   Ownership of the responsibility object and object buffer pass to this
   function. The client is not responsible for cleanup.

   Args:
       ObjLayer (:py:obj:`~.LLVMOrcOpaqueObjectLayer`/:py:obj:`~.object`):
           (undocumented)

       R (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcObjectLayerEmit(LLVMOrcObjectLayerRef ObjLayer, LLVMOrcMaterializationResponsibilityRef R, LLVMMemoryBufferRef ObjBuffer)


.. py:function:: LLVMOrcDisposeObjectLayer(ObjLayer)

   Dispose of an ObjectLayer.

   Args:
       ObjLayer (:py:obj:`~.LLVMOrcOpaqueObjectLayer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeObjectLayer(LLVMOrcObjectLayerRef ObjLayer)


.. py:function:: LLVMOrcIRTransformLayerEmit(IRTransformLayer, MR, TSM)

   Dispose of an ObjectLayer.

   Args:
       IRTransformLayer (:py:obj:`~.LLVMOrcOpaqueIRTransformLayer`/:py:obj:`~.object`):
           (undocumented)

       MR (:py:obj:`~.LLVMOrcOpaqueMaterializationResponsibility`/:py:obj:`~.object`):
           (undocumented)

       TSM (:py:obj:`~.LLVMOrcOpaqueThreadSafeModule`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcIRTransformLayerEmit(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcMaterializationResponsibilityRef MR, LLVMOrcThreadSafeModuleRef TSM)


.. py:function:: LLVMOrcIRTransformLayerSetTransform(IRTransformLayer, TransformFunction, Ctx)

   Set the transform function of the provided transform layer, passing through a
   pointer to user provided context.

   Args:
       IRTransformLayer (:py:obj:`~.LLVMOrcOpaqueIRTransformLayer`/:py:obj:`~.object`):
           (undocumented)

       TransformFunction (:py:obj:`~.LLVMOrcIRTransformLayerTransformFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcIRTransformLayerSetTransform(LLVMOrcIRTransformLayerRef IRTransformLayer, LLVMOrcIRTransformLayerTransformFunction TransformFunction, void * Ctx)


.. py:function:: LLVMOrcObjectTransformLayerSetTransform(ObjTransformLayer, TransformFunction, Ctx)

   Set the transform function on an LLVMOrcObjectTransformLayer.

   Args:
       ObjTransformLayer (:py:obj:`~.LLVMOrcOpaqueObjectTransformLayer`/:py:obj:`~.object`):
           (undocumented)

       TransformFunction (:py:obj:`~.LLVMOrcObjectTransformLayerTransformFunction`/:py:obj:`~.object`):
           (undocumented)

       Ctx (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcObjectTransformLayerSetTransform(LLVMOrcObjectTransformLayerRef ObjTransformLayer, LLVMOrcObjectTransformLayerTransformFunction TransformFunction, void * Ctx)


.. py:function:: LLVMOrcCreateLocalIndirectStubsManager(TargetTriple)

   Create a LocalIndirectStubsManager from the given target triple.

   The resulting IndirectStubsManager is owned by the client
   and must be disposed of by calling LLVMOrcDisposeDisposeIndirectStubsManager.

   Args:
       TargetTriple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcIndirectStubsManagerRef LLVMOrcCreateLocalIndirectStubsManager(const char * TargetTriple)


.. py:function:: LLVMOrcDisposeIndirectStubsManager(ISM)

   Dispose of an IndirectStubsManager.

   Args:
       ISM (:py:obj:`~.LLVMOrcOpaqueIndirectStubsManager`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeIndirectStubsManager(LLVMOrcIndirectStubsManagerRef ISM)


.. py:function:: LLVMOrcCreateLocalLazyCallThroughManager(TargetTriple, ES, ErrorHandlerAddr, LCTM)

   Dispose of an IndirectStubsManager.

   Args:
       TargetTriple (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       ES (:py:obj:`~.LLVMOrcOpaqueExecutionSession`/:py:obj:`~.object`):
           (undocumented)

       ErrorHandlerAddr (:py:obj:`~.int`):
           (undocumented)

       LCTM (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcCreateLocalLazyCallThroughManager(const char * TargetTriple, LLVMOrcExecutionSessionRef ES, LLVMOrcJITTargetAddress ErrorHandlerAddr, LLVMOrcLazyCallThroughManagerRef * LCTM)


.. py:function:: LLVMOrcDisposeLazyCallThroughManager(LCTM)

   Dispose of an LazyCallThroughManager.

   Args:
       LCTM (:py:obj:`~.LLVMOrcOpaqueLazyCallThroughManager`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeLazyCallThroughManager(LLVMOrcLazyCallThroughManagerRef LCTM)


.. py:function:: LLVMOrcCreateDumpObjects(DumpDir, IdentifierOverride)

   Create a DumpObjects instance.

   DumpDir specifies the path to write dumped objects to. DumpDir may be empty
   in which case files will be dumped to the working directory.

   IdentifierOverride specifies a file name stem to use when dumping objects.
   If empty then each MemoryBuffer's identifier will be used (with a .o suffix
   added if not already present). If an identifier override is supplied it will
   be used instead, along with an incrementing counter (since all buffers will
   use the same identifier, the resulting files will be named <ident>.o,
   <ident>.2.o, <ident>.3.o, and so on). IdentifierOverride should not contain
   an extension, as a .o suffix will be added by DumpObjects.

   Args:
       DumpDir (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

       IdentifierOverride (:py:obj:`~.rocm.bindings.util.types.CStr`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMOrcDumpObjectsRef LLVMOrcCreateDumpObjects(const char * DumpDir, const char * IdentifierOverride)


.. py:function:: LLVMOrcDisposeDumpObjects(DumpObjects)

   Dispose of a DumpObjects instance.

   Args:
       DumpObjects (:py:obj:`~.LLVMOrcOpaqueDumpObjects`/:py:obj:`~.object`):
           (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       void LLVMOrcDisposeDumpObjects(LLVMOrcDumpObjectsRef DumpObjects)


.. py:function:: LLVMOrcDumpObjects_CallOperator(DumpObjects, ObjBuffer)

   Dump the contents of the given MemoryBuffer.

   Args:
       DumpObjects (:py:obj:`~.LLVMOrcOpaqueDumpObjects`/:py:obj:`~.object`):
           (undocumented)

       ObjBuffer (:py:obj:`~.rocm.bindings.util.types.Pointer`/:py:obj:`~.object`):
           (undocumented)

   Returns:
       :py:obj:`~.None`: (undocumented)

   .. rubric:: C signature

   .. code-block:: c

       LLVMErrorRef LLVMOrcDumpObjects_CallOperator(LLVMOrcDumpObjectsRef DumpObjects, LLVMMemoryBufferRef * ObjBuffer)


